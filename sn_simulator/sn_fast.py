import numpy as np
from astropy.table import Table
import time
import pandas as pd
from sn_tools.sn_calcFast import LCfast, srand
from sn_simu_wrapper.sn_object import SN_Object
from sn_tools.sn_utils import SNTimer
from sn_tools.sn_io import dustmaps
from astropy.coordinates import SkyCoord
from dustmaps.sfd import SFDQuery


class SN(SN_Object):
    """ SN class - inherits from SN_Object
          Input parameters (as given in the input yaml file):
          - SN parameters (x1, color, daymax, z, ...)
          - simulation parameters

         Output:
         - astropy table with the simulated light curve:
               - columns : band, flux, fluxerr, snr_m5,flux_e,zp,zpsys,time
               - metadata : SNID,RA,Dec,DayMax,X1,Color,z

    """

    def __init__(self, param, simu_param, reference_lc=None, gamma=None, mag_to_flux=None, dustcorr=None, snr_fluxsec=''):
        super().__init__(param.name, param.sn_parameters, param.simulator_parameters,
                         param.gen_parameters,
                         param.cosmology, param.telescope, param.SNID, param.area, param.x0_grid,
                         param.salt2Dir,
                         mjdCol=param.mjdCol, RACol=param.RACol, DecCol=param.DecCol,
                         filterCol=param.filterCol, exptimeCol=param.exptimeCol,
                         m5Col=param.m5Col, seasonCol=param.seasonCol)

        # x1 and color are unique for this simulator
        x1 = np.unique(self.sn_parameters['x1']).item()
        color = np.unique(self.sn_parameters['color']).item()
        sn_type = self.sn_parameters['type']
        self.error_model = self.simulator_parameters['errorModel']
        #self.error_model_cut = self.simulator_parameters['errorModelCut']
        """
        # Loading reference file
        fname = '{}/LC_{}_{}_vstack.hdf5'.format(
                self.templateDir, x1, color)

        reference_lc = GetReference(
            fname, self.gammaFile, param.telescope)
        """
        self.reference_lc = reference_lc
        self.gamma = gamma
        self.mag_to_flux = mag_to_flux
        # blue and red cutoffs are taken into account in the reference files

        # SN parameters for Fisher matrix estimation
        self.param_Fisher = ['x0', 'x1', 'color', 'daymax']

        bluecutoff = self.sn_parameters['blueCutoff']
        redcutoff = self.sn_parameters['redCutoff']
        self.lcFast = LCfast(reference_lc, dustcorr, x1, color, param.telescope,
                             param.mjdCol, param.RACol, param.DecCol,
                             param.filterCol, param.exptimeCol,
                             param.m5Col, param.seasonCol,
                             lightOutput=False,
                             bluecutoff=bluecutoff,
                             redcutoff=redcutoff)

        self.premeta = dict(
            zip(['x1', 'color', 'x0', 'sn_type'], [x1, color, -1., sn_type]))
        for vv in self.param_Fisher:
            vvv = 'epsilon_{}'.format(vv)
            dd = dict(zip([vvv], [np.unique(self.gen_parameters[vvv]).item()]))
            self.premeta.update(dd)

    def __call__(self, obs, display=False, time_display=0):
        """ Simulation of the light curve
        We use multiprocessing (one band per process) to increase speed

        Parameters
        ---------
        obs: array
         array of observations
        gen_par: array
         simulation parameters
        display: bool,opt
         to display LC as they are generated (default: False)
        time_display: float, opt
         time persistency of the displayed window (defalut: 0 sec)

        Returns
        ---------
        astropy table with:
        columns: band, flux, fluxerr, snr_m5,flux_e,zp,zpsys,time
        metadata : SNID,RA,Dec,DayMax,X1,Color,z
        """

        RA = np.mean(obs[self.RACol])
        Dec = np.mean(obs[self.DecCol])
        pixRA = np.mean(obs['pixRA'])
        pixDec = np.mean(obs['pixDec'])
        pixID = np.unique(obs['healpixID']).item()
        dL = -1
        season_length = np.max(obs[self.mjdCol])-np.min(obs[self.mjdCol])
        # get ebvofMW from dust maps
        ebvofMW = self.sn_parameters['ebvofMW']
        if ebvofMW < 0.:
            # in that case ebvofMW value is taken from a map
            coords = SkyCoord(pixRA, pixDec, unit='deg')
            try:
                sfd = SFDQuery()
            except Exception as err:
                from dustmaps.config import config
                config['data_dir'] = 'dustmaps'
                import dustmaps.sfd
                dustmaps.sfd.fetch()

            sfd = SFDQuery()
            ebvofMW = sfd(coords)

        # start timer
        ti = SNTimer(time.time())
        # Are there observations with the filters?
        goodFilters = np.in1d(obs[self.filterCol],
                              np.array([b for b in 'grizy']))

        if len(obs[goodFilters]) == 0:
            return [self.nosim(ra, dec, pixRA, pixDec, pixID, season, season_length, ti, -1, ebvofMW)]

        tab_tot = self.lcFast(obs, ebvofMW, self.gen_parameters)

        # remove LC points with too high error model value
        """
        if self.error_model:
            if self.error_model_cut >0:
                idx = tab_tot['fluxerr_model']/tab_tot['flux']<= self.error_model_cut
                tab_tot = tab_tot[idx]
        """
        
        # apply dust correction here
        tab_tot = self.dust_corrections(tab_tot, pixRA, pixDec)
        
        ptime = ti.finish(time.time())['ptime'].item()
        self.premeta.update(dict(zip(['RA', 'Dec', 'pixRA', 'pixDec', 'healpixID', 'dL', 'ptime', 'status', 'ebvofMW'],
                                     [RA, Dec, pixRA, pixDec, pixID, dL, ptime, 1, ebvofMW])))

        """
        ii = tab_tot['band'] == 'LSST::z'

        sel = tab_tot[ii]
        for io, row in sel.iterrows():
            print(row[['phase', 'z', 'band', 'flux',
                       'old_flux', 'fluxerr', 'old_fluxerr']].values)
        """
        list_tables = self.transform(tab_tot)

        # if the user chooses to display the results...
        if display:
            for table_lc in list_tables:
                self.plotLC(table_lc['time', 'band',
                                     'flux', 'fluxerr', 'zp', 'zpsys'], time_display)

        return list_tables

    def transform(self, tab):
        """
        Method to transform a pandas df to a set of astropytables with metadata

        Parameters
        ---------------
        tab: pandas df
          LC points

        Returns
        -----------
        list of astropy tables with metadata

        """

        groups = tab.groupby(['z', 'daymax', 'season'])

        tab_tot = []
        for name, grp in groups:
            newtab = Table.from_pandas(grp)
            newtab.meta = dict(
                zip(['z', 'daymax', 'season'], name))
            newtab.meta.update(self.premeta)
            tab_tot.append(newtab)

        return tab_tot

    def nosim(self, RA, Dec, pixRA, pixDec, healpixID, season, season_length, ti, status, ebvofMW):
        """
        Method to construct an empty table when no simulation was not possible

        Parameters
        ---------------
        ra: float
          SN RA
        dec: float
          SN Dec
        pixRA: float
          pixel RA
        pixDec: float
          pixel Dec
        pixID: int
          healpixID
        season: int
          season of interest
        season_length: float
          season length
        ptime: float
           processing time
        status: int
          status of the processing(1=ok, -1=no simu)
        ebvofMW: float
          E(B-V)
        """
        ptime = ti.finish(time.time())['ptime'].item()
        table_lc = Table()
        # set metadata
        table_lc.meta = self.metadata(
            ra, dec, pix, area, season, season_length, ptime, snr_fluxsec, status, ebvofMW)
        return table_lc
