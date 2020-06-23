import numpy as np
from astropy.table import Table
import time
import pandas as pd
from sn_tools.sn_calcFast import LCfast, srand
from sn_wrapper.sn_object import SN_Object
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
        super().__init__(param.name, param.sn_parameters, param.gen_parameters,
                         param.cosmology, param.telescope, param.SNID, param.area, param.x0_grid,
                         param.salt2Dir,
                         mjdCol=param.mjdCol, RACol=param.RACol, DecCol=param.DecCol,
                         filterCol=param.filterCol, exptimeCol=param.exptimeCol,
                         m5Col=param.m5Col, seasonCol=param.seasonCol)

        # x1 and color are unique for this simulator
        x1 = np.unique(self.sn_parameters['x1']).item()
        color = np.unique(self.sn_parameters['color']).item()

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
        self.dustcorr = dustcorr
        # blue and red cutoffs are taken into account in the reference files

        # SN parameters for Fisher matrix estimation
        self.param_Fisher = ['x0', 'x1', 'color', 'daymax']

        self.lcFast = LCfast(reference_lc, x1, color, param.telescope,
                             param.mjdCol, param.RACol, param.DecCol,
                             param.filterCol, param.exptimeCol,
                             param.m5Col, param.seasonCol, lightOutput=False)

        self.premeta = dict(zip(['x1', 'color', 'x0', ], [x1, color, -1.]))
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

        # start timer
        ti = SNTimer(time.time())
        # Are there observations with the filters?
        goodFilters = np.in1d(obs[self.filterCol],
                              np.array([b for b in 'grizy']))

        if len(obs[goodFilters]) == 0:
            return [self.nosim(ra, dec, pixRA, pixDec, pixID, season, ti, -1)]

        tab_tot = self.lcFast(obs, self.gen_parameters)

        # apply dust correction here
        tab_tot = self.dust_corrections(tab_tot, pixRA, pixDec)

        ptime = ti.finish(time.time())['ptime'].item()
        self.premeta.update(dict(zip(['RA', 'Dec', 'pixRA', 'pixDec', 'healpixID', 'dL', 'ptime', 'status'],
                                     [RA, Dec, pixRA, pixDec, pixID, dL, ptime, 1])))

        """
        ii = tab_tot['band'] == 'LSST::z'

        sel = tab_tot[ii]
        for io, row in sel.iterrows():
            print(row[['phase', 'z', 'band', 'flux',
                       'old_flux', 'fluxerr', 'old_fluxerr']].values)
        """
        list_tables = self.transform(tab_tot)

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

        groups = tab.groupby(['z', 'daymax'])

        tab_tot = []
        for name, grp in groups:
            newtab = Table.from_pandas(grp)
            newtab.meta = dict(zip(['z', 'daymax'], name))
            newtab.meta.update(self.premeta)
            tab_tot.append(newtab)

        return tab_tot

    def nosim(self, RA, Dec, pixRA, pixDec, healpixID, season, ti, status):
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
        ptime: float
           processing time
        status: int
          status of the processing(1=ok, -1=no simu)

        """
        ptime = ti.finish(time.time())['ptime'].item()
        table_lc = Table()
        # set metadata
        table_lc.meta = self.metadata(
            ra, dec, pix, area, season, ptime, snr_fluxsec, status)
        return table_lc

    def dust_corrections(self, tab, pixRA, pixDec):
        """
        Method to apply dust corrections on flux and related data

        Parameters
        ---------------
        tab: astropy Table
          LC points to apply dust corrections on
        pixRA: float
           pixel RA
        pixDec: float
           pixel Dec
        Returns
        -----------
        tab: astropy Table
          LC points with dust corrections applied
        """

        ebvofMW = self.sn_parameters['ebvofMW']

        if ebvofMW < 0.:
            # in that case ebvofMW value is taken from a map
            coords = SkyCoord(pixRA, pixDec, unit='deg')
            try:
                sfd = SFDQuery()
            except Exception as err:
                dustmaps('dustmaps')
            sfd = SFDQuery()
            ebvofMW = sfd(coords)

        # no dust correction here
        if np.abs(ebvofMW) < 1.e-5:
            return tab

        tab['ebvofMW'] = ebvofMW

        """
        for vv in ['F_x0x0', 'F_x0x1', 'F_x0daymax', 'F_x0color', 'F_x1x1',
                   'F_x1daymax', 'F_x1color', 'F_daymaxdaymax', 'F_daymaxcolor',
                   'F_colorcolor']:
            tab[vv] *= tab['fluxerr']**2
        """
        # test = pd.DataFrame(tab)

        tab = tab.groupby(['band']).apply(
            lambda x: self.corrFlux(x)).reset_index()

        # mag correction - after flux correction
        tab['mag'] = -2.5 * np.log10(tab['flux'] / 3631.0)
        # snr_m5 correction
        tab['snr_m5'] = 1./srand(tab['gamma'], tab['mag'], tab['m5'])
        tab['magerr'] = (2.5/np.log(10.))/tab['snr_m5']
        tab['fluxerr'] = tab['flux']/tab['snr_m5']

        # tab['old_flux'] = test['flux']
        # tab['old_fluxerr'] = test['fluxerr']

        # print(toat)

        """
        for vv in ['F_x0x0', 'F_x0x1', 'F_x0daymax', 'F_x0color', 'F_x1x1',
                   'F_x1daymax', 'F_x1color', 'F_daymaxdaymax', 'F_daymaxcolor',
                   'F_colorcolor']:
            tab[vv] /= tab['fluxerr']**2
        """
        return tab

    def corrFlux(self, grp):
        """
        Method to correct flux and Fisher matrix elements for dust

        Parameters
        ---------------
        grp: pandas group
           data to process

        Returns
        ----------
        pandas grp with corrected values

        """

        corrdust = self.dustcorr[grp.name.split(':')[-1]](
            (grp['phase'], grp['z'], grp['ebvofMW']))

        for vv in ['flux', 'flux_e_sec']:
            grp[vv] *= corrdust
        for vv in ['F_x0x0', 'F_x0x1', 'F_x0daymax', 'F_x0color', 'F_x1x1',
                   'F_x1daymax', 'F_x1color', 'F_daymaxdaymax', 'F_daymaxcolor',
                   'F_colorcolor']:
            grp[vv] *= corrdust*corrdust

        return grp
