import numpy as np
from lsst.sims.maf.metrics import BaseMetric
from sn_simulation.sn_simclass import SN_Simulation
from sn_stackers.coadd_stacker import CoaddStacker
import healpy as hp
import numpy.lib.recfunctions as rf
from astropy.cosmology import w0waCDM
from importlib import import_module
from sn_tools.sn_telescope import Telescope
from sn_simulation.sn_object import SN_Object
from sn_tools.sn_utils import SimuParameters
from sn_tools.sn_obs import season as seasoncalc
import os
import time
import multiprocessing
from astropy.table import vstack, Table, Column
import h5py


class SNSimulation(BaseMetric):
    """LC simulation wrapper class

    Parameters
    ---------------

    mjdCol: str, opt
     mjd col name in observations (default: 'observationStartMJD')
    RACol: str, opt
     RA col name in observations (default: 'fieldRA')
    DecCol:str, opt
     Dec col name in observations (default: 'fieldDec')
    filterCol: str, opt
     filter col name in observations (default: filter')
    m5Col: str, opt
     5-sigma depth col name in observations (default: 'fiveSigmaDepth')
    exptimeCol: str, opt
     exposure time  col name in observations (default: 'visitExposureTime)
    nightCol: str, opt
     night col name in observations (default: 'night')
    obsidCol: str, opt
     observation id col name in observations (default: 'observationId')
    nexpCol: str, opt
     number of exposures col name in observations (default: 'numExposures')
    visitimeCol: str, opt
     visit time col name in observations (default: 'visiTime')
    seeingEffCol: str, opt
     seeing eff col name in observations (default: 'seeingFwhmEff')
    seeingGeomCol: str, opt
     seeing geom  col name in observations (default: 'seeingFwhmGeom')
    coadd: bool, opt
     coaddition of obs (per band and per night) if set to True (default: True)
    config: dict
     configuration dict for simulation (SN parameters, cosmology, telescope, ...)
     ex: {'ProductionID': 'DD_baseline2018a_Cosmo', 'SN parameters': {'Id': 100, 'x1_color': {'type': 'fixed', 'min': [-2.0, 0.2], 'max': [0.2, 0.2], 'rate': 'JLA'}, 'z': {'type': 'uniform', 'min': 0.01, 'max': 0.9, 'step': 0.05, 'rate': 'Perrett'}, 'daymax': {'type': 'unique', 'step': 1}, 'min_rf_phase': -20.0, 'max_rf_phase': 60.0, 'absmag': -19.0906, 'band': 'bessellB', 'magsys': 'vega', 'differential_flux': False}, 'Cosmology': {'Model': 'w0waCDM', 'Omega_m': 0.3, 'Omega_l': 0.7, 'H0': 72.0, 'w0': -1.0, 'wa': 0.0}, 'Instrument': {
         'name': 'LSST', 'throughput_dir': 'LSST_THROUGHPUTS_BASELINE', 'atmos_dir': 'THROUGHPUTS_DIR', 'airmass': 1.2, 'atmos': True, 'aerosol': False}, 'Observations': {'filename': '/home/philippe/LSST/DB_Files/kraken_2026.db', 'fieldtype': 'DD', 'coadd': True, 'season': 1}, 'Simulator': {'name': 'sn_simulator.sn_cosmo', 'model': 'salt2-extended', 'version': 1.0, 'Reference File': 'LC_Test_today.hdf5'}, 'Host Parameters': 'None', 'Display_LC': {'display': True, 'time': 1}, 'Output': {'directory': 'Output_Simu', 'save': True}, 'Multiprocessing': {'nproc': 1}, 'Metric': 'sn_mafsim.sn_maf_simulation', 'Pixelisation': {'nside': 64}}
    x0_norm: array of float
     grid ox (x1,color,x0) values
    reference_lc: class
     reference lc for the fast simulation

    """

    def __init__(self, metricName='SNSimulation',
                 mjdCol='observationStartMJD', RACol='fieldRA', DecCol='fieldDec',
                 filterCol='filter', m5Col='fiveSigmaDepth', exptimeCol='visitExposureTime',
                 nightCol='night', obsidCol='observationId', nexpCol='numExposures',
                 vistimeCol='visitTime', seeingEffCol='seeingFwhmEff',
                 seeingGeomCol='seeingFwhmGeom',
                 uniqueBlocks=False, config=None, x0_norm=None, reference_lc=None, **kwargs):

        self.mjdCol = mjdCol
        self.m5Col = m5Col
        self.filterCol = filterCol
        self.RACol = RACol
        self.DecCol = DecCol
        self.exptimeCol = exptimeCol
        self.seasonCol = 'season'
        self.nightCol = nightCol
        self.obsidCol = obsidCol
        self.nexpCol = nexpCol
        self.vistimeCol = vistimeCol
        self.seeingEffCol = seeingEffCol
        self.seeingGeomCol = seeingGeomCol
        self.reference_lc = reference_lc
        self.index_hdf5 = 100

        cols = [self.RACol, self.DecCol, self.nightCol, self.m5Col, self.filterCol, self.mjdCol, self.obsidCol,
                self.nexpCol, self.vistimeCol, self.exptimeCol, self.seeingEffCol, self.seeingGeomCol, self.nightCol]
        self.stacker = None

        coadd = config['Observations']['coadd']
        if coadd:
            # cols += ['sn_coadd']
            self.stacker = CoaddStacker(mjdCol=self.mjdCol,
                                        RACol=self.RACol, DecCol=self.DecCol,
                                        m5Col=self.m5Col, nightCol=self.nightCol,
                                        filterCol=self.filterCol, numExposuresCol=self.nexpCol,
                                        visitTimeCol=self.vistimeCol, visitExposureTimeCol='visitExposureTime')
        super(SNSimulation, self).__init__(
            col=cols, metricName=metricName, **kwargs)

        # bands considered
        self.filterNames = 'ugrizy'

        # grab config file
        self.config = config

        # healpix nside and area
        self.nside = config['Pixelisation']['nside']
        self.area = hp.nside2pixarea(self.nside, degrees=True)

        # prodid
        prodid = config['ProductionID']

        # load cosmology
        cosmo_par = config['Cosmology']
        self.cosmology = w0waCDM(H0=cosmo_par['H0'],
                                 Om0=cosmo_par['Omega_m'],
                                 Ode0=cosmo_par['Omega_l'],
                                 w0=cosmo_par['w0'], wa=cosmo_par['wa'])

        # load telescope
        tel_par = config['Instrument']
        self.telescope = Telescope(name=tel_par['name'],
                                   throughput_dir=tel_par['throughput_dir'],
                                   atmos_dir=tel_par['atmos_dir'],
                                   atmos=tel_par['atmos'],
                                   aerosol=tel_par['aerosol'],
                                   airmass=tel_par['airmass'])

        # sn parameters
        self.sn_parameters = config['SN parameters']
        self.gen_par = SimuParameters(self.sn_parameters, cosmo_par, mjdCol=self.mjdCol, area=self.area,
                                      min_rf_phase=self.sn_parameters['min_rf_phase_qual'],
                                      max_rf_phase=self.sn_parameters['max_rf_phase_qual'],
                                      dirFiles=self.sn_parameters['x1_color']['dirFile'])

        # this is for output

        save_status = config['Output']['save']
        self.save_status = save_status
        outdir = config['Output']['directory']
        # if saving activated, prepare output dirs
        if self.save_status:
            self.prepareSave(outdir, prodid)

        # simulator parameter
        self.simu_config = config['Simulator']

        # LC display in "real-time"
        self.display_lc = config['Display_LC']['display']
        self.time_display = config['Display_LC']['time']

        # fieldtype, season
        self.field_type = config['Observations']['fieldtype']
        self.season = config['Observations']['season']

        self.type = 'simulation'

        # get the x0_norm values to be put on a 2D(x1,color) griddata
        self.x0_grid = x0_norm

        # number of procs to run simu here
        self.nprocs = config['Multiprocessing']['nproc']

        # SALT2DIR
        self.salt2Dir = self.sn_parameters['salt2Dir']

    def run(self, obs, slicePoint=None):
        """ LC simulations

        Parameters
        --------------
        obs: array
          array of observations

        """
        # estimate seasons
        obs = seasoncalc(obs)
        # stack if necessary
        if self.stacker is not None:
            obs = self.stacker._run(obs)
            print('stacked')
        # obs = Observations(data=tab, names=self.names)
        self.fieldname = 'unknown'
        self.fieldid = 0
        if self.season == -1:
            seasons = np.unique(obs[self.seasonCol])
        else:
            seasons = self.season

        if 'sn_fast' not in self.simu_config['name']:
            for seas in seasons:
                self.index_hdf5_count = self.index_hdf5
                time_ref = time.time()
                idxa = obs[self.seasonCol] == seas
                obs_season = obs[idxa]
                print('obs', len(obs_season))
                # remove the u band
                idx = [i for i, val in enumerate(
                    obs_season[self.filterCol]) if val[-1] != 'u']
                if len(obs_season[idx]) >= 5:
                    print('running here')
                    self.simuSeason(obs_season[idx], seas)
        else:
            time_ref = time.time()
            if season != -1:
                for seas in seasons:
                    idxa = obs[self.seasonCol] == seas
                    self.processFast(obs[idxa], fieldname,
                                     fieldid, reference_lc)
            else:
                self.processFast(obs, fieldname, fieldid, reference_lc)

        print('End of simulation', time.time()-time_ref)

    def prepareSave(self, outdir, prodid):
        """ Prepare output directories for data

        Parameters
        --------------
        outdir: str
         output directory where to copy the data
        prodid: str
         production id (label for input files)

        Returns
        ----------
        Two output files are open:
        - astropy table with (SNID,RA,Dec,X1,Color,z) parameters
        -> name: Simu_prodid.hdf5
        - astropy tables with LC
        -> name : LC_prodid.hdf5

        """

        if not os.path.exists(outdir):
            print('Creating output directory', outdir)
            os.makedirs(outdir)
        # Two files to be opened (fieldname and fieldid
        # given in the input yaml file)
        # One containing a summary of the simulation:
        # astropy table with (SNID,RA,Dec,X1,Color,z) parameters
        # -> name: Simu_prodid.hdf5
        # A second containing the Light curves (list of astropy tables)
        # -> name : LC_prodid.hdf5
        self.simu_out = outdir+'/Simu_'+prodid+'.hdf5'
        self.lc_out = outdir+'/LC_'+prodid+'.hdf5'
        self.sn_meta = []
        # and these files will be removed now (before processing)
        # if they exist (to avoid confusions)
        if os.path.exists(self.simu_out):
            os.remove(self.simu_out)
        if os.path.exists(self.lc_out):
            os.remove(self.lc_out)

    def simuSeason(self, obs, season):
        """ Generate LC for a season (multiprocessing available) and all simu parameters

        Parameters
        --------------
        obs: array
          array of observations
        season: int
          season number

        """

        gen_params = self.gen_par.Params(obs)

        if gen_params is None:
            return
        nlc = len(gen_params)
        batch = range(0, nlc, self.nprocs)
        batch = np.append(batch, nlc)

        for i in range(len(batch)-1):
            result_queue = multiprocessing.Queue()

            ida = batch[i]
            idb = batch[i+1]

            for j in range(ida, idb):
                self.index_hdf5_count += 1
                p = multiprocessing.Process(name='Subprocess-'+str(j), target=self.simuLCs, args=(
                    obs, season, gen_params[j], self.index_hdf5_count, j, result_queue))
                p.start()

            resultdict = {}
            for j in range(ida, idb):
                resultdict.update(result_queue.get())

            for p in multiprocessing.active_children():
                p.join()

            for j in range(ida, idb):

                if self.save_status:
                    metadata = resultdict[j][1]
                    n_lc_points = 0
                    if resultdict[j][0] is not None:
                        n_lc_points = len(resultdict[j][0])
                        resultdict[j][0].write(self.lc_out,
                                               path='lc_' +
                                               str(metadata['index_hdf5']),
                                               append=True,
                                               compression=True)
                    self.sn_meta.append((metadata['SNID'], metadata['RA'],
                                         metadata['Dec'],
                                         metadata['x0'], metadata['epsilon_x0'],
                                         metadata['x1'], metadata['epsilon_x1'],
                                         metadata['color'], metadata['epsilon_color'],
                                         metadata['daymax'], metadata['epsilon_daymax'],
                                         metadata['z'], metadata['index_hdf5'], season,
                                         self.fieldname, self.fieldid,
                                         n_lc_points, metadata['survey_area'],
                                         metadata['pixID'],
                                         metadata['pixRA'],
                                         metadata['pixDec'],
                                         metadata['dL']))

            """
            for i, val in enumerate(gen_params[:]):
            self.index_hdf5 += 1
            self.Process_Season_Single(obs,season,val)
            """

    def simuLCs(self, obs, season, gen_params, index_hdf5, j=-1, output_q=None):
        """ Generate LC for one season and a set of simu parameters

        Parameters
        --------------
        obs: array
          array of observations
        season: int
          season number
        gen_params: dict
           generation parameters
           ex:
        index_hdf5: int
          SN index in hdf5 output file
        j: int
          used for multiprocessing
        output_q: dict
          output for multiprocessing

        Returns
        ----------
        lc_table: astropy table
          table with LC informations (flux, time, ...)
        metadata: dict
          metadata of the simulation
        """
        sn_par = self.sn_parameters.copy()

        for name in ['z', 'x1', 'color', 'daymax']:
            sn_par[name] = gen_params[name]

        SNID = sn_par['Id']+index_hdf5
        sn_object = SN_Object(self.simu_config['name'],
                              sn_par,
                              gen_params,
                              self.cosmology,
                              self.telescope, SNID, self.area,
                              mjdCol=self.mjdCol, RACol=self.RACol,
                              DecCol=self.DecCol,
                              filterCol=self.filterCol, exptimeCol=self.exptimeCol,
                              m5Col=self.m5Col,
                              salt2Dir=self.salt2Dir,
                              x0_grid=self.x0_grid)

        module = import_module(self.simu_config['name'])
        simu = module.SN(sn_object, self.simu_config)
        # simulation
        lc_table, metadata = simu(
            obs, index_hdf5, self.display_lc, self.time_display)

        if output_q is not None:
            output_q.put({j: (lc_table, metadata)})
        else:
            return (lc_table, metadata)

    def Finish(self):
        """ Copy data to disk

        """
        if len(self.sn_meta) > 0:
            Table(rows=self.sn_meta,
                  names=['SNID', 'RA', 'Dec', 'x0', 'epsilon_x0',
                         'x1', 'epsilon_x1',
                         'color', 'epsilon_color',
                         'daymax', 'epsilon_daymax',
                         'z', 'id_hdf5', 'season',
                         'fieldname', 'fieldid',
                         'n_lc_points', 'survey_area', 'pixID', 'pixRA', 'pixDec', 'dL'],
                  dtype=('i4', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8',
                         'f8', 'f8', h5py.special_dtype(vlen=str), 'i4', 'S3', 'i8', 'i8', 'f8', 'i8', 'f8', 'f8', 'f8')).write(
                             self.simu_out, 'summary', compression=True)

    def processFast(self, obs, fieldname, fieldid, reference_lc):
        """ SN fast simulator

        Parameters
        --------------
        obs: array
         array of observations
        fieldname: str
         name of the field
        fieldid: int
         int label for the field
        reference_lc: class (see GetReference in sn_tools.sn_utils)
         set od dicts with reference values (flux, flux error, ...)


        """

        # generate simulation parameters depending on obs
        gen_params = self.gen_par(obs)

        if gen_params is None:
            return
        # get SN simulation parameters
        sn_par = self.sn_parameters.copy()
        for name in ['z', 'x1', 'color', 'daymax']:
            sn_par[name] = gen_params[name]

        # sn_object instance
        sn_object = SN_Object(self.simu_config['name'],
                              sn_par,
                              gen_params,
                              self.cosmology,
                              self.telescope, sn_par['Id'], self.area,
                              mjdCol=self.mjdCol, RACol=self.RACol,
                              DecCol=self.DecCol,
                              filterCol=self.filterCol, exptimeCol=self.exptimeCol,
                              m5Col=self.m5Col,
                              salt2Dir=self.salt2Dir,
                              x0_grid=self.x0_grid)

        # import the module as defined in the config file
        module = import_module(self.simu_config['name'])

        # SN class instance
        simu = module.SN(sn_object, self.simu_config, reference_lc)

        # perform simulation here
        df = simu(obs, self.index_hdf5, gen_params)

        index_hdf5 = self.index_hdf5

        def applyGroup(grp, x1, color, index_hdf5, SNID):

            z = np.unique(grp['z'])[0]
            daymax = np.unique(grp['daymax'])[0]

            season = np.unique(grp['season'])[0]
            pixID = np.unique(grp['healpixID'])[0]
            pixRA = np.unique(grp['pixRA'])[0]
            pixDec = np.unique(grp['pixDec'])[0]
            dL = np.unique(grp['dL'])[0]
            RA = np.round(pixRA, 3)
            Dec = np.round(pixDec, 3)

            formeta = (SNID, RA, Dec, daymax, -1., 0.,
                       x1, 0., color, 0.,
                       z, '{}_{}_{}'.format(
                           RA, Dec, index_hdf5), season, fieldname,
                       fieldid, len(grp), self.area,
                       pixID, pixRA, pixDec, dL)

            self.sn_meta.append(formeta)

            meta = dict(zip(['SNID', 'RA', 'Dec', 'x0', 'epsilon_x0',
                             'x1', 'epsilon_x1',
                             'color', 'epsilon_color',
                             'daymax', 'epsilon_daymax',
                             'z', 'id_hdf5', 'season',
                             'fieldname', 'fieldid',
                             'n_lc_points', 'survey_area', 'pixID', 'pixRA', 'pixDec', 'dL'], list(formeta)))

            # print('metadata',meta)
            tab = Table.from_pandas(grp)
            tab.meta = meta

            tab.write(self.lc_out,
                      path='lc_{}_{}_{}'.format(RA, Dec, index_hdf5),
                      # path = 'lc_'+key[0],
                      append=True,
                      compression=True)

        # now a tricky part: save results on disk
        if self.save_status:
            print('saving data on disk - yes this can be long')
            x1 = np.unique(sn_par['x1'])[0]
            color = np.unique(sn_par['color'])[0]
            groups = df.groupby(['healpixID', 'z', 'daymax', 'season'])

            for name, group in groups:
                index_hdf5 += 1
                SNID = sn_par['Id']+index_hdf5

                applyGroup(group, x1, color, index_hdf5, SNID)
