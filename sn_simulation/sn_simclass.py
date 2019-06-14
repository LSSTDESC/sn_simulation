
from astropy.cosmology import w0waCDM
from importlib import import_module
import numpy as np
import time
from astropy.table import vstack, Table, Column
import os
import h5py
import multiprocessing

from sn_tools.sn_telescope import Telescope
from sn_simulation.sn_object import SN_Object
from sn_tools.sn_utils import GenerateSample
from sn_tools.observations import Observations
from scipy.interpolate import interp2d


class SN_Simulation:
    """ Main class for simulation

    Parameters
    --------------

    cosmo_par: dict
     cosmology parameters
     ex: {'Model': 'w0waCDM', 'Omega_m': 0.3,
         'Omega_l': 0.7, 'H0': 72.0, 'w0': -1.0, 'wa': 0.0}
    tel_par: dict
     telescope parameters
     ex: {'name': 'LSST', 'throughput_dir': 'LSST_THROUGHPUTS_BASELINE',
         'atmos_dir': 'THROUGHPUTS_DIR', 'airmass': 1.2, 'atmos': True, 'aerosol': False}
    sn_parameters: dict
     supernova parameters
      ex:  {'Id': 100, 'x1_color': {'type': 'fixed', 'min': [-2.0, 0.2], 'max': [0.2, 0.2], 'rate': 'JLA'}, 'z': {'type': 'uniform', 'min': 0.01, 'max': 0.9, 'step': 0.05, 'rate': 'Perrett'}, 'daymax': {
          'type': 'unique', 'step': 1}, 'min_rf_phase': -20.0, 'max_rf_phase': 60.0, 'absmag': -19.0906, 'band': 'bessellB', 'magsys': 'vega', 'differential_flux': False}
    save_status: bool
     to save (True) or not (False) LC on disk
    outdir: str
     output directory (used if save_status=True)
    prodid: str
     production id (ie output file name)
    simu_config: dict
     Simulator configuration
     ex: {'name': 'sn_simulator.sn_cosmo', 'model': 'salt2-extended',
         'version': 1.0, 'Reference File': 'LC_Test_today.hdf5'}
    x0_norm: numpy array
     array with (x1,color,x0_norm) values
    display_lc: bool,opt
     to display (True) or not (False-default) LC during generation
    time_display: float,opt
     duration of display LC window persistency (sec)
    area: float,opt
     observed area (default: 9.6 deg2)
    mjdCol: str, opt
     mjd col name in observations (default: 'mjd')
    RaCol: str, opt
     RA col name in observations (default: 'pixRa')
    DecCol:str, opt
     Dec col name in observations (default: 'pixDec')
    filterCol: str, opt
     filter col name in observations (default: band')
    exptimeCol: str, opt
     exposure time  col name in observations (default: 'exptime')
    nexpCol: str, opt
     number of exposures col name in observations (default: 'numExposures')
    m5Col: str, opt
     5-sigma depth col name in observations (default: 'fiveSigmaDepth')
    seasonCol: str, opt
     season col name in observations (default: 'season')
    seeingEffCol: str, opt
     seeing eff col name in observations (default: 'seeingFwhmEff')
    seeingGeomCol: str, opt
     seeing geom  col name in observations (default: 'seeingFwhmGeom')
    x1color_dir: str, opt
      dir where (x1,color) distribution files are located
    nproc: int,opt
     number of multiprocess (default: 1)

    """

    def __init__(self, cosmo_par, tel_par, sn_parameters,
                 save_status, outdir, prodid,
                 simu_config, x0_norm, display_lc=False, time_display=0., area=9.6,
                 mjdCol='mjd', RaCol='pixRa', DecCol='pixDec',
                 filterCol='band', exptimeCol='exptime', nexpCol='numExposures',
                 m5Col='fiveSigmaDepth', seasonCol='season',
                 seeingEffCol='seeingFwhmEff', seeingGeomCol='seeingFwhmGeom',
                 x1colorDir='reference_files',
                 salt2Dir='SALT2_Files',
                 nproc=1):

        self.sn_parameters = sn_parameters
        self.simu_config = simu_config
        self.display_lc = display_lc
        self.time_display = time_display
        self.index_hdf5 = 100
        self.save_status = save_status
        self.mjdCol = mjdCol
        self.RaCol = RaCol
        self.DecCol = DecCol
        self.filterCol = filterCol
        self.exptimeCol = exptimeCol
        self.nexpCol = nexpCol
        self.m5Col = m5Col
        self.seasonCol = seasonCol
        self.seeingEffCol = seeingEffCol
        self.seeingGeomCol = seeingGeomCol
        self.nproc = nproc
        self.area = area
        self.salt2Dir = salt2Dir
        # generate simulation parameters
        self.gen_par = GenerateSample(sn_parameters, cosmo_par, mjdCol=self.mjdCol, area=self.area,
                                      min_rf_phase=sn_parameters['min_rf_phase'],
                                      max_rf_phase=sn_parameters['max_rf_phase'], dirFiles=x1colorDir)

        # instantiate cosmology
        self.cosmology = w0waCDM(H0=cosmo_par['H0'],
                                 Om0=cosmo_par['Omega_m'],
                                 Ode0=cosmo_par['Omega_l'],
                                 w0=cosmo_par['w0'], wa=cosmo_par['wa'])
        # instantiate telescope
        self.telescope = Telescope(name=tel_par['name'],
                                   throughput_dir=tel_par['throughput_dir'],
                                   atmos_dir=tel_par['atmos_dir'],
                                   atmos=tel_par['atmos'],
                                   aerosol=tel_par['aerosol'],
                                   airmass=tel_par['airmass'])

        # get the x0_norm values to be put on a 2D(x1,color) griddata
        self.x0_grid = x0_norm

        # if saving activated, prepare output dirs
        if self.save_status:
            self.prepareSave(outdir, prodid)

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
        - astropy table with (SNID,Ra,Dec,X1,Color,z) parameters
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
        # astropy table with (SNID,Ra,Dec,X1,Color,z) parameters
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

    def __call__(self, obs, fieldname, fieldid, season):
        """ LC simulations

        Parameters
        --------------
        obs: array
          array of observations
        fieldname: str
          name of the field
        fieldid: int
           int label of the field
        season: int
           season number

        """
        # obs = Observations(data=tab, names=self.names)
        self.fieldname = fieldname
        self.fieldid = fieldid
        if season == -1:
            seasons = np.unique(obs[self.seasonCol])
        else:
            seasons = [season]

        if self.simu_config['name'] != 'SN_Fast':
            for seas in seasons:
                self.index_hdf5_count = self.index_hdf5
                time_ref = time.time()
                idxa = obs[self.seasonCol] == seas
                obs_season = obs[idxa]
                # remove the u band
                idx = [i for i, val in enumerate(
                    obs_season[self.filterCol]) if val[-1] != 'u']
                if len(obs_season[idx]) >= 5:
                    # if self.simu_config['name'] != 'SN_Fast':
                    self.processSeason(obs_season[idx], seas)
        else:
            time_ref = time.time()
            if season != -1:
                for seas in seasons:
                    idxa = obs[self.seasonCol] == seas
                    self.processFast(obs[idxa], fieldname, fieldid)
            else:
                self.processFast(obs, fieldname, fieldid)

        print('End of simulation', time.time()-time_ref)
    """
    def __call__(self, tab,fieldname,fieldid):

        all_obs = Observations(data=tab, names=self.names)
        self.fieldname = fieldname
        self.fieldid = fieldid
        print('number of seasons',len(all_obs.seasons))
        for season in range(len(all_obs.seasons)):
            time_ref = time.time()
            obs = all_obs.seasons[season]
            # remove the u band
            idx = [i for i, val in enumerate(
                obs[self.filterCol]) if val[-1] != 'u']
            if len(obs[idx]) > 0:
                self.Process_Season(obs[idx], season)
            print('End of simulation',time.time()-time_ref)
    """

    def processSeason(self, obs, season):
        """ Generate LC for one season (multiprocessing available) and all simu parameters

        Parameters
        --------------
        obs: array
          array of observations
        season: int
          season number

        """
        gen_params = self.gen_par(obs)
        if gen_params is None:
            return
        nlc = len(gen_params)
        batch = range(0, nlc, self.nproc)
        batch = np.append(batch, nlc)

        for i in range(len(batch)-1):
            result_queue = multiprocessing.Queue()

            ida = batch[i]
            idb = batch[i+1]

            for j in range(ida, idb):
                self.index_hdf5_count += 1
                p = multiprocessing.Process(name='Subprocess-'+str(j), target=self.processSeasonSingle, args=(
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
                    self.sn_meta.append((metadata['SNID'], metadata['Ra'],
                                         metadata['Dec'], metadata['daymax'],
                                         metadata['x0'], metadata['epsilon_x0'],
                                         metadata['x1'], metadata['epsilon_x1'],
                                         metadata['color'], metadata['epsilon_color'],
                                         metadata['z'], metadata['index_hdf5'], season,
                                         self.fieldname, self.fieldid,
                                         n_lc_points, metadata['survey_area'],
                                         metadata['pixID'],
                                         metadata['pixRa'],
                                         metadata['pixDec']))

            """
            for i, val in enumerate(gen_params[:]):
            self.index_hdf5 += 1
            self.Process_Season_Single(obs,season,val)
            """

    def processSeasonSingle(self, obs, season, gen_params, index_hdf5, j=-1, output_q=None):
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
                              mjdCol=self.mjdCol, RaCol=self.RaCol,
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
                  names=['SNID', 'Ra', 'Dec', 'daymax', 'x0', 'epsilon_x0',
                         'x1', 'epsilon_x1',
                         'color', 'epsilon_color',
                         'z', 'id_hdf5', 'season',
                         'fieldname', 'fieldid',
                         'n_lc_points', 'survey_area', 'pixID', 'pixRa', 'pixDec'],
                  dtype=('i4', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8',
                         'f8', h5py.special_dtype(vlen=str), 'i4', 'S3', 'i8', 'i8', 'f8', 'i8', 'f8', 'f8')).write(
                             self.simu_out, 'summary', compression=True)

    def processFast(self, obs, fieldname, fieldid):
        """ SN fast simulator

        Parameters
        --------------
        obs: array
         array of observations
        fieldname: str
         name of the field
        fieldid: int
         int label for the field

        """

        gen_params = self.gen_par(obs)

        if gen_params is None:
            return
        # print('genpar',gen_params)
        sn_par = self.sn_parameters.copy()
        for name in ['z', 'X1', 'Color', 'DayMax']:
            sn_par[name] = gen_params[name]

        epsilon = {}

        for val in ['x0', 'x1', 'color']:
            epsilon[val] = np.asscalar(np.unique(gen_params['epsilon_'+val]))

        sn_object = SN_Object(self.simu_config['name'],
                              sn_par,
                              gen_params,
                              self.cosmology,
                              self.telescope, sn_par['Id'], self.area,
                              mjdCol=self.mjdCol, RaCol=self.RaCol,
                              DecCol=self.DecCol,
                              filterCol=self.filterCol, exptimeCol=self.exptimeCol,
                              m5Col=self.m5Col)

        module = import_module(self.simu_config['name'])
        simu = module.SN(sn_object, self.simu_config)
        ra, dec, tab = simu(obs, self.index_hdf5,
                            self.display_lc, self.time_display, gen_params)

        index_hdf5 = self.index_hdf5

        if self.save_status and tab is not None:
            for season in np.unique(tab['season']):
                for z in np.unique(tab['z']):
                    idx = tab['season'] == season
                    idx &= np.abs(tab['z']-z) < 1.e-5
                    sel = tab[idx]
                    index_hdf5 += 1
                    SNID = sn_par['Id']+index_hdf5

                    self.sn_meta.append((SNID, ra, dec, -1,
                                         -1.,
                                         epsilon['x0'],
                                         np.asscalar(np.unique(sn_par['x1'])),
                                         epsilon['x1'],
                                         np.asscalar(
                                             np.unique(sn_par['color'])),
                                         epsilon['color'],
                                         z, index_hdf5, season, fieldname, fieldid, len(sel), self.area))

                    sel.write(self.lc_out,
                              path='lc_'+str(ra)+'_'+str(dec) +
                              '_'+str(index_hdf5),
                              # path = 'lc_'+key[0],
                              append=True,
                              compression=True)
