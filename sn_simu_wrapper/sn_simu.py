import numpy as np
import healpy as hp
import os
import time
import multiprocessing
import astropy
from astropy.table import Table, vstack
from astropy.cosmology import w0waCDM
from importlib import import_module
from sn_simu_wrapper.sn_object import SN_Object
from sn_tools.sn_utils import SimuParameters, multiproc
from sn_tools.sn_obs import season as seasoncalc
from sn_tools.sn_calcFast import GetReference, LoadDust
from sn_tools.sn_stacker import CoaddStacker
import numpy.lib.recfunctions as rf
import pandas as pd
# import tracemalloc


class SNSimu_Params:
    def __init__(self, mjdCol,
                 RACol, DecCol,
                 filterCol, m5Col,
                 exptimeCol,
                 nightCol, obsidCol,
                 nexpCol,
                 vistimeCol, seeingEffCol,
                 airmassCol,
                 skyCol, moonCol,
                 seeingGeomCol, config, x0_norm):
        """
        Class to load simulation parameters

        Parameters
        ----------
        config : dict
            dict of parameters

        Returns
        -------
        None.

        """
        # data columns
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
        self.airmassCol = airmassCol
        self.skyCol = skyCol
        self.moonCol = moonCol

        # load stacker
        self.stacker = self.load_stacker(config['Observations']['coadd'])

        # bands considered
        self.filterNames = 'grizy'

        # grab config file
        self.config = config

        # healpix nside and area
        self.nside = config['Pixelisation']['nside']
        self.area = hp.nside2pixarea(self.nside, degrees=True)

        # prodid
        self.prodid = config['ProductionIDSimu']

        # load cosmology
        self.cosmology = self.load_cosmology(config['Cosmology'])

        # sn parameters
        self.sn_parameters = config['SN']

        """
        dirFiles = None
        if 'modelPar' in self.sn_parameters.keys():
            dirFiles = self.sn_parameters['modelPar']['dirFile']
        """
        self.gen_par = SimuParameters(self.sn_parameters, config['Cosmology'],
                                      mjdCol=self.mjdCol, area=self.area,
                                      web_path=config['WebPathSimu'])

        # simulator parameters
        self.simulator_parameters = config['Simulator']

        # simu params from file
        self.simuParamsFile = self.simu_params_from_file(
            self.sn_parameters['simuFile'])

        # this is for output

        save_status = config['OutputSimu']['save']
        self.save_status = save_status
        clean_status = config['OutputSimu']['clean']
        self.clean_status = clean_status

        self.outdir = config['OutputSimu']['directory']
        self.throwaway_empty = config['OutputSimu']['throwempty']
        self.throwafterdump = config['OutputSimu']['throwafterdump']
        # number of procs to run simu here
        self.nprocs = config['MultiprocessingSimu']['nproc']

        # simulator parameter
        self.simu_config = config['Simulator']

        # reference files
        self.reffiles = config['ReferenceFiles']

        # load zp vs airmass
        """
        self.zp_airmass = self.load_zp(config['WebPathSimu'],
                                       self.reffiles['zpDir'],
                                       self.reffiles['zpFile'])
        """
        # LC display in "real-time"

        self.display_lc = config['Display']['LC']['display']
        self.time_display = config['Display']['LC']['time']

        # fieldtype
        self.field_type = config['Observations']['fieldtype']

        # seasons
        self.season = self.load_season(config['Observations']['season'])

        self.type = 'simulation'

        # get the x0_norm values to be put on a 2D(x1,color) griddata
        self.x0_grid = x0_norm

        # SALT2DIR
        self.salt2Dir = self.sn_parameters['salt2Dir']

        # hdf5 index
        self.index_hdf5 = 100

        # load reference LC if simulator is sn_fast
        self.reference_lc = None
        # self.gamma = None
        # self.mag_to_flux = None
        self.dustcorr = None
        web_path = config['WebPathSimu']
        self.error_model = self.simu_config['errorModel']

        if 'sn_fast' in self.simu_config['name']:
            self.reference_lc, self.dustcorr = self.load_for_snfast(web_path)
        """
        else:
            gammas = LoadGamma(
                'grizy',  self.reffiles['GammaDir'],
                self.reffiles['GammaFile'],
                web_path)
            self.gamma = gammas.gamma
            self.mag_to_flux = gammas.mag_to_flux
        """
        self.filterNames = ['g', 'r', 'i', 'z', 'y']

        self.nprocdict = {}
        self.simu_out = {}
        self.lc_out = {}
        self.SNID = {}
        self.sn_meta = {}

        # load the instrument(telescope)
        from sn_telmodel.sn_telescope import load_telescope_from_config
        self.telescope = load_telescope_from_config(config['InstrumentSimu'])
        # estimate zp vs airmass
        self.zp_from_config(config['InstrumentSimu'])

    def simu_params_from_file(self, simuFile):
        """
        Method to grab simu parameters from file

        Parameters
        ----------
        simuFile : str
            simu file Name.

        Returns
        -------
        numpy array 
            array with simu parameters

        """
        # sn simu parameters from file
        df = pd.DataFrame()
        simuFile = self.sn_parameters['simuFile']
        if simuFile != 'None':
            df = pd.read_hdf(simuFile)

        # complete df with other simulation parameters

        ccols = ['healpixID', 'season', 'z', 'daymax', 'x1', 'color',
                 'epsilon_x0', 'epsilon_x1',
                 'epsilon_color', 'epsilon_daymax', 'SNID']
        ccolsb = ['minRFphase', 'maxRFphase',
                  'minRFphaseQual', 'maxRFphaseQual']

        for vv in ccolsb:
            df[vv] = self.sn_parameters[vv]

        return df[ccols+ccolsb].to_records(index=False)

    def zp_from_config(self, config):
        """
        Method to estimate zp vs airmass

        Parameters
        ----------
        config : dict
            Telescope config.

        Returns
        -------
        None.

        """

        from sn_telmodel.sn_telescope import Zeropoint_airmass
        tel_dir = config['telescope']['dir']
        tel_tag = config['telescope']['tag']
        through_dir = config['throughputDir']
        atmos_dir = config['atmosDir']
        airmass = config['airmass']
        aerosol = config['aerosol']
        zp = Zeropoint_airmass(tel_dir=tel_dir,
                               through_dir=through_dir,
                               atmos_dir=atmos_dir,
                               tag=tel_tag,
                               aerosol=aerosol)

        self.zp_airmass = zp.get_fit_params()

    def load_for_snfast(self, web_path):
        """
        Method to load reference files for sn_fast

        Parameters
        ----------
        web_path : str
              web dir where files are located.

        Returns
        -------
        reference_lc : astropy table
          lc reference files
        dustcorr : astropy table
          dust correction data.

        """
        n_to_load = 1
        templateDir = self.reffiles['TemplateDir']
        gammaDir = self.reffiles['GammaDir']
        gammaFile = self.reffiles['GammaFile']
        dustDir = self.reffiles['DustCorrDir']

        # x1 and color are unique for this simulator
        x1 = self.sn_parameters['x1']['min']
        color = self.sn_parameters['color']['min']
        ebvofMW = self.sn_parameters['ebvofMW']
        sn_model = self.simulator_parameters['model']
        sn_version = self.simulator_parameters['version']

        cutoff = 'cutoff'
        if self.error_model:
            cutoff = 'error_model'
        lcname = 'LC_{}_{}_{}_{}_{}_ebvofMW_{}_vstack.hdf5'.format(
            x1, color, cutoff, sn_model, sn_version, ebvofMW)
        lcname = 'LC_{}_{}_{}_{}_{}_ebvofMW_{}_vstack.hdf5'.format(
            x1, color, cutoff, sn_model, sn_version, ebvofMW)
        dustFile = 'Dust_{}_{}_{}.hdf5'.format(
            x1, color, cutoff)

        print('loading reference files')
        time_ref = time.time()

        result_queue = multiprocessing.Queue()
        p = multiprocessing.Process(name='Subprocess-0',
                                    target=self.loadReference, args=(
                                        templateDir, lcname, gammaDir,
                                        gammaFile, web_path, 0,
                                        result_queue))
        p.start()

        if np.abs(ebvofMW) > 1.e-5:
            n_to_load = 2
            print('loading dust files')
            pb = multiprocessing.Process(
                name='Subprocess-1', target=self.loadDust,
                args=(dustDir, dustFile, web_path, 1, result_queue))
            pb.start()

        resultdict = {}
        for j in range(n_to_load):
            resultdict.update(result_queue.get())

        for p in multiprocessing.active_children():
            p.join()

        reference_lc = resultdict[0]
        dustcorr = None
        if n_to_load > 1:
            dustcorr = resultdict[1]

        return reference_lc, dustcorr

    def load_season(self, seasons):
        """
        Method to get the list of seasons

        Parameters
        ----------
        seasons : str
              list of seasons.

        Returns
        -------
        season : list(int)
           list of seasons to process

        """
        if '-' not in seasons or seasons[0] == '-':
            season = list(map(int, seasons.split(',')))
        else:
            seasl = seasons.split('-')
            seasmin = int(seasl[0])
            seasmax = int(seasl[1])
            season = list(range(seasmin, seasmax+1))

        return season

    def load_stacker(self, coadd=False):
        """
        Method to load stacker class

        Parameters
        ----------
        coadd : bool, optional
              to stack or not. The default is False.

        Returns
        -------
        stacker : class
              CoaddStacker class instance.

        """

        stacker = None

        if coadd:
            stacker = CoaddStacker(col_sum=[self.nexpCol, self.vistimeCol,
                                            'visitExposureTime'],
                                   col_mean=[self.mjdCol,
                                             self.RACol, self.DecCol,
                                             'pixRA',
                                             'pixDec', 'healpixID',
                                             'season', 'airmass'],
                                   col_median=[
                                       'sky', 'moonPhase', 'seeingFwhmEff',
                                       'lsst_start'],
                                   col_group=[
                                       self.filterCol, self.nightCol],
                                   col_coadd=self.m5Col,
                                   col_visit='visitExposureTime')
        return stacker

    def load_cosmology(self, cosmo_par):
        """
        Method to load cosmology parameters

        Parameters
        ----------
        cosmo_par : dict
              cosmology parameters

        Returns
        -------
        cosmology class

        """

        cosmology = w0waCDM(H0=cosmo_par['H0'],
                            Om0=cosmo_par['Om'],
                            Ode0=cosmo_par['Ol'],
                            w0=cosmo_par['w0'], wa=cosmo_par['wa'])
        return cosmology

    def loadReference(self, templateDir, lcname, gammaDir,
                      gammaFile, web_path, j=-1, output_q=None):
        """
        Method to load reference files (lc and gamma)

        Parameters
        ----------
        templateDir : str
              template dir.
        lcname : str
              lc reference name.
        gammaDir : str
              gamma loc dir.
        gammaFile : str
              gamma file name.
        web_path : str
              web path of original files.
        j : int, optional
              internal tag for multiproc. The default is -1.
        output_q : multiprocessing queue, optional
              queue for multiprocessing. The default is None.

        Returns
        -------
        None or multiprocessing queue.

        """

        reference_lc = GetReference(templateDir,
                                    lcname, gammaDir, gammaFile, web_path)

        if output_q is not None:
            output_q.put({j: reference_lc})
        else:
            return None

    def loadDust(self, dustDir, dustFile, web_path, j=-1, output_q=None):
        """
        Method to load dust files

        Parameters
        ----------
        dustDir : str
              dust location dir.
        dustFile : str
              dust file name.
        web_path : str
              web path of original files.
        j : int, optional
              internal tag for multiproc. The default is -1.
        output_q : multiprocessing queue, optional
              queue for multiprocessing. The default is None.

        Returns
        -------
        None or multiprocessing queue.

        """

        dustcorr = LoadDust(dustDir, dustFile, web_path).dustcorr

        if output_q is not None:
            output_q.put({j: dustcorr})
        else:
            return None

    def load_zp(self, web_path, templateDir, fName):
        """
        Method to load zp_airmass results

        Parameters
        ----------
        web_path : str
              web path of original files.
        templateDir : str
              location dir
        fName : str
              file name

        Returns
        -------
        res : record array
              array with data.

        """

        from sn_tools.sn_io import check_get_file
        check_get_file(web_path, templateDir, fName)
        fullName = '{}/{}'.format(templateDir, fName)

        res = np.load(fullName)

        return res


class SNSimulation(SNSimu_Params):
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
     configuration dict for simulation (SN parameters, cosmology, telescope,..)
    x0_norm: array of float
     grid ox (x1,color,x0) values

    """

    def __init__(self, metricName='SNSimulation',
                 mjdCol='observationStartMJD',
                 RACol='fieldRA', DecCol='fieldDec',
                 filterCol='filter', m5Col='fiveSigmaDepth',
                 exptimeCol='visitExposureTime',
                 nightCol='night', obsidCol='observationId',
                 nexpCol='numExposures',
                 vistimeCol='visitTime', seeingEffCol='seeingFwhmEff',
                 airmassCol='airmass',
                 skyCol='sky', moonCol='moonPhase',
                 seeingGeomCol='seeingFwhmGeom',
                 uniqueBlocks=False, config=None, x0_norm=None, **kwargs):
        super().__init__(mjdCol=mjdCol,
                         RACol=RACol, DecCol=DecCol,
                         filterCol=filterCol, m5Col=m5Col,
                         exptimeCol=exptimeCol,
                         nightCol=nightCol, obsidCol=obsidCol,
                         nexpCol=nexpCol,
                         vistimeCol=vistimeCol, seeingEffCol=seeingEffCol,
                         airmassCol=airmassCol,
                         skyCol=skyCol, moonCol=moonCol,
                         seeingGeomCol=seeingGeomCol,
                         config=config, x0_norm=x0_norm)

    def run(self, obs, slicePoint=None, imulti=0):
        """ LC simulations

        Parameters
        --------------
        obs: array
          array of observations

        """
        iproc = 1

        # select filters
        goodFilters = np.in1d(obs[self.filterCol], self.filterNames)
        obs = obs[goodFilters]

        # estimate seasons
        obs = seasoncalc(obs, season_gap=80., force_calc=True)

        # check the number of seasons
        # if too low get seasons using clusters

        nseasons = len(np.unique(obs['season']))
        if nseasons <= 8:
            obs = self.get_season_from_cluster(obs)

            """
                delta_max = self.get_delta_per_season(obs)
                # obs = rf.drop_fields(obs, 'season')
                print('delta_max', delta_max)
                obs = seasoncalc(obs, season_gap=65., force_calc=True)
                """
        # save these obs.
        # np.save('obs_pixel.npy', obs)

        # plot seasons
        # self.plot_seasons(obs)

        if len(obs) == 0:
            return None

        # stack if necessary
        if self.stacker is not None:
            obs = self.stacker._run(obs)

        self.fieldname = 'unknown'
        self.fieldid = 0
        try:
            iter(self.season)
        except TypeError:
            self.season = [self.season]

        if self.season == [-1]:
            seasons = np.unique(obs[self.seasonCol])
        else:
            seasons = self.season

        time_ref = time.time()
        """
        tracemalloc.start()
        start = tracemalloc.take_snapshot()
        """

        """
        print('before hhh', len(obs), obs.dtype.names,
              np.unique(obs['season']))
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(obs['observationStartMJD'], obs['airmass'], 'ko')
        plt.show()
        """
        # select obs corresponding to seasons
        idx = np.in1d(obs['season'], seasons)
        obs = obs[idx]

        if len(obs) == 0:
            return None
        # get simulation parameters

        gen_params = self.get_all_gen_params(obs, seasons)

        list_lc = []

        if gen_params is not None:
            print('NLC to simulate:', len(gen_params),
                  np.unique(obs['healpixID']))

            # LC simulation using multiprocessing
            par = {}
            par['obs'] = obs
            par['nspectra'] = self.sn_parameters['nspectra']
            """
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.plot(obs['observationStartMJD'], obs['filter'], 'ko')
            print(np.max(np.diff(obs['observationStartMJD'])))
            plt.show()
            """

            list_lc = multiproc(gen_params, par, self.simuLoop, self.nprocs)

        """
        top_stats = snapshot.statistics('lineno')

        print("[ Top 10 ]")
        for stat in top_stats[:10]:
            print(stat)
        """

        if list_lc:
            return list_lc

        return None

    def get_season_from_cluster(self, obs):
        """
        Method to estimate seasons from clusters

        Parameters
        ----------
        obs : numpy array
            Data to process.

        Returns
        -------
        obs : numpy array
            Original data plus season col.

        """

        from sn_tools.sn_clusters import makeClusters, anaClusters
        nclusters = 10
        obs.sort(order=self.mjdCol)

        nobs = len(obs)
        if nobs < nclusters:
            nclusters = nobs
        points, clus, labels = makeClusters(
            nclusters, obs, 'pixRA', self.mjdCol)

        dfcluster = anaClusters(
            nclusters, obs, points, clus, labels, 'pixRA', self.mjdCol)

        dfcluster = dfcluster.sort_values(by=['pixRA', self.mjdCol])
        dfclmean = dfcluster.groupby(
            'clusId')[self.mjdCol].mean().reset_index()
        dfclmean = dfclmean.sort_values(by=[self.mjdCol]).reset_index()
        dfclmean['season'] = dfclmean.index+1

        dfcluster = dfcluster.merge(dfclmean[['clusId', 'season']],
                                    left_on=['clusId'],
                                    right_on=['clusId'],
                                    suffixes=('', ''))
        seasons = dfcluster['season'].to_list()
        obs.sort(order=['pixRA', self.mjdCol])
        obs = rf.drop_fields(obs, 'season')
        obs = rf.append_fields(obs, 'season', seasons)

        return obs

    def get_delta_per_season(self, obs, vara='observationStartMJD'):
        """
        Method to get info on data/season

        Parameters
        ----------
        obs : numpy array
            Data to process.
        vara : str, optional
            col to get info from. The default is 'observationStartMJD'.

        Returns
        -------
        float
            Max delta_MJD (over seasons)

        """

        delta_max = []
        for seas in np.unique(obs['season']):
            idx = obs['season'] == seas
            sel = obs[idx]
            season_length = np.max(sel[vara])-np.min(sel[vara])
            if len(sel) >= 2:
                delta_max.append(np.max(np.diff(sel[vara])))
                print(seas, np.max(np.diff(sel[vara])), season_length)

        return np.max(delta_max)

    def plot_seasons(self, obs):
        """
        Method to plot seasons

        Parameters
        ----------
        obs : array
            Data to process.

        Returns
        -------
        None.

        """

        print('seasons', np.unique(obs['season']))
        vara = 'observationStartMJD'
        varb = 'fiveSigmaDepth'
        import matplotlib.pyplot as plt
        plt.plot(obs[vara],
                 obs[varb], 'ko', mfc='None')
        for seas in np.unique(obs['season']):
            idx = obs['season'] == seas
            sel = obs[idx]
            season_length = np.max(sel[vara])-np.min(sel[vara])
            if len(sel) >= 2:
                # print(sel[[vara, self.filterCol, 'fieldRA', 'fieldDec']])
                print(seas, np.max(np.diff(sel[vara])), season_length,
                      np.min(sel[vara]), np.max(sel[vara]))

            plt.plot(sel[vara],
                     sel[varb], marker='*', linestyle='None')

        plt.show()

    def get_all_gen_params(self, obs, seasons):
        """
        Method to get simu parameters for all seasons

        Parameters
        ----------
        obs: array
           observations.
        seasons: list(int)
          list of seasons

        Returns
        -------
        array
         simulation parameters.

        """

        gp = None
        for seas in seasons:

            if len(self.simuParamsFile) == 0:
                gen_pars = self.gen_params_from_season(obs, seas)
            else:
                gen_pars = self.gen_params_from_file(obs, seas)

            if gen_pars is None:
                continue

            if gp is None:
                gp = gen_pars
            else:
                gp = np.concatenate((gp, gen_pars))

        return gp

    def gen_params_from_file(self, obs, seas):
        """
        Method to grab simu parameters from input file

        Parameters
        ----------
        obs : numpy array
            array of observations.
        seas : int
            season to process.

        Returns
        -------
        sel : numpy array
            array of simu parameters.

        """

        healpixID = np.unique(obs['healpixID'])

        idx = self.simuParamsFile['healpixID'] == healpixID
        idx &= self.simuParamsFile['season'] == seas

        sel = self.simuParamsFile[idx]

        return sel

    def gen_params_from_season(self, obs, seas):
        """
        Method to grab simu params (estimated from obs) for a season 

        Parameters
        ----------
        obs : numpy array
            Observations
        seas : int
            season of observation.

        Returns
        -------
        gen_pars : numpy array
            simulation parameters.

        """

        idxa = obs[self.seasonCol] == seas
        obs_season = obs[idxa]
        gen_pars = self.gen_par.simuparams(obs_season)

        if gen_pars is None:
            return gen_pars

        print('hello gen_pars', gen_pars)
        print('hello gen_pars', gen_pars.dtype.names)

        gen_pars = rf.append_fields(gen_pars,
                                    'season',
                                    [seas]*len(gen_pars))

        return gen_pars

    def simuLoop(self, gen_params, params, j=0, output_q=None):
        """
        Method to simulate LC (llop, multiproc)

        Parameters
        ----------
        gen_params : array
            simulation parameters.
        params : dict
            parameters to use.
        j : int, optional
            internal tag for multiproc. The default is 0.
        output_q : multiprocessing queue, optional
            container for output if multiproc used. The default is None.

        Returns
        -------
        list(astropy table)
            list of simulated LC.

        """

        obs = params['obs']

        lsst_start = -1
        if 'lsst_start' in obs.dtype.names:
            lsst_start = np.median(obs['lsst_start'])

        nspectra = params['nspectra']
        simu_out, lc_out = None, None

        if self.save_status:
            simu_out, lc_out, sed_out = self.prepareSave(
                self.outdir, self.prodid, j)

        if 'sn_fast' not in self.simu_config['name']:
            lc_list, lc_list_keep, tab_meta, sed_list = \
                self.loop_gen(obs, gen_params, j, lc_out)
        else:
            lc_list = self.simuLCs(obs, gen_params)
            if not self.throwafterdump:
                import copy
                lc_list_keep = copy.deepcopy(lc_list)

        if self.save_status:
            if nspectra <= 0:
                if len(lc_list) > 0:
                    self.dump(lc_list, lc_out)
                    lc_list = []
                tab_meta.meta['lc_dir'] = self.outdir
                tab_meta.meta['lc_fileName'] = lc_out.split('/')[-1]
                tab_meta.meta['lsst_start'] = lsst_start
                self.write_meta(tab_meta, simu_out)
            else:
                self.dump_df(tab_meta, simu_out, lc_list,
                             lc_out, sed_list, sed_out)

        if output_q is not None:
            return output_q.put({j: lc_list_keep})
        else:
            return lc_list_keep

    def dump_df(self, tab_meta, simu_out, lc_list, lc_out, sed_list, sed_out):
        """
        Method to dump output data in pandas df
        using format defined by N. Regnault
        to account for Spectra production

        Parameters
        ----------
        tab_meta : astropy table
            Metadata.
        simu_out : str
            output path (full) for metadata.
        lc_list : list(astropy table)
            list of light curves.
        lc_out : str
            output path (full) for lc.
        sed_list : list(astropy table)
            list of spectra.
        sed_out : str
            output path (full) for spectra.

        Returns
        -------
        None.

        """
        import pandas as pd

        # meta data or Sn_data
        df_meta = tab_meta.to_pandas()
        rename_dict = dict(zip(['SNID', 'daymax', 'color', 'ebvofMW'], [
            'sn', 'tmax', 'col', 'ebv']))
        df_meta = df_meta.rename(columns=rename_dict)
        df_meta['valid'] = 1
        df_meta['IAU'] = 0
        cols = ['sn', 'z', 'tmax', 'x0', 'x1', 'col', 'ebv', 'valid', 'IAU']
        df_meta[cols].to_hdf(simu_out, key='sn_data')
        # print(df_meta[cols])

        # light curves
        df_lc = pd.DataFrame()
        for io, lc in enumerate(lc_list):
            lca = lc.to_pandas()
            snid = lc.meta['SNID']
            lca['sn'] = snid
            lca['lc'] = 'lc_{}'.format(io)
            df_lc = pd.concat((df_lc, lca))

        df_lc['magsys'] = 'AB'
        df_lc['valid'] = 1
        rename_dict = dict(zip(['time', 'sky', 'seeingFwhmEff'],
                               ['mjd', 'mag_sky', 'seeing']))

        df_lc = df_lc.rename(columns=rename_dict)
        cols = ['sn', 'mjd', 'flux', 'fluxerr', 'band', 'magsys', 'exptime',
                'valid', 'lc', 'zp', 'mag_sky', 'seeing']
        df_lc[cols].to_hdf(lc_out, key='lc_data')

        # spectra
        df_spectra = pd.DataFrame()
        for sed in sed_list:
            df_spectra = pd.concat((df_spectra, sed.to_pandas()))

        cols = ['sn', 'mjd', 'wavelength', 'flux',
                'fluxerr', 'valid', 'spec', 'exptime']

        df_spectra.to_hdf(sed_out, key='spec_data')

    def loop_gen(self, obs, gen_params, j, lc_out):
        """
        Method to generate LCs by looping on genparams

        Parameters
        ----------
        obs : array
            data to process (observations).
        gen_params : array
            simulation parameters.
        j : int
            tag for SNID and output file.
        lc_out : str
            output file name.

        Returns
        -------
        lc_list : TYPE
            DESCRIPTION.
        lc_list_keep : TYPE
            DESCRIPTION.
        tab_meta : TYPE
            DESCRIPTION.

        """

        lc_list = []
        lc_list_keep = []
        sed_list = []
        isn = 0
        tab_meta = Table()

        for genpar in gen_params:
            isn += 1
            season = genpar['season']
            idx = obs['season'] == season
            obs_season = obs[idx]
            lc, sed = self.simuLCs(obs_season, genpar)
            if len(lc) == 0:
                continue

            lc = lc[0]

            hpix = int(np.mean(obs['healpixID']))
            isn_str = str(isn)

            if 'SNID' in genpar.dtype.names:
                sn_id = genpar['SNID']
            else:
                sn_id = 'SN_{}_{}_{}_{}'.format(
                    str(hpix).zfill(7), str(season).zfill(2), isn_str.zfill(5), j)

            lc.meta['SNID'] = sn_id
            lc_list += [lc]
            if sed:
                sed = sed[0]
                sed['sn_id'] = sn_id
                sed_list += [sed]

            if not self.throwafterdump:
                lc_list_keep += [lc]
            # every xx SN: dump to file
            if self.save_status:
                if len(lc_list) >= 100:
                    self.dump(lc_list, lc_out)
                    lc_list = []
                tab_meta = vstack([tab_meta, Table(rows=[lc.meta])])

        return lc_list, lc_list_keep, tab_meta, sed_list

    def prepareSave(self, outdir, prodid, iproc):
        """ Prepare output directories for data

        Parameters
        --------------
        outdir: str
         output directory where to copy the data
        prodid: str
         production id(label for input files)
        iproc: int
          internal tag for multiprocessing

        Returns
        ----------
        Two output files are open:
        - astropy table with (SNID, RA, Dec, X1, Color, z) parameters
        -> name: Simu_prodid.hdf5
        - astropy tables with LC
        -> name: LC_prodid.hdf5

        """

        if not os.path.exists(outdir):
            print('Creating output directory', outdir)
            os.makedirs(outdir)
        # Two files  to be opened - tagged by iproc
        # One containing a summary of the simulation:
        # astropy table with (SNID,RA,Dec,X1,Color,z) parameters
        # -> name: Simu_prodid.hdf5
        # A second containing the Light curves (list of astropy tables)
        # -> name : LC_prodid.hdf5

        simu_out = '{}/Simu_{}_{}.hdf5'.format(
            outdir, prodid, iproc)
        lc_out = '{}/LC_{}_{}.hdf5'.format(outdir, prodid, iproc)
        sed_out = '{}/Spectra_{}_{}.hdf5'.format(outdir, prodid, iproc)

        if self.clean_status:
            self.check_del(simu_out)
            self.check_del(lc_out)
            self.check_del(sed_out)

        return simu_out, lc_out, sed_out

    def check_del(self, fileName):
        """
        Method to remove a file if already exist

        Parameters
        ----------
        fileName: str
          file to remove(full path)

        """
        if os.path.exists(fileName):
            os.remove(fileName)

    def simuLCs(self, obs, gen_params):
        """ Generate LC for one season and a set of simu parameters

        Parameters
        --------------
        obs: array
          array of observations
        gen_params: dict
           generation parameters

        Returns
        ----------
        lc_table: astropy table
          table with LC informations(flux, time, ...)
        """
        sn_par = self.sn_parameters.copy()
        simulator_par = self.simulator_parameters.copy()

        for name in ['z', 'x1', 'color', 'daymax']:
            if name in gen_params.dtype.names:
                sn_par[name] = gen_params[name]

        SNID = sn_par['Id']
        sn_object = SN_Object(self.simu_config['name'],
                              sn_par,
                              simulator_par,
                              gen_params,
                              self.cosmology,
                              self.telescope,
                              self.zp_airmass,
                              SNID, self.area,
                              x0_grid=self.x0_grid,
                              salt2Dir=self.salt2Dir,
                              mjdCol=self.mjdCol,
                              RACol=self.RACol,
                              DecCol=self.DecCol,
                              filterCol=self.filterCol,
                              exptimeCol=self.exptimeCol,
                              m5Col=self.m5Col)

        module = import_module(self.simu_config['name'])
        simu = module.SN(sn_object, self.simu_config,
                         self.reference_lc, self.dustcorr)
        # simulation - this is supposed to be a list of astropytables
        lc_table = simu(obs, self.display_lc, self.time_display)

        seds = []
        nspectra = self.sn_parameters['nspectra']
        if nspectra > 0:
            seds = simu.SN_SED(gen_params, nspectra=nspectra)
        del simu
        del module
        return lc_table, seds

    def dump(self, lc_list, lc_out):
        """
        Method to dum lc on file

        Parameters
        ----------
        list_lc : list(astropytable)
            data to dump.
        lc_out : str
            outputfile name.

        Returns
        -------
        None.

        """
        for lc in lc_list:
            astropy.io.misc.hdf5.write_table_hdf5(
                lc, lc_out, path=lc.meta['SNID'],
                append=True, serialize_meta=True)

    def write_meta(self, meta, out_meta):
        """
        Method to dum lc on file

        Parameters
        ----------
        list_lc : list(astropytable)
            data to dump.
        lc_out : str
            outputfile name.

        Returns
        -------
        None.

        """

        path = 'meta_{}'.format(int(np.mean(meta['healpixID'])))

        astropy.io.misc.hdf5.write_table_hdf5(
            meta, out_meta, path=path,
            append=True, serialize_meta=False)

    def setIndex(self, healpixID, x1, color, z, daymax,
                 season, epsilon, SNID):
        """
        Method to set specific index

        Parameters
        ----------
        healpixID : int
            healpixID.
        x1 : float
            SN x1.
        color : float
            SN color.
        z : float
            SN redshift.
        daymax : float
            SN T0.
        season : int
            season num.
        epsilon : float
            epsilon for SN simu params.
        SNID : str
            SNID.

        Returns
        -------
        index_hdf5 : str
            resulting index.

        """

        index_hdf5 = '{}_{}_{}_{}_{}'.format(
            healpixID, z, daymax, season, SNID)

        if x1 != 'undef':
            index_hdf5 += '_{}_{}'.format(x1, color)

        # epsilon should be last!!
        index_hdf5 += '_{}'.format(epsilon)

        return index_hdf5

    def simuSeason_deprecated(self, obs, season, iproc):
        """ Generate LC for a season(multiprocessing available)
        and all simu parameters

        Parameters
        --------------
        obs: array
          array of observations
        season: int
          season number
          iproc: int
          internal tag for multiprocessing

        """

        gen_params = self.gen_par.simuparams(obs)

        if gen_params is None:
            return

        # self.simuLoop(obs, season, gen_params, iproc)

        npp = self.nprocs
        if 'sn_fast' in self.simu_config['name']:
            npp = 1

        list_lc = []

        if npp == 1:
            metadict, list_lc = self.simuLoop(obs, season, gen_params, iproc)
            if self.save_status:
                if not self.sn_meta[iproc]:
                    self.sn_meta[iproc] = metadict
                else:
                    # self.sn_meta[iproc]= self.sn_meta[iproc].update(metadict)
                    for key in metadict.keys():
                        self.sn_meta[iproc][key] += metadict[key]

        else:
            list_lc = self.multiSeason(obs, season, gen_params, iproc, npp)

        if len(list_lc):
            return list_lc

        return None

    def multiSeason_deprecated(self, obs, season, gen_params, iproc, npp):

        nlc = len(gen_params)
        batch = np.linspace(0, nlc, npp+1, dtype='int')
        print('batch for seasons', npp, batch, season)

        result_queue = multiprocessing.Queue()

        for i in range(npp):

            ida = batch[i]
            idb = batch[i+1]
            p = multiprocessing.Process(name='Subprocess',
                                        target=self.simuLoop, args=(
                                            obs, season, gen_params[ida:idb],
                                            iproc, i, result_queue))
            p.start()

        resultdict = {}
        for j in range(npp):
            resultdict.update(result_queue.get())

        for p in multiprocessing.active_children():
            p.join()

        list_lc = []
        for j in range(npp):
            metadict = resultdict[j][0]
            list_lc += resultdict[j][1]
            """
            if not self.sn_meta[iproc]:
                self.sn_meta[iproc] = metadict
            else:
                # self.sn_meta[iproc]= self.sn_meta[iproc].update(metadict)
                for key in metadict.keys():
                    self.sn_meta[iproc][key] += metadict[key]
            """
        return list_lc
        # self.save_metadata()
        """
        SNID = 100
        for j in range(npp):
            # the output is supposed to be a list of astropytables
            # for each proc: loop on the list to:
            # - get the lc
            # get the metadata

            if self.save_status:
                for lc in resultdict[j]:
                    # number of lc points
                    if lc is not None:
                        SNID += 1
                        self.writeLC(SNID, lc, season)
        """

    def complete_meta_deprecated(self, lc, meta_lc):

        epsilon = 0.
        if 'epsilon_x0' in lc.meta.keys():
            epsilon = np.int(1000*1.e8*lc.meta['epsilon_x0'])
            epsilon += np.int(100*1.e8*lc.meta['epsilon_x1'])
            epsilon += np.int(10*1.e8*lc.meta['epsilon_color'])
            epsilon += np.int(1*1.e8*lc.meta['epsilon_daymax'])

        if 'x1' not in lc.meta.keys():
            x1 = 'undef'
            color = 'undef'
        else:
            x1 = lc.meta['x1']
            color = lc.meta['color']

        SNID_tot = '{}_{}'.format(
            lc.meta['sn_type'], lc.meta['healpixID'])

        index_hdf5 = SNID_tot
        meta_lc['SNID'] = SNID_tot

        return meta_lc

    def writeLC_deprecated(self, SNID, lc, season, iproc, meta_lc):
        """
        Method to save lc on disk
        and to update metadata

        Parameters
        ---------------
        SNID: int
         Supernova ID
        lc: astropy Table
           sn light curve
        season: int
          season of observations
        iproc: int
          index for multiprocessing

        """
        # save LC on disk

        epsilon = 0.
        if 'epsilon_x0' in lc.meta.keys():
            epsilon = np.int(1000*1.e8*lc.meta['epsilon_x0'])
            epsilon += np.int(100*1.e8*lc.meta['epsilon_x1'])
            epsilon += np.int(10*1.e8*lc.meta['epsilon_color'])
            epsilon += np.int(1*1.e8*lc.meta['epsilon_daymax'])

        if 'x1' not in lc.meta.keys():
            x1 = 'undef'
            color = 'undef'
        else:
            x1 = lc.meta['x1']
            color = lc.meta['color']

        SNID_tot = '{}_{}_{}_{}'.format(
            lc.meta['sn_type'], lc.meta['healpixID'], iproc, SNID)
        """
        index_hdf5 = self.setIndex(lc.meta['healpixID'],
                                   x1, color,
                                   np.round(lc.meta['z'], 4),
                                   np.round(lc.meta['daymax'], 4),
                                   season, epsilon, SNID)
        """
        index_hdf5 = SNID_tot
        lc.meta['SNID'] = SNID_tot
        """
        idx = lc['snr_m5'] > 0.
        lc = lc[idx]
        lc.meta = {}
        """
        # print('writing',lc,lc.meta)
        """
        lc.write(self.lc_out[iproc],
                 path='lc_{}'.format(index_hdf5),
                 append=True,
                 compression=True, serialize_meta=True)
        """
        lc.write(self.lc_out[iproc],
                 path='lc_{}'.format(index_hdf5),
                 append=True,
                 compression=True)

        # build metadata dict
        n_lc_points = len(lc)
        """
        metanames = ['SNID', 'index_hdf5', 'season',
                     'fieldname', 'fieldid', 'n_lc_points', 'area']
        metavals = [SNID, index_hdf5, season,
                    self.fieldname, self.fieldid,
                    n_lc_points, self.area]
        """
        metanames = ['index_hdf5', 'season',
                     'fieldname', 'fieldid', 'n_lc_points', 'area', 'lcName']
        metavals = [index_hdf5, season,
                    self.fieldname, self.fieldid,
                    n_lc_points, self.area, self.lc_out[iproc]]
        metadict = dict(zip(metanames, metavals))
        metadict.update(lc.meta)

        # update main metadata dict

        """
        if not self.sn_meta[iproc]:
            for key in metadict.keys():
                self.sn_meta[iproc][key] = [metadict[key]]
        else:
            for key in metadict.keys():
                self.sn_meta[iproc][key].extend([metadict[key]])
        """
        if not meta_lc:
            for key in metadict.keys():
                meta_lc[key] = [metadict[key]]
        else:
            for key in metadict.keys():
                meta_lc[key].extend([metadict[key]])

    def simuLoop_deprecated(self, obs, season, gen_params,
                            iproc, j=0, output_q=None):
        """
        Method to simulate LC

        Parameters
        ---------------
        obs: numpy array
          array of observations
        season: int
          season number
        gen_params: numpy array
          array of observations
        iproc: int
          internal tag for multiprocessing
        j: int
           internal parameter for multiprocessing
        output_q: multiprocessing.Queue()

        """

        time_ref = time.time()
        list_lc = []
        list_lc_keep = []
        meta_lc = {}

        if 'sn_fast' not in self.simu_config['name']:
            for genpar in gen_params:
                lc = self.simuLCs(obs, season, genpar)
                if lc:
                    list_lc += lc
                    if not self.throwafterdump:
                        list_lc_keep += lc
                # every 20 SN: dump to file
                if self.save_status:
                    if len(list_lc) >= 20:
                        self.dump(list_lc, season, iproc, meta_lc)
                        list_lc = []

        else:
            list_lc = self.simuLCs(obs, season, gen_params)
            if not self.throwafterdump:
                import copy
                list_lc_keep = copy.deepcopy(list_lc)

        if len(list_lc) > 0 and self.save_status:
            self.dump(list_lc, season, iproc, meta_lc)
            list_lc = []

        if output_q is not None:
            output_q.put({j: (meta_lc, list_lc_keep)})
        else:
            return (meta_lc, list_lc_keep)

    def dump_deprecated(self, list_lc, season, j, meta_lc):
        """
        Method to write a list of lc on disk

        Parameters
        ----------
        list_lc: list
          list of astropy tables
        season: int
          season for observations
        j: int
         tag for multiprocessing

        """

        for lc in list_lc:
            ido = True
            if self.throwaway_empty and len(lc) == 0:
                ido = False
            if ido:
                self.SNID[j] += 1
                self.writeLC(self.SNID[j], lc, season, j, meta_lc)

    def save_metadata_deprecated(self, isav=-1):
        """ Copy metadata to disk

        """
        if self.sn_meta:
            for key, vals in self.sn_meta.items():
                if vals:
                    # print('metadata',vals)
                    Table(vals).write(
                        self.simu_out[key], 'summary_{}'.format(isav),
                        append=True, compression=True)

    def prepareSave_deprecated(self, outdir, prodid, iproc):
        """ Prepare output directories for data

        Parameters
        --------------
        outdir: str
         output directory where to copy the data
        prodid: str
         production id(label for input files)
        iproc: int
          internal tag for multiprocessing

        Returns
        ----------
        Two output files are open:
        - astropy table with (SNID, RA, Dec, X1, Color, z) parameters
        -> name: Simu_prodid.hdf5
        - astropy tables with LC
        -> name: LC_prodid.hdf5

        """

        if not os.path.exists(outdir):
            print('Creating output directory', outdir)
            os.makedirs(outdir)
        # Two files  to be opened - tagged by iproc
        # One containing a summary of the simulation:
        # astropy table with (SNID,RA,Dec,X1,Color,z) parameters
        # -> name: Simu_prodid.hdf5
        # A second containing the Light curves (list of astropy tables)
        # -> name : LC_prodid.hdf5

        self.simu_out[iproc] = '{}/Simu_{}_{}.hdf5'.format(
            outdir, prodid, iproc)
        self.lc_out[iproc] = '{}/LC_{}_{}.hdf5'.format(outdir, prodid, iproc)
        self.check_del(self.simu_out[iproc])
        self.check_del(self.lc_out[iproc])
        self.SNID[iproc] = 10**iproc
        self.sn_meta[iproc] = {}

        """
       # and these files will be removed now (before processing)
       # if they exist (to avoid confusions)
       if os.path.exists(self.simu_out):
           os.remove(self.simu_out)
       if os.path.exists(self.lc_out):
           os.remove(self.lc_out)
       """

    def run_deprecated(self, obs, slicePoint=None, imulti=0):
        """ LC simulations

        Parameters
        --------------
        obs: array
          array of observations

        """
        iproc = 1

        if 'iproc' in obs.dtype.names:
            iproc = int(np.mean(obs['iproc']))

        if iproc not in self.nprocdict:
            self.nprocdict[iproc] = iproc
            if self.save_status:
                self.prepareSave(self.outdir, self.prodid, iproc)

        # if 'healpixID' not in obs.dtype.names:
        if slicePoint is not None:
            import numpy.lib.recfunctions as rf
            healpixID = hp.ang2pix(
                slicePoint['nside'], np.rad2deg(slicePoint['ra']),
                np.rad2deg(slicePoint['dec']), nest=True, lonlat=True)
            pixRA, pixDec = hp.pix2ang(
                self.nside, healpixID, nest=True, lonlat=True)
            obs = rf.append_fields(obs, 'healpixID', [healpixID]*len(obs))
            obs = rf.append_fields(obs, 'pixRA', [pixRA]*len(obs))
            obs = rf.append_fields(obs, 'pixDec', [pixDec]*len(obs))

            print('processing pixel', healpixID)

        # estimate seasons
        obs = seasoncalc(obs)

        # select filters
        goodFilters = np.in1d(obs[self.filterCol], self.filterNames)
        obs = obs[goodFilters]

        # stack if necessary
        if self.stacker is not None:
            obs = self.stacker._run(obs)

        self.fieldname = 'unknown'
        self.fieldid = 0
        try:
            iter(self.season)
        except TypeError:
            self.season = [self.season]

        if self.season == [-1]:
            seasons = np.unique(obs[self.seasonCol])
        else:
            seasons = self.season

        time_ref = time.time()
        """
        tracemalloc.start()
        start = tracemalloc.take_snapshot()
        """
        list_lc = []

        for seas in seasons:
            self.index_hdf5 += 10000*(seas-1)

            idxa = obs[self.seasonCol] == seas
            obs_season = obs[idxa]

            if len(obs_season) >= 5:
                print('obs in season', len(obs_season), seas, iproc)
                simres = self.simuSeason(obs_season, seas, iproc)
                if simres is not None:
                    list_lc += simres

            """
            current = tracemalloc.take_snapshot()
            stats = current.compare_to(start, 'filename')
            for i, stat in enumerate(stats[:5], 1):
                print("since_start", i, str(stat))
            """
        # save metadata
        if self.save_status:
            self.save_metadata(np.unique(obs['healpixID']).item())
        # reset metadata dict
        self.sn_meta[iproc] = {}

        """
        top_stats = snapshot.statistics('lineno')

        print("[ Top 10 ]")
        for stat in top_stats[:10]:
            print(stat)
        """
        # print('End of simulation', time.time()-time_ref)

        if list_lc:
            return list_lc

        return None
