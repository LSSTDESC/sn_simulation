from sn_simu_wrapper.sn_simu import SNSimulation
import numpy as np
import yaml
import os
from sn_tools.sn_io import check_get_file
import pandas as pd
import operator
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class MakeYaml:
    """
    class to generate a yaml file from a generic one

    Parameters
    ---------------
    dbDir: str
      location dir of the database
    dbName: str
      OS name
    db Extens: str
      db extension (npy or db)
    nside: int
      nside for healpix
    nproc: int
      number of proc for multiprocessing
    diffflux: bool
      to allow for simulation with differential params (ex: x1+epsilon_x1)
    seasnum: list(int)
      season numbers
    outDir: str
      output directory for the production (and also for this yaml file)
    fieldType: str
        type of the field to process (DD, WFD, fake)
     x1Type: str
       x1 type for simulation (unique, uniform, random)
     x1min: float
       x1 min value
     x1max: float
       x1 max value
     x1step: float
        x1 step value
    colorType: str
       color type for simulation (unique, uniform, random)
     colormin: float
       color min value
     colormax: float
       color max value
     colorstep: float
        color step value
     zType: str
       z type for simulation (unique, uniform, random)
     zmin: float
       z min value
     zmax: float
       z max value
     zstep: float
        z step value
     simu: str
       simulator type
     daymaxType: str
       daymax type for simulation (unique, uniform, random)
     daymaxstep: float
        daymax step value
     coadd: bool
       to coadd (True) or not (Fals) observations per night
    prodid: str
       production id ; the resulting yaml file is prodid.yaml
    ebvmw: float
      to specify an extinction value
    bluecutoff: float
       blue cutoff for SN
    redcutoff: float
       redcutoff for SN
    error_model: int
      error model for flux error estimation
    """

    def __init__(self, dbDir, dbName, dbExtens, nside, nproc, diffflux,
                 seasnum, outDir, fieldType,
                 x1Type, x1min, x1max, x1step,
                 colorType, colormin, colormax, colorstep,
                 zType, zmin, zmax, zstep,
                 simu, daymaxType, daymaxstep,
                 coadd, prodid,
                 ebvofMW, bluecutoff, redcutoff, error_model):

        self.dbDir = dbDir
        self.dbName = dbName
        self.dbExtens = dbExtens
        self.nside = nside
        self.nproc = nproc
        self.diffflux = diffflux
        self.seasnum = seasnum
        self.outDir = outDir
        self.fieldType = fieldType
        self.x1Type = x1Type
        self.x1min = x1min
        self.x1max = x1max
        self.x1step = x1step
        self.colorType = colorType
        self.colormin = colormin
        self.colormax = colormax
        self.colorstep = colorstep
        self.zmin = zmin
        self.zmax = zmax
        self.zstep = zstep
        self.simu = simu
        self.zType = zType
        self.daymaxType = daymaxType
        self.daymaxstep = daymaxstep
        self.coadd = coadd
        self.prodid = prodid
        self.ebvofMW = ebvofMW
        self.bluecutoff = bluecutoff
        self.redcutoff = redcutoff
        self.error_model = error_model

    def genYaml(self, input_file):
        """
        method to generate a yaml file
        with parameters from generic input_file

        Parameters
        ---------------
        input_file: str
        input generic yaml file

        Returns
        -----------
        yaml file with parameters


        """
        with open(input_file, 'r') as file:
            filedata = file.read()

        fullDbName = '{}/{}.{}'.format(self.dbDir, self.dbName, self.dbExtens)
        filedata = filedata.replace('prodid', self.prodid)
        filedata = filedata.replace('fullDbName', fullDbName)
        filedata = filedata.replace('nnproc', str(self.nproc))
        filedata = filedata.replace('nnside', str(self.nside))
        filedata = filedata.replace('outputDir', self.outDir)
        filedata = filedata.replace('diffflux', str(self.diffflux))
        filedata = filedata.replace('seasval', str(self.seasnum))
        filedata = filedata.replace('ftype', self.fieldType)
        filedata = filedata.replace('x1Type', self.x1Type)
        filedata = filedata.replace('x1min', str(self.x1min))
        filedata = filedata.replace('x1max', str(self.x1max))
        filedata = filedata.replace('x1step', str(self.x1step))
        filedata = filedata.replace('colorType', self.colorType)
        filedata = filedata.replace('colormin', str(self.colormin))
        filedata = filedata.replace('colormax', str(self.colormax))
        filedata = filedata.replace('colorstep', str(self.colorstep))
        filedata = filedata.replace('zmin', str(self.zmin))
        filedata = filedata.replace('zmax', str(self.zmax))
        filedata = filedata.replace('zstep', str(self.zstep))
        filedata = filedata.replace('zType', self.zType)
        filedata = filedata.replace('daymaxType', self.daymaxType)
        filedata = filedata.replace('daymaxstep', str(self.daymaxstep))
        filedata = filedata.replace('fcoadd', str(self.coadd))
        filedata = filedata.replace('mysimu', self.simu)
        filedata = filedata.replace('ebvofMWval', str(self.ebvofMW))
        filedata = filedata.replace('bluecutoffval', str(self.bluecutoff))
        filedata = filedata.replace('redcutoffval', str(self.redcutoff))
        filedata = filedata.replace('errmod', str(self.error_model))

        return yaml.load(filedata, Loader=yaml.FullLoader)


class FitWrapper:
    def __init__(self, yaml_config_fit):
        """
        Class to fit a set of light curves

        Parameters
        ----------
        config_fit : dict
            parameters fot 

        Returns
        -------
        None.

        """
        from sn_fit.process_fit import Fitting

        # Fit instance
        config = load_config(yaml_config_fit)
        self.fit = Fitting(config)
        self.nproc = config['MultiprocessingFit']['nproc']

        self.saveData = config['OutputFit']['save']

        self.outDir = config['OutputFit']['directory']

        self.prodid = config['Simulations']['prodid']

        if self.saveData:
            from sn_tools.sn_io import checkDir
            checkDir(self.outDir)

    def __call__(self, lc_list):
        """
        Method to fit light curves

        Parameters
        ----------
        lc_list : list(astropy table)
            LC to fit

        Returns
        -------
        None.

        """
        """
        from astropy.table import Table, vstack
        res = Table()
        for lc in lc_list:
            lc.convert_bytestring_to_unicode()
            resfit = self.fit(lc)
            if resfit is not None:
                res = vstack([res, resfit])

        return res
        """
        from sn_tools.sn_utils import multiproc
        params = {}

        res = multiproc(lc_list, params, self.fit_lcs, self.nproc)

        return res

    def fit_lcs(self, lc_list, params, j=-1, output_q=None):
        """
        Method to fit LCs

        Parameters
        ----------
        lc_list : list(astropy table)
            light-curves to fit.
        params : dict
            parameters.
        j : int, optional
            Tag for multiprocessing. The default is 0.
        output_q : multiprocessing queue, optional
            queue managing multiprocessing run. The default is None.

        Returns
        -------
        astropytable
            Result of the fit.

        """

        from astropy.table import Table, vstack
        res = Table()
        for lc in lc_list:
            lc.convert_bytestring_to_unicode()
            resfit = self.fit(lc)
            if resfit is not None:
                resfit = self.check_correct(resfit)
                res = vstack([res, resfit])

        if output_q is not None:
            return output_q.put({j: res})
        else:
            return res

    def check_correct(self, sn):
        """
        Method to correct for Cov_xy col names

        Parameters
        ----------
        sn : astropy Table
            Data to process.

        Returns
        -------
        sn : astropy Table
            Processed data

        """

        varlist = ['z', 't0', 'x0', 'x1', 'color']

        if 'Cov_zz' not in sn.columns:
            varlist = ['t0', 'x0', 'x1', 'color']

        for i, namea in enumerate(varlist):
            for j, nameb in enumerate(varlist):
                if j >= i:
                    vva = 'Cov_{}{}'.format(namea, nameb)
                    vvb = 'Cov_{}{}'.format(nameb, namea)
                    if vva not in sn.columns:
                        sn.rename_column(vvb, vva)

        return sn


class InfoWrapper:
    def __init__(self, confDict):
        """
        class to estimate global parameters of LC
        and add a selection flag according to selection values in dict

        Parameters
        ----------
        confDict : dict
            parameters for selection

        Returns
        -------
        None.

        """

        self.nproc = confDict['nproc_sel']
        from astropy.table import Table
        selfile = confDict['selection_params']
        selpars = Table.read(selfile, format='csv', guess=False, comment='#')

        self.snr_min_value = 0
        self.snr_min_op = operator.ge
        idx = selpars['selname'] == 'snr_min'
        selb = selpars[idx]
        if len(selb) > 0:
            self.snr_min_value = selb['selval'][0]
            self.snrmin_op = selb['selop'][0]
            selpars = selpars[~idx]

        self.selparams = selpars

    def __call__(self, light_curves):
        """
        Main method to estimate LC shepe params
        and add a flag for selection

        Parameters
        ----------
        light_curves : list of astropytables
            LC curves to process

        Returns
        -------
        None.

        """

        conf_names = ['n_epochs_bef',
                      'n_epochs_aft',
                      'n_epochs_phase_minus_10',
                      'n_epochs_phase_plus_20',
                      'n_epochs_m10_p35',
                      'n_epochs_m10_p5',
                      'n_epochs_p5_p20',
                      'n_bands_m8_p10']

        configs = [('night', 'phase', operator.ge, -20, operator.le, 0),
                   ('night', 'phase', operator.gt, 0, operator.le, 60),
                   ('night', 'phase', operator.ge, -20, operator.le, -10.),
                   ('night', 'phase', operator.gt, 20., operator.le, 60.),
                   ('night', 'phase', operator.ge, -10., operator.le, 35.),
                   ('night', 'phase', operator.ge, -10., operator.le, 5.),
                   ('night', 'phase', operator.ge, 5., operator.le, 20.),
                   ('band', 'phase', operator.ge, -8, operator.le, 10.)]

        getInfos = dict(zip(conf_names, configs))

        """
        selParams = [('n_epochs_m10_p35', operator.ge, 4),
                     ('n_epochs_m10_p5', operator.ge, 1),
                     ('n_epochs_p5_p20', operator.ge, 1),
                     ('n_bands_m8_p10', operator.ge, 2)]
        """
        from sn_tools.sn_utils import multiproc
        params = {}
        params['getInfos'] = getInfos
        # params['selParams'] = selParams

        lc_list = multiproc(light_curves, params, self.run_list, self.nproc)

        return lc_list

    def run_list(self, light_curves, params, j, output_q=None):

        getInfos = params['getInfos']
        # selParams = params['selParams']

        lc_list = []
        snr_max = [10, 15, 20]

        for lc in light_curves:
            T0 = lc.meta['daymax']
            z = lc.meta['z']

            if len(lc) == 0:
                resdict = self.calc_dummy(getInfos, snr_max)
            else:
                # apply SNR selection
                idx = self.snr_min_op(lc['snr'], self.snr_min_value)
                lc_sel = lc[idx]
                if len(lc_sel) == 0:
                    resdict = self.calc_dummy(getInfos, snr_max)
                else:
                    resdict = self.calc_infos(lc_sel, T0, z,
                                              getInfos, selParams={})
                    for vval in [10, 15, 20]:
                        vc = 'Nfilt_{}'.format(vval)
                        resdict[vc] = self.nfilt_snrmax(lc_sel, snr_max=vval)

            # update meta data
            lc.meta.update(resdict)
            lc_list.append(lc)

        if output_q is not None:
            return output_q.put({j: lc_list})
        else:
            return lc_list

    def calc_dummy(self, getInfos, snr_max):
        """
        Method returning dummy infos

        Parameters
        ----------
        getInfos :  dict
            dict of selection criteria to measure.
        snr_max: list(int).
             snr_max values for Nfilt estimation

        Returns
        -------
        None.

        """

        resdict = {}
        for key in getInfos.keys():
            resdict[key] = -1

        for b in 'ugrizy':
            resdict['SNR_{}'.format(b)] = -1.0

        resdict['SNR'] = -1.0
        resdict['selected'] = 0
        for vv in snr_max:
            resdict['Nfilt_{}'.format(vv)] = 0

        return resdict

    def calc_infos(self, lc_sel, T0, z, getInfos, selParams={}):
        """
        Method returning infos related to getInfos

        Parameters
        ----------
        lc_sel : astropy table
            LC.
        T0 : float
            SN daymax.
        z : float
            SN z.
        getInfos : dict
            dict of selection criteria to measure.
        selParams: dict, opt.
           selection parameters. The default is {}.

        Returns
        -------
        None.

        """

        resdict = {}
        # add phase column
        lc_sel['phase'] = (lc_sel['time']-T0)/(1+z)
        if 'filter' in lc_sel.columns:
            lc_sel.remove_columns(['filter'])
        # self.plotLC(lc_sel)
        for key, vals in getInfos.items():
            resdict[key] = self.nepochs_phase(
                lc_sel, vals[0], vals[1], vals[2],
                vals[3], vals[4], vals[5])

        if selParams:
            resdict['selected'] = int(self.select(resdict, selParams))
        # add snr per band
        SNRtot = 0.
        for b in 'ugrizy':
            idx = lc_sel['band'] == 'lsst{}'.format(b)
            sel = lc_sel[idx]
            SNR = 0.
            if len(sel) > 0:
                SNR = np.sum(sel['snr_m5']**2)
                SNRtot += SNR
            resdict['SNR_{}'.format(b)] = np.sqrt(SNR)

        resdict['SNR'] = SNRtot

        return resdict

    def select(self, res, list_sel):
        """
        Method to estimate if a LC passes the cut or not

        Parameters
        ----------
        dictval : dict
            dict of values

        Returns
        -------
        bool decision (1= selected, 0=not selected)

        """

        idx = True
        for vals in list_sel:
            idx &= vals[1](res[vals[0]], vals[2])
            if not idx:
                return idx

        return idx
        """
        for key, vals in dictval.items():
            idx = self.selparams['selname'] == key
            pp = self.selparams[idx]
            if len(pp) > 0:
                op = pp['selop'][0]
                selval = pp['selval'][0]
                selstr = '{}({},{})'.format(op, vals, selval)
                resu = eval(selstr)
                if not resu:
                    return False

        return True
        """

    def nepochs_phase(self, tab, colnum='night',
                      colsel='phase', opa=operator.ge, vala=0,
                      opb=operator.le, valb=10):
        """
        Method to get the number of epochs between two vals vala and valb

        Parameters
        ----------
        tab : astropy table
            data to process
        colnum : str, optional
            column to extract the number of epochs from.
            The default is 'night'.
        colsel : str, optional
            selection column name. The default is 'phase'.
        opa : operator, optional
            operator to apply. The default is operator.ge.
        vala : float, optional
            selection value. The default is 0.
        opb: operator, optional
            operator to apply. The default is operator.le.
        valb : float, optional
            selection value. The default is 10.  

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        idx = opa(tab[colsel], vala)
        idx &= opb(tab[colsel], valb)
        tt = tab[idx]

        return len(np.unique(tt[colnum]))

    def nfilt_snrmax(self, lc, snr_max=10):
        """
        Method to estimate the number of bands with max SNR >= snr_max

        Parameters
        ----------
        lc : astropy table
            data to process.
        snr_max : float, optional
            SNR max value. The default is 10.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        bands = np.unique(lc['band'])

        r = []
        for b in bands:
            idx = lc['band'] == b
            sel = lc[idx]
            lc_snr_max = np.max(sel['snr_m5'])
            if lc_snr_max >= snr_max:
                r.append(b)

        return len(r)

    def plotLC(self, tab):
        """
        Method to plot LC for cross-checks

        Parameters
        ----------
        tab : astropy table
            data to process

        Returns
        -------
        None.

        """
        """
        from sn_simu_wrapper.sn_object import SN_Object
        SN_Object.plotLC(tab,time_display)
        """
        import matplotlib.pyplot as plt
        plt.plot(tab['phase'], tab['flux_e_sec'], 'ko')
        plt.show()


class SimInfoFitWrapper:
    def __init__(self, yaml_config_simu, infoDict, yaml_config_fit):
        """


        Parameters
        ----------
        yaml_config_simu : yaml file
            config file for simulation
        infoDict : dict
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.name = 'sim_info_fit'
        self.simu_wrapper = SimuWrapper(yaml_config_simu)
        self.info_wrapper = InfoWrapper(infoDict)
        self.fit_wrapper = FitWrapper(yaml_config_fit)

        self.outName = ''

        self.ccolref = []
        if self.fit_wrapper.saveData:
            outFile = 'SN_{}.hdf5'.format(self.simu_wrapper.prodid)
            self.outName = '{}/{}'.format(self.fit_wrapper.outDir, outFile)
            # check wether this file already exist and remove it
            # import os
            if os.path.isfile(self.outName):
                os.system('rm {}'.format(self.outName))

        self.outdf = pd.DataFrame()

    def run(self, obs, imulti=0):
        """


        Parameters
        ----------
        obs : array
            array of observations
        imulti : int, optional
            Internal parameter. The default is 0.

        Returns
        -------
        None.

        """

        # get Light curves from simuWrapper
        # print('processing pixel', np.unique(
        #    obs[['healpixID', 'pixRA', 'pixDec']]))
        # import time
        # time_ref = time.time()
        light_curves = self.simu_wrapper(obs, imulti)

        # analyze these LC + flag for selection
        if light_curves is None:
            return None

        light_curves_ana = self.info_wrapper(light_curves)
        # print('nlc analyzed', len(light_curves_ana))

        # fitting here
        fitlc = self.fit_wrapper(light_curves_ana)

        self.myconcat(fitlc)
        if len(self.outdf) > 10000:
            self.dump_df()
            self.outdf = pd.DataFrame()
        #ccol = ['RA', 'Dec', 'sn_type']

        # print('nsn', len(fitlc), time.time()-time_ref)
        del light_curves
        del light_curves_ana
        del fitlc

        return None
        """
        if self.fit_wrapper.saveData:
            outFile = 'SN_{}.hdf5'.format(self.simu_wrapper.prodid)
            outName = '{}/{}'.format(self.fit_wrapper.outDir,outFile)
            import astropy
            astropy.io.misc.hdf5.write_table_hdf5(fitlc, outName,
                                                  path='SN', overwrite=True,
                                                  serialize_meta=True)
        """

    def dump(self, fitlc):
        """


        Parameters
        ----------
        fitlc : astropyTable
            data to dump

        Returns
        -------
        None.

        """
        """
        if self.outName != '':
            keyhdf = '{}'.format(int(sn['healpixID'].mean()))
            sn.write(self.outName, keyhdf, append=True, compression=True)
        """
        if self.fit_wrapper.saveData:
            fitlc.convert_bytestring_to_unicode()
            df = pd.DataFrame(fitlc.to_pandas())

            if 'selected' in df.columns:
                df = df.drop(columns=['selected'])

            if not self.ccolref:
                self.ccolref = df.columns.to_list()
            else:
                df = df.reindex(columns=self.ccolref)

            """
            print('chisq', df['chisq'])
            for vv in df.columns:
                print(vv, df[vv].dtype)
            """
            """
            for vv in self.ccolref:
                print(vv, df[vv].dtype)
            """
            """
            cols = ['sn_type', 'sn_model', 'sn_version', 'fitstatus']

            print(df['fitstatus'].unique(), df['SNID'].unique())
            """

            print('dumping', len(df))
            df.to_hdf(self.outName, key='SN', append=True)

    def dump_df(self):
        """
        Method to dum a pandaf df to a file

        Returns
        -------
        None.

        """

        print('dumping df', len(self.outdf))
        self.outdf.to_hdf(self.outName, key='SN', append=True)

    def myconcat(self, fitlc):
        """
        Method to concat a set of astropy tables with a pandas df

        Parameters
        ----------
        fitlc : astropy table
            Data to concat.

        Returns
        -------
        None.

        """

        fitlc.convert_bytestring_to_unicode()
        df = pd.DataFrame(fitlc.to_pandas())

        if 'selected' in df.columns:
            df = df.drop(columns=['selected'])

        if not self.ccolref:
            self.ccolref = df.columns.to_list()
        else:
            df = df.reindex(columns=self.ccolref)

        self.outdf = pd.concat((self.outdf, df))

    def finish(self):
        """
        Method to use at the end of the run

        Returns
        -------
        None.

        """

        if len(self.outdf) > 0:
            self.dump_df()


class SimuWrapper:
    """
    Wrapper class for simulation

    Parameters
    ---------------
    yaml_config: str
      name of the yaml configuration file

    """

    def __init__(self, yaml_config):

        config = load_config(yaml_config)

        self.name = 'simulation'

        # get X0 for SNIa normalization
        x0_tab = self.x0(config)

        # load references if simulator = sn_fast
        # reference_lc = self.load_reference(config)

        # now define the metric instance
        # self.metric = SNMAFSimulation(config=config, x0_norm=x0_tab,
        #                              reference_lc=reference_lc,
        #                              coadd=config['Observations']['coadd'])

        self.metric = SNSimulation(
            config=config, x0_norm=x0_tab)

        self.prodid = config['ProductionIDSimu']

    def x0(self, config):
        """
        Method to load x0 data

        Parameters
        ---------------
        config: dict
          parameters to load and (potentially) regenerate x0s

        Returns
        -----------

        """
        # check whether X0_norm file exist or not
        # (and generate it if necessary)
        absMag = config['SN']['absmag']
        x0normFile = 'reference_files/X0_norm_{}.npy'.format(absMag)
        if not os.path.isfile(x0normFile):
            # if this file does not exist, grab it from a web server
            check_get_file(config['WebPathSimu'], 'reference_files',
                           'X0_norm_{}.npy'.format(absMag))

        if not os.path.isfile(x0normFile):
            # if the file could not be found, then have to generate it!
            salt2Dir = config['SN']['salt2Dir']
            model = config['Simulator']['model']
            version = str(config['Simulator']['version'])

            # need the SALT2 dir for this
            from sn_tools.sn_io import check_get_dir
            check_get_dir(config['Web path'], 'SALT2', salt2Dir)
            from sn_tools.sn_utils import X0_norm
            X0_norm(salt2Dir=salt2Dir, model=model, version=version,
                    absmag=absMag, outfile=x0normFile)

        return np.load(x0normFile)

    def run(self, obs, imulti=0):
        """
        Method to run the metric

        Parameters
        ---------------
        obs: array
          data to process

        """

        light_curves = self.metric.run(obs, imulti=imulti)

        """
        if light_curves is not None:
            print('light curves', len(light_curves))
        else:
            print('no lc on output')
        """
        return light_curves

    __call__ = run

    def finish(self):
        """
        Method to save metadata to disk

        """
        self.metric.save_metadata()


def load_config(yaml_config):
    """


    Parameters
    ----------
    yaml_config : str
        yaml fine name

    Returns
    -------
    config : dict
        dict of the yaml file

    """
    config = {}
    if isinstance(yaml_config, dict):
        config = yaml_config
    else:
        with open(yaml_config) as file:
            config = yaml.full_load(file)

    return config


class InfoFitWrapper:
    def __init__(self, infoDict, yaml_config_fit):
        """


        Parameters
        ----------
        yaml_config_simu : yaml file
            config file for simulation
        infoDict : dict
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.name = 'info_fit'
        self.info_wrapper = InfoWrapper(infoDict)
        self.fit_wrapper = FitWrapper(yaml_config_fit)

        self.outName = ''

        if self.fit_wrapper.saveData:
            outFile = 'SN_{}.hdf5'.format(self.fit_wrapper.prodid)
            self.outName = '{}/{}'.format(self.fit_wrapper.outDir, outFile)
            # check wether this file already exist and remove it
            import os
            if os.path.isfile(self.outName):
                os.system('rm {}'.format(self.outName))

    def run(self, light_curves):
        """


        Parameters
        ----------
        light_curves : list of astropy table
            LC to process

        Returns
        -------
        None.

        """

        # analyze these LC + flag for selection
        light_curves_ana = self.info_wrapper(light_curves)
        # print('nlc analyzed', len(light_curves_ana))

        # fitting here
        fitlc = self.fit_wrapper(light_curves_ana)

        self.dump(fitlc)

        return fitlc

    def dump(self, sn):
        """


        Parameters
        ----------
        sn : astropyTable
            data to dump

        Returns
        -------
        None.

        """
        if self.outName != '':
            keyhdf = '{}'.format(int(sn['healpixID'].mean()))
            sn.write(self.outName, keyhdf, append=True, compression=True)
