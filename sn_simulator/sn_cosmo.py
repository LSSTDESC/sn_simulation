import sncosmo
import numpy as np
from astropy.table import Table, vstack
from scipy.interpolate import griddata, interp1d
from sn_simu_wrapper.sn_object import SN_Object
import time
from sn_tools.sn_utils import SNTimer
import pandas as pd
import os


class SN(SN_Object):
    def __init__(self, param, simu_param, reference_lc=None, dustcorr=None):
        super().__init__(param.name,
                         param.sn_parameters,
                         param.simulator_parameters,
                         param.gen_parameters,
                         param.cosmology,
                         param.zp_airmass,
                         param.SNID,
                         param.area, param.x0_grid,
                         mjdCol=param.mjdCol,
                         RACol=param.RACol,
                         DecCol=param.DecCol,
                         filterCol=param.filterCol,
                         exptimeCol=param.exptimeCol,
                         nexpCol=param.nexpCol,
                         nightCol=param.nightCol,
                         m5Col=param.m5Col,
                         seasonCol=param.seasonCol,
                         seeingEffCol=param.seeingEffCol,
                         seeingGeomCol=param.seeingGeomCol,
                         airmassCol=param.airmassCol,
                         skyCol=param.skyCol, moonCol=param.moonCol,
                         salt2Dir=param.salt2Dir)
        """ SN class - inherits from SN_Object

        Parameters
        --------------
        param: dict
         parameters requested for the simulation (SN_Object)
        simu_param : dict
         parameters for the simulation:
               name: simulator name (str)
               model: model name for SN (exempla: salt2-extended) (str)
               version: version of the model (str)
        reference_lc : griddata,opt
           reference_light curves (default: None)

        """

        self.error_model = self.simulator_parameters['errorModel']
        # self.error_model_cut = self.simulator_parameters['errorModelCut']

        # dust map
        self.dustmap = sncosmo.OD94Dust()
        # self.dustmap = sncosmo.CCM89Dust()

        # load info for sigma_mb shift
        self.nsigmamb = self.sn_parameters['modelPar']['mbsigma']

        if np.abs(self.nsigmamb) > 1.e-5:
            df = pd.read_hdf(self.sn_parameters['modelPar']['mbsigmafile'])
            self.sigma_mb_z = interp1d(
                df['z'], df['sigma_mb'], bounds_error=False, fill_value=0.)

        model = simu_param['model']
        version = str(simu_param['version'])

        self.sn_model = model
        self.sn_version = version

        self.dL = self.cosmology.luminosity_distance(
            self.sn_parameters['z']).value*1.e3  # in kpc

        self.sn_type = self.sn_parameters['type']
        if self.sn_type == 'SN_Ia':
            self.source(model, version)
        else:
            self.random_source(self.sn_type, model, version)

        if self.sn_type == 'SN_Ia':
            self.SN.set(t0=self.sn_parameters['daymax'] +
                        self.gen_parameters['epsilon_daymax'])
            if 'salt' in model:
                self.SN_SALT(model)
        else:
            # self.SN.set(t0=self.sn_parameters['daymax'])
            # self.SN.set(t0=-36.1+self.sn_parameters['daymax'])
            self.SN.set(t0=-37.)
            # self.SN.set(t0=-36.06+2.964+self.sn_parameters['daymax'])

        self.defname = dict(zip(['healpixID', 'pixRA', 'pixDec'], [
                            'observationId', param.RACol, param.DecCol]))

        # names for metadata
        self.names_meta = ['RA', 'Dec', 'sn_type', 'sn_model', 'sn_version',
                           'daymax',
                           'z', 'survey_area',
                           'healpixID', 'pixRA', 'pixDec',
                           'season', 'season_length', 'dL', 'ptime',
                           'status', 'ebvofMW']

        if self.sn_type == 'SN_Ia':
            self.names_meta += ['x0', 'epsilon_x0', 'x1',
                                'epsilon_x1', 'color', 'epsilon_color',
                                'epsilon_daymax']

        self.mag_inf = 100.  # mag values to replace infs

        self.nspectra = self.sn_parameters['nspectra']
        """
        bands = self.zp_airmass['band'].tolist()
        slope = self.zp_airmass['slope'].tolist()
        intercept = self.zp_airmass['intercept'].tolist()
        self.zp_slope = dict(zip(bands, slope))
        self.zp_intercept = dict(zip(bands, intercept))
        """
        # band registery in sncosmo - deprecated
        """
        for band in 'grizy':
            name = 'LSST::'+band
            throughput = self.telescope.atmosphere[band]
            try:
                band = sncosmo.get_bandpass(name)
            except Exception as err:
                print(err)
                bandcosmo = sncosmo.Bandpass(
                    throughput.wavelen, throughput.sb, name=name,
                    wave_unit=u.nm)
                sncosmo.registry.register(bandcosmo)
        """

    def source(self, model, version):
        """
        method to instantiate a source from sncosmo

        Parameters
        --------------
        model: str
           built-in from sncosmo
        version: str
          version number

        """
        source = sncosmo.get_source(model, version)

        # print('allo', source.minwave(), source.maxwave())
        self.SN = sncosmo.Model(source=source,
                                effects=[self.dustmap, self.dustmap],
                                effect_names=['host', 'mw'],
                                effect_frames=['rest', 'obs'])
        self.model = model
        # set cosmology here
        self.SN.cosmo = self.cosmology
        self.version = version
        # self.X0 = self.x0(self.dL)
        # self.SN.set(x0=self.X0)
        self.SN.set(z=self.sn_parameters['z'])
        # print(self.SN)
        """
        self.dL = self.cosmology.luminosity_distance(
            self.sn_parameters['z']).value*1.e3
        """

    def random_source(self, sn_type, sn_model='random', sn_version=''):
        """
        Method to choose a random source depending on the type
         This will occur for non-Ia models
         in that case choose return randomly one among available models.

        Parameters
        ---------------
        sn_type: str
           supernovae type
        sn_model: str, opt
          specific model to run on (default: random)
        sn_version: str, opt
          specific version to run on (default : )

        """

        # load the possible models for this type of supernova
        # get the location of the file
        location = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__)))

        location = location.replace('sn_simulator', 'sn_simu_input')

        # load parameters
        df = pd.read_csv(
            '{}/sncosmo_builtins.txt'.format(location), delimiter=' ')

        # sn_model!='random': choose this model
        df['version'] = df['version'].astype(str)
        # print(df.dtypes)
        if sn_model != 'random':
            idx = df['name'] == sn_model
            idx &= df['version'] == str(sn_version)
            selm = df[idx]
            sn_type = '{}_{}'.format(
                selm['type'].values.item(), selm['subtype'].values.item())
            sn_version = str(selm['version'].values.item())
        else:
            sn_type, sn_model, sn_version = self.get_sn_fromlist(sn_type, df)

        if sn_type == 'SN_Ia':
            sn_type += 'T'
        self.sn_type = sn_type
        self.sn_model = sn_model
        self.sn_version = sn_version

        # self.source(self.sn_model, self.sn_version)
        source = sncosmo.get_source(sn_model, sn_version)
        self.SN = sncosmo.Model(source=source,
                                effects=[self.dustmap, self.dustmap],
                                effect_names=['host', 'mw'],
                                effect_frames=['rest', 'obs'])
        self.SN.set(z=self.sn_parameters['z'])

        self.SN.set_source_peakabsmag(self.sn_parameters['absmag'],
                                      self.sn_parameters['band'],
                                      self.sn_parameters['magsys'])
        # print(self.SN)
        # self.SN.set(amplitude=2.e-8)

    def get_sn_fromlist(self, sn_type, df):
        """
        Method to get a sn model, type, version from a list

        Parameters
        --------------
        sn_type: str
          type of sn
        df: pandas df
          list of sn

        Returns
        ----------
        sn_type, sn_model, sn_version
        """

        tt = sn_type.split('_')[0]
        st = sn_type.split('_')[1]

        if sn_type == 'Non_Ia':
            idx = df['type'] == 'SN'
            idx &= df['subtype'] != 'Ia'
        else:
            if sn_type == 'SN_IaT':
                idx = df['type'] == tt
                idx &= df['subtype'] == 'Ia'
            else:
                idx = df['type'] == tt
                idx &= df['subtype'] == st

        sel = df[idx]
        sela = sel.groupby(['type', 'subtype']).size().to_frame(
            'size').reset_index()
        io = np.random.choice(
            len(sela), 1, p=sela['size'].values/sela['size'].sum())[0]
        sn_type = '{}_{}'.format(
            sela.iloc[io]['type'], sela.iloc[io]['subtype'])

        main_type = sn_type.split('_')[0]
        sub_type = sn_type.split('_')[1]

        idx = df['type'] == main_type
        idx &= df['subtype'] == sub_type
        idx &= df['name'] != 'snana-04d4jv'  # this crashes

        sel = df[idx]

        # print(sel)
        # take a random
        io = np.random.randint(0, len(sel), 1)[0]

        return sn_type, sel.iloc[io]['name'], str(sel.iloc[io]['version'])

    def SN_SALT(self, model):
        """
        Method to set SALT2 parameters for SN

        Parameters
        --------------
        model: str
           model name

        """

        if model == 'salt2-extended' or model == 'salt3':
            model_min = 300.
            model_max = 180000.
            wave_min = 2000.
            wave_max = 11000.

        if model == 'salt2':
            model_min = 3400.
            model_max = 11501.
            wave_min = model_min
            wave_max = model_max

        self.wave = np.arange(wave_min, wave_max, 1.)
        self.wave *= (1.+self.sn_parameters['z'])

        """
        if not self.error_model:
            source = sncosmo.get_source(model, version=version)
        else:
            SALT2Dir = 'SALT2.Guy10_UV2IR'
            self.SALT2Templates(
                SALT2Dir=SALT2Dir,
                blue_cutoff=10.*self.sn_parameters['blue_cutoff'])
            source = sncosmo.SALT2Source(modeldir=SALT2Dir)
        """
        # set x1 and c parameters
        self.SN.set(c=self.sn_parameters['color'] +
                    self.gen_parameters['epsilon_color'])
        self.SN.set(x1=self.sn_parameters['x1'] +
                    self.gen_parameters['epsilon_x1'])

        self.X0 = self.x0(self.dL)
        """
        self.SN.set_source_peakabsmag(self.sn_parameters['absmag'],
                                      self.sn_parameters['band'],
                                      self.sn_parameters['magsys'])

        # get X0 fro source_abspeak norm

        self.X0 = self.SN.get('x0')
        # need to correct X0 for alpha and beta
        alpha = 0.13
        beta = 3.1
        self.X0 *= np.power(10., 0.4*(alpha *
                                 self.sn_parameters['x1'] - beta *
                                 self.sn_parameters['color']))
        # estimate mb
        mb = -2.5*np.log10(self.X0)+10.635

        # smear if
        from random import gauss

        print('before iii',mb,self.X0,self.sn_parameters['sigmaInt'])
        mb = gauss(mb,self.sn_parameters['sigmaInt'])


        # and recalculate X0
        self.X0 = 10**(-0.4*(mb-10.635))
        print('after',mb,self.X0)

        self.X0 += self.gen_parameters['epsilon_x0']

        """
        # set X0
        self.SN.set(x0=self.X0)
        # print('after',self.SN.get('x0'),self.SN.get('x1'),self.SN.get('c'))

    def x0(self, lumidist):
        """"
        Method to estimate x0 from a griddata

        Parameters
        ---------------
        lumidist: float
          luminosity distance

        """

        X0_grid = griddata((self.x0_grid['x1'], self.x0_grid['color']),
                           self.x0_grid['x0_norm'], (
            self.sn_parameters['x1'], self.sn_parameters['color']),
            method='nearest')
        X0 = X0_grid / lumidist ** 2
        alpha = 0.13
        beta = 3.1
        X0 *= np.power(10., 0.4*(alpha *
                                 self.sn_parameters['x1'] - beta *
                                 self.sn_parameters['color']))

        if self.sn_parameters['sigmaInt'] > 0 or np.abs(self.nsigmamb) > 1.e-5:

            # estimate mb
            mb = -2.5*np.log10(X0)+10.635

            if self.sn_parameters['sigmaInt'] > 0:
                # smear it
                from random import gauss

                mb = gauss(mb, self.sn_parameters['sigmaInt'])

            if np.abs(self.nsigmamb) > 1.e-5:
                # get sigma from z
                sigmamb = self.sigma_mb_z(self.sn_parameters['z'])
                mb += self.nsigmamb*sigmamb

            # and recalculate X0
            X0 = 10**(-0.4*(mb-10.635))

        X0 += self.gen_parameters['epsilon_x0']

        return X0

    def SALT2Templates(self, SALT2Dir='SALT2.Guy10_UV2IR', blue_cutoff=3800.):
        """
        Method to load SALT2 templates and apply cutoff on SED.

        Parameters
        --------------
        SALT2Dir: str, opt
          SALT2 directory(default: SALT2.Guy10_UV2IR)
        blue_cutoff: float, opt
           blue cut off to apply(in nm - default: 3800.)

        """

        for vv in ['salt2_template_0', 'salt2_template_1']:
            fName = '{}/{}_orig.dat'.format(SALT2Dir, vv)
            data = np.loadtxt(fName, dtype={'names': ('phase', 'wavelength',
                                                      'flux'),
                                            'formats': ('f8', 'i4', 'f8')})
            # print(data)
            data['flux'][data['wavelength'] <= blue_cutoff] = 0.0

            # print(data)
            np.savetxt('{}/{}.dat'.format(SALT2Dir, vv),
                       data, fmt=['%1.2f', '%4d', '%.7e', ])

    def __call__(self, obs, display=False, time_display=0.):
        """ Simulation of the light curve

        Parameters
        --------------
        obs: array
          a set of observations
        display: bool, opt
          if True: the simulated LC is displayed
          default: False
        time_display: float
          duration(sec) for which the display is visible
          default: 0

        Returns
        -----------
        astropy table:
        metadata:
          # SNID: ID of the supernova(int)
          RA: SN RA(float)
          Dec: SN Dec(float)
          daymax: day of the max luminosity(float)
          epsilon_daymax: epsilon added to daymax for simulation(float)
          x0: SN x0(float)
          epsilon_x0: epsilon added to x0 for simulation(float)
          x1: SN x1(float)
          epsilon_x1: epsilon added to x1 for simulation(float)
          color: SN color(float)
          epsilon_color: epsilon added to color for simulation(float)
          z: SN redshift(float)
          survey_area: survey area for this SN(float)
          pixID: pixel ID
          pixRA: pixel RA
          pixDec: pixel Dec
          season: season
          dL: luminosity distance
        fields:
          flux: SN flux(Jy)
          fluxerr: EN error flux(Jy)
          snr_m5: Signal-to-Noise Ratio(float)
          gamma: gamma parameter(see LSST:
                From Science...data products eq. 5)(float)
          m5: five-sigma depth(float)
          seeingFwhmEff: seeing eff(float)
          seeingFwhmGeom: seeing geom(float)
          flux_e_sec: flux in pe.s-1 (float)
          mag: magnitude(float)
          exptime: exposure time(float)
          magerr: magg error(float)
          band: filter(str)
          zp: zeropoint(float)
          zpsys: zeropoint system(float)
          time: time(days)(float)
          phase: phase(float)
        """

        ra = np.mean(obs[self.RACol])
        dec = np.mean(obs[self.DecCol])
        area = self.area
        season = np.unique(obs['season'])[0]
        season_length = np.max(obs[self.mjdCol])-np.min(obs[self.mjdCol])

        pix = {}
        for vv in ['healpixID', 'pixRA', 'pixDec']:
            if vv in obs.dtype.names:
                pix[vv] = np.unique(obs[vv])[0]
            else:
                pix[vv] = np.mean(obs[self.defname[vv]])

        ebvofMW = self.sn_parameters['ebvofMW']
        # apply dust here since Ra, Dec is known

        if ebvofMW < 0.:
            ebvofMW = self.ebvofMW_calc(pix['pixRA'], pix['pixDec'])

        self.SN.set(mwebv=ebvofMW)

        # start timer
        ti = SNTimer(time.time())

        obs = self.select_filter_cutoff(obs, ra, dec, pix, area, season,
                                        season_length,
                                        ti, ebvofMW)

        if len(obs) == 0:
            return [self.nosim(ra, dec, pix, area, season, season_length,
                               ti, -1, ebvofMW)]

        # preparing the results : stored in lcdf pandas DataFrame
        outvals = [self.m5Col, self.mjdCol,
                   self.exptimeCol, self.nexpCol,
                   self.filterCol, self.nightCol]
        for bb in [self.airmassCol, self.skyCol, self.moonCol,
                   self.seeingEffCol, self.seeingGeomCol]:
            if bb in obs.dtype.names:
                outvals.append(bb)

        lcdf = pd.DataFrame(np.copy(obs[outvals]))

        band_cosmo = '{}_cosmo'.format(self.filterCol)
        lcdf[band_cosmo] = 'lsst'+lcdf[self.filterCol]

        # zp variation vs airmass
        lst = lcdf[self.filterCol].tolist()
        lcdf['zp_slope'] = np.array([*map(self.zp_slope.get, lst)])
        lcdf['zp_intercept'] = np.array([*map(self.zp_intercept.get, lst)])
        lcdf['zp'] = lcdf['zp_slope']*lcdf['airmass']+lcdf['zp_intercept']

        lcdf['zpsys'] = 'ab'

        # get band flux
        lcdf['flux'] = self.SN.bandflux(
            lcdf[band_cosmo], lcdf[self.mjdCol], zpsys=lcdf['zpsys'],
            zp=lcdf['zp'])

        """
        # flux in JY
        lcdf['flux_old'] = self.SN.bandflux(
            lcdf[band_cosmo], lcdf[self.mjdCol], zpsys='ab', zp=2.5*np.log10(3631))
        lcdf['mag_old'] = -2.5 * np.log10(lcdf['flux_old'] / 3631.0)
        """

        # estimate error model (if necessary)

        lcdf['fluxerr_model'] = 0.
        if self.error_model and self.sn_type == 'SN_Ia':
            fluxcov_cosmo = self.SN.bandfluxcov(
                lcdf[band_cosmo], lcdf[self.mjdCol], zpsys='ab',
                zp=2.5*np.log10(3631))
            lcdf['fluxerr_model'] = np.sqrt(np.diag(fluxcov_cosmo[1]))

        """
        lcdf.loc[lcdf.flux <= 0., 'fluxerr_photo'] = -1.
        lcdf.loc[lcdf.flux <= 0., 'fluxerr_model'] = -1.
        lcdf.loc[lcdf.flux <= 0., 'flux'] = 9999.
        lcdf.loc[lcdf.flux_old <= 0., 'flux'] = 9999.
        """
        # positive flux only
        """
        idx = lcdf['flux'] > 0.
        lcdf = lcdf[idx]
        """

        if len(lcdf) == 0:
            return []
        # ti(time.time(), 'fluxes_b')

        # magnitudes - fluxes are in ADU/s
        lcdf['mag'] = -2.5 * np.log10(lcdf['flux'])+lcdf['zp']

        # if mag have inf values -> set to 50.
        lcdf['mag'] = lcdf['mag'].replace([np.inf, -np.inf], self.mag_inf)

        # flux error
        flux5 = 10**(-0.4*(lcdf[self.m5Col]-lcdf['zp']))
        sigma_5 = flux5/5.
        shot_noise = np.sqrt(lcdf['flux']/lcdf[self.exptimeCol])
        lcdf['fluxerr'] = np.sqrt(sigma_5**2+shot_noise**2)
        lcdf['snr_m5'] = lcdf['flux']/lcdf['fluxerr']

        # complete the LC
        lcdf['magerr_phot'] = (2.5/np.log(10.))/lcdf['snr_m5']  # mag error
        lcdf['fluxerr_photo'] = lcdf['fluxerr']

        lcdf['fluxerr'] = np.sqrt(
            lcdf['fluxerr_model']**2+lcdf['fluxerr_photo']**2)  # flux error
        lcdf['snr'] = lcdf['flux']/lcdf['fluxerr']  # snr
        lcdf['magerr'] = (2.5/np.log(10.))/lcdf['snr']  # mag error
        # lcdf['zpsys'] = 'ab'  # zpsys
        lcdf['phase'] = (lcdf[self.mjdCol]-self.sn_parameters['daymax']
                         )/(1.+self.sn_parameters['z'])  # phase

        # rename some of the columns
        lcdf = lcdf.rename(
            columns={self.mjdCol: 'time', self.filterCol: 'band',
                     self.m5Col: 'm5', self.exptimeCol: 'exptime'})

        lcdf['filter'] = lcdf['band']
        lcdf['band'] = lcdf[band_cosmo]

        # remove rows with mag_inf values
        """
        idf = lcdf['mag'] < self.mag_inf
        lcdf = lcdf[idf]
        """

        lcdf.loc[lcdf.fluxerr_model < 0, 'flux'] = 0.
        lcdf.loc[lcdf.fluxerr_model < 0, 'fluxerr_photo'] = 10.
        lcdf.loc[lcdf.fluxerr_model < 0, 'fluxerr'] = 10.
        lcdf.loc[lcdf.fluxerr_model < 0, 'snr_m5'] = 0.
        lcdf.loc[lcdf.fluxerr_model < 0, 'fluxerr_model'] = 10.

        filters = np.array(lcdf['filter'])
        filters = filters.reshape((len(filters), 1))

        lcdf['lambdabar'] = np.apply_along_axis(self.lambdabar, 1, filters)

        if len(lcdf) == 0:
            return [self.nosim(ra, dec, pix, area, season, season_length,
                               ti, -1, ebvofMW)]

        # get the processing time
        ptime = ti.finish(time.time())['ptime'].item()

        """
        print('kkkkk', lcdf[['mag', 'mag_old', 'flux',
               'flux_old', 'snr_m5', 'm5', 'band', 'filter']], len(lcdf))
        """

        # transform pandas df to astropy Table
        table_lc = Table.from_pandas(lcdf)
        # set metadata
        table_lc.meta = self.metadata(
            ra, dec, pix, area, season, season_length, ptime,
            1, ebvofMW)

        # if the user chooses to display the results...
        if display:
            self.plotLC(table_lc['time', 'band',
                                 'flux', 'fluxerr', 'zp', 'zpsys'],
                        time_display)

        # remove LC points with too high error model value
        """
        if self.error_model:
            if self.error_model_cut >0:
                idx = table_lc['fluxerr_model']/ \
                    table_lc['flux']<= self.error_model_cut
                table_lc = table_lc[idx]
        """
        toremove = ['m5', 'exptime', 'numExposures', 'filter_cosmo',
                    'airmass', 'moonPhase',
                    'seeingFwhmEff', 'seeingFwhmGeom', 'gamma', 'mag',
                    'magerr', 'flux_e_sec', 'magerr_phot']

        toremove = ['m5', 'exptime', 'numExposures', 'filter_cosmo',
                    'airmass', 'moonPhase',
                    'seeingFwhmEff', 'seeingFwhmGeom', 'gamma', 'mag',
                    'magerr', 'magerr_phot']

        toremove = ['filter_cosmo', 'airmass', 'moonPhase',
                    'gamma', 'mag', 'magerr', 'magerr_phot']

        # table_lc.remove_columns(toremove)

        return [table_lc]

    def select_filter_cutoff(self, obs, ra, dec, pix, area, season,
                             season_length,
                             ti, ebvofMW):
        """
        Function to select obs: filters and blue and red cutoffs

        Parameters
        ----------
        obs : array
            data to process.
        ra : float
            pixRA.
        dec : float
            pixDec.
        pix : TYPE
            DESCRIPTION.
        area : str
            pixarea.
        season : int
            season of obs.
        season_length : float
            season length
        ti : float
            timer val.
        ebvofMW : float
            E(B-V) MW

        Returns
        -------
        array
            selected data.

        """

        # Are there observations with the filters?
        goodFilters = np.in1d(obs[self.filterCol],
                              np.array([b for b in 'grizy']))

        if len(obs[goodFilters]) == 0:
            return [self.nosim(ra, dec, pix, area, season, season_length,
                               ti, -1, ebvofMW)]

        # Select obs depending on min and max phases
        # blue and red cutoffs applied

        blue_cutoffs, red_cutoffs = self.get_cutoffs()

        obs = self.cutoff(obs, self.sn_parameters['daymax'],
                          self.sn_parameters['z'],
                          self.sn_parameters['minRFphase'],
                          self.sn_parameters['maxRFphase'],
                          blue_cutoffs, red_cutoffs)
        if len(obs) > 0:
            # Sort data according to mjd
            obs.sort(order=self.mjdCol)

        return obs

    def get_cutoffs(self):
        """
        Method to estimate blue and red cutoffs as dict

        Returns
        -------
        blue_cutoffs : dict
            blue cutoffs (key=band).
        red_cutoffs : dict
            red cutoffs (key=band)

        """

        blue_cutoff = 0.
        red_cutoff = 1.e8
        blue_cutoffs = dict(zip('urgizy', [blue_cutoff]*6))
        red_cutoffs = dict(zip('urgizy', [red_cutoff]*6))
        if self.sn_type == 'SN_Ia' and not self.error_model:
            for b in 'ugrizy':
                blue_cutoffs[b] = self.sn_parameters['blueCutoff{}'.format(
                    b)]
                red_cutoffs[b] = self.sn_parameters['redCutoff{}'.format(
                    b)]

        return blue_cutoffs, red_cutoffs

    def calcSNR_Flux_deprecated(self, df, transm):
        """
        Method to estimate SNRs and fluxes(in e.sec)
        using lsst sims estimators

        Parameters
        ---------------
        df: pandas df
           data to process
        transm: array
           throughputs

        Returns
        ----------
        original df plus the following cols:
        gamma: gamma values
        snr_m5: snr values
        flux_e_sec: flux in pe/sec

        """

        # estimate SNR
        # Get photometric parameters to estimate SNR
        from rubin_sim.phot_utils.photometric_parameters \
            import PhotometricParameters
        from rubin_sim.phot_utils import signaltonoise
        photParams = [PhotometricParameters(
            exptime=vv[self.exptimeCol]/vv[self.nexpCol],
            nexp=vv[self.nexpCol]) for index, vv in df.iterrows()]

        nvals = range(len(df))
        calc = [signaltonoise.calcSNR_m5(
            df.iloc[i]['mag'], transm[i], df.iloc[i][self.m5Col],
            photParams[i]) for i in nvals]

        df['snr_m5'] = [calc[i][0] for i in nvals]
        df['gamma'] = [calc[i][1] for i in nvals]
        # estimate the flux in elec.sec-1
        df['flux_e_sec'] = self.telescope.mag_to_flux_e_sec(
            df['mag'].values, df[self.filterCol].values,
            df[self.exptimeCol]/df[self.nexpCol], df[self.nexpCol])[:, 1]

        return df

    def interp_gamma_flux_deprecated(self, grp, gammaName='gamma_int',
                                     fluxName='flux_e_sec_int'):
        """
        Method to estimate gamma and mag_to_flux values from interpolation

        Parameters
        ---------------
        grp: pandas group
          data to process

        Returns
        ----------
        original group with two new cols:
          gamma: gamma values
          flux_e_sec: flux in pe.sec-1
        """

        single_exptime = grp[self.exptimeCol]/grp[self.nexpCol]

        # gamma interp
        grp.loc[:, gammaName] = self.gamma[grp.name](
            (grp[self.m5Col].values, single_exptime, grp[self.nexpCol]))

        # mag_to_flux interp
        grp.loc[:, fluxName] = self.mag_to_flux[grp.name](
            (grp['mag'], single_exptime, grp[self.nexpCol]))

        """
        # SNR_photo_bd
        grp.loc[:, 'f5'] = self.mag_to_flux[grp.name](
            (grp['fiveSigmaDepth'], single_exptime, grp[self.nexpCol]))

        grp.loc[:, 'SNR_photo_bd'] = grp[fluxName]/(grp['f5']/5.)
        """
        return grp

    def nosim(self, ra, dec, pix, area, season, season_length,
              ti, status, ebvofMW):
        """
        Method to construct an empty table when no simulation was not possible

        Parameters
        ---------------
        ra: float
          SN RA
        dec: float
          SN Dec
        pix: dict
          pixel infos
        area: float
           survey area
        season: int
          season of interest
        season_length: float
          season length
        ptime: float
           processing time
        status: int
          status of the processing(1=ok, -1=no simu)
        ebvofMW: float
          E(B-V) of MW
        """
        ptime = ti.finish(time.time())['ptime'].item()
        table_lc = Table()
        # set metadata
        table_lc.meta = self.metadata(
            ra, dec, pix, area, season, season_length, ptime,
            status, ebvofMW)
        return table_lc

    def metadata(self, ra, dec, pix, area, season, season_length,
                 ptime, status, ebvofMW):
        """
        Method to fill metadata

        Parameters
        ---------------
        ra: float
          SN ra
        dec: float
          SN dec
        pix: dict
          pixel infos(ID, RA, Dec)
        area: float
           area of the survey
        season: float
           season number
        season_length: float
          season length
        ptime: float
           processing time
        status: int
          status of the simulation(1=ok, -1=not simulation)

        Returns
        -----------
        dict of metadata

        """

        val_meta = [ra, dec, self.sn_type, self.sn_model, self.sn_version,
                    self.sn_parameters['daymax'],
                    self.sn_parameters['z'], area,
                    pix['healpixID'], pix['pixRA'], pix['pixDec'],
                    season, season_length, self.dL, ptime,
                    status, ebvofMW]

        if self.sn_type == 'SN_Ia':
            val_meta += [self.X0, self.gen_parameters['epsilon_x0'],
                         self.sn_parameters['x1'],
                         self.gen_parameters['epsilon_x1'],
                         self.sn_parameters['color'],
                         self.gen_parameters['epsilon_color'],
                         self.gen_parameters['epsilon_daymax']]

        return dict(zip(self.names_meta, val_meta))

    def SN_SED(self, gen_params):
        """
        Method to generate SN SEDs

        Parameters
        ---------
        gen_params : list
            Simulation parameters.

        Returns
        -------
        sed : astropy table
            Generated SEDs

        """

        daymax = gen_params['daymax']

        min_mjd = daymax-10
        max_mjd = daymax+10

        mjds = np.random.uniform(low=min_mjd, high=max_mjd, size=self.nspectra)

        sed = Table()
        for io, mjd in enumerate(mjds):
            diff = mjd-daymax
            sedm = self.SN_SED_mjd(mjd)
            sedm['spec'] = 'spec_{}'.format(io)
            #sedm['time'] = int(diff)
            sedm['mjd'] = mjd
            sedm['exptime'] = 0.0
            sedm['valid'] = 1

            sed = vstack([sed, Table(sedm)])

        # self.plot_SED(sed)

        return [sed]

    def plot_SED(self, sed):
        """
        Method to plot generated SEDs

        Parameters
        ----------
        sed : astropy table
            generated seds.

        Returns
        -------
        None.

        """

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        # this is to plot SEDs
        for sedid in np.unique(sed['SED_id']):
            idx = sed['SED_id'] == sedid
            sel = sed[idx]
            ax.plot(sel['wave'], sel['flux'], marker='.',
                    label='{}'.format(np.mean(sel['time'])))
        ax.legend()
        plt.show()

    def SN_SED_mjd(self, mjd):
        """
        Method to generate SED flux

        Parameters
        ----------
        mjd : float
            MJD for the flux generation.

        Returns
        -------
        sed : astropy table
            generated fluxes.

        """

        fluxes = 10.*self.SN.flux(mjd, self.wave)
        sed = Table([fluxes], names=['flux'])
        sed['wavelength'] = self.wave
        sed['fluxerr'] = 0.0

        return sed

    def fluxSED(self, obs):
        """
        Method to estimate the fluxes(in Jy) using
        integration of the flux over bandwidth

        Parameters
        --------------
        obs: array
          observations to consider

        Returns
        ----------
        int_fluxes: array
         the fluxes

        """
        from rubin_sim.phot_utils import Sed
        # Get the fluxes (vs wavelength) for each obs
        fluxes = 10.*self.SN.flux(obs[self.mjdCol], self.wave)

        # ti(time.time(), 'fluxes')

        wavelength = self.wave/10.

        wavelength = np.repeat(wavelength[np.newaxis, :], len(fluxes), 0)
        SED_time = Sed(wavelen=wavelength, flambda=fluxes)

        fluxes = []
        transes = []
        nvals = range(len(SED_time.wavelen))
        # Arrays of SED, transmissions to estimate integrated fluxes
        seds = [Sed(wavelen=SED_time.wavelen[i], flambda=SED_time.flambda[i])
                for i in nvals]
        transes = np.asarray([self.telescope.atmosphere[obs[self.filterCol][i]]
                              for i in nvals])
        int_fluxes = np.asarray(
            [seds[i].calcFlux(bandpass=transes[i]) for i in nvals])

        # negative fluxes are a pb for mag estimate
        # -> set neg flux to nearly nothing
        int_fluxes[int_fluxes < 0.] = 1.e-10

        return int_fluxes

    def lambdabar(self, band):

        return self.mean_wavelength[band[0]]

    def ebvofMW_calc(self, RA, Dec):
        """
        Method to estimate E(B-V)

        Parameters
        --------------
        RA: float
          RA coord.
        Dec: float
          Dec coord

        Returns
        ----------
        E(B-V)

        """
        # in that case ebvofMW value is taken from a map
        from astropy.coordinates import SkyCoord
        from dustmaps.sfd import SFDQuery
        coords = SkyCoord(RA, Dec, unit='deg')
        try:
            sfd = SFDQuery()
        except Exception:
            from dustmaps.config import config
            config['data_dir'] = 'dustmaps'
            import dustmaps.sfd
            dustmaps.sfd.fetch()
            # dustmaps('dustmaps')
        sfd = SFDQuery()
        ebvofMW = sfd(coords)

        return ebvofMW
