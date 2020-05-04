import sncosmo
import numpy as np
from lsst.sims.photUtils import Bandpass, Sed
from lsst.sims.photUtils import SignalToNoise
from lsst.sims.photUtils import PhotometricParameters
from astropy.table import vstack, Table, Column
import matplotlib.animation as manimation
import pylab as plt
import os
from scipy import interpolate, integrate
import h5py
from lsst.sims.catUtils.dust import EBV
from scipy.interpolate import griddata

from sn_simulation.sn_object import SN_Object
from sn_tools.sn_throughputs import Throughputs


class SN(SN_Object):
    def __init__(self, param, simu_param):
        super().__init__(param.name, param.sn_parameters, param.gen_parameters,
                         param.cosmology, param.telescope, param.SNID, param.area, param.x0_grid,
                         mjdCol=param.mjdCol, RACol=param.RACol, DecCol=param.DecCol,
                         filterCol=param.filterCol, exptimeCol=param.exptimeCol,
                         nexpCol=param.nexpCol,
                         m5Col=param.m5Col, seasonCol=param.seasonCol,
                         seeingEffCol=param.seeingEffCol, seeingGeomCol=param.seeingGeomCol,
                         airmassCol=param.airmassCol, skyCol=param.skyCol, moonCol=param.moonCol,
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
      """
        model = simu_param['model']
        version = str(simu_param['version'])
        self.model = model
        self.version = version

        if model == 'salt2-extended':
            model_min = 300.
            model_max = 180000.
            wave_min = 3000.
            wave_max = 11501.

        if model == 'salt2':
            model_min = 3400.
            model_max = 11501.
            wave_min = model_min
            wave_max = model_max

        self.wave = np.arange(wave_min, wave_max, 1.)

        source = sncosmo.get_source(model, version=version)

        self.dustmap = sncosmo.OD94Dust()

        self.lsstmwebv = EBV.EBVbase()
        """
        self.SN = sncosmo.Model(source=source,
                                effects=[self.dustmap, self.dustmap],
                                effect_names=['host', 'mw'],
                                effect_frames=['rest', 'obs'])
        """
        self.SN = sncosmo.Model(source=source)
        self.SN.set(z=self.sn_parameters['z'])
        self.SN.set(t0=self.sn_parameters['daymax'] +
                    self.gen_parameters['epsilon_daymax'])
        self.SN.set(c=self.sn_parameters['color'] +
                    self.gen_parameters['epsilon_color'])
        self.SN.set(x1=self.sn_parameters['x1'] +
                    self.gen_parameters['epsilon_x1'])
        # need to correct X0 for alpha and beta
        lumidist = self.cosmology.luminosity_distance(
            self.sn_parameters['z']).value*1.e3
        X0_grid = griddata((self.x0_grid['x1'], self.x0_grid['color']), self.x0_grid['x0_norm'], (
            self.sn_parameters['x1'], self.sn_parameters['color']),  method='nearest')
        X0 = X0_grid / lumidist ** 2
        alpha = 0.13
        beta = 3.
        X0 *= np.power(10., 0.4*(alpha *
                                 self.sn_parameters['x1'] - beta *
                                 self.sn_parameters['color']))
        X0 += self.gen_parameters['epsilon_x0']
        self.X0 = X0
        self.dL = lumidist
        self.SN.set(x0=X0)
        """
        self.SN.set_source_peakabsmag(self.sn_parameters['absmag'],
        self.sn_parameters['band'], self.sn_parameters['magsys'])

        self.X0=self.SN.get('x0')
        """

        self.defname = dict(zip(['healpixID', 'pixRa', 'pixDec'], [
                            'observationId', param.RACol, param.DecCol]))

    def __call__(self, obs, index_hdf5, display=False, time_display=0.):
        """ Simulation of the light curve

        Parameters
        --------------
        obs: array
          a set of observations
        index_hdf5: int
          index of the light curve in the hdf5 file
        display: bool,opt
          if True: the simulated LC is displayed
          default: False
        time_display: float
          duration (sec) for which the display is visible
          default: 0

        Returns
        -----------
        astropy table:
        metadata:
          SNID: ID of the supernova (int)
          Ra: SN RA (float)
          Dec: SN Dec (float)
          daymax: day of the max luminosity (float)
          epsilon_daymax: epsilon added to daymax for simulation (float)
          x0: SN x0 (float)
          epsilon_x0: epsilon added to x0 for simulation (float)
          x1: SN x1 (float)
          epsilon_x1: epsilon added to x1 for simulation (float)
          color: SN color (float)
          epsilon_color: epsilon added to color for simulation (float)
          z: SN redshift (float)
          survey_area: survey area for this SN (float)
          index_hdf5: SN index in the hdf5 file
          pixID: pixel ID
          pixRa: pixel RA 
          pixDec: pixel Dec 
          season: season
          dL: luminosity distance
        fields:
          flux: SN flux (Jy)
          fluxerr: EN error flux (Jy)
          snr_m5: Signal-to-Noise Ratio (float)
          gamma: gamma parameter (see LSST: From Science...data products eq. 5) (float)
          m5: five-sigma depth (float)
          seeingFwhmEff: seeing eff (float)
          seeingFwhmGeom: seeing geom (float)
          flux_e_sec: flux in pe.s-1 (float)
          mag: magnitude (float)
          exptime: exposure time (float)
          magerr: magg error (float)
          band: filter (str)
          zp: zeropoint (float)
          zpsys: zeropoint system (float)
          time: time (days) (float)
          phase: phase (float)
        """
        #assert (len(np.unique(obs[self.RaCol])) == 1)
        #assert (len(np.unique(obs[self.DecCol])) == 1)
        ra = np.mean(obs[self.RACol])
        dec = np.mean(obs[self.DecCol])
        area = self.area
        season = np.unique(obs['season'])[0]
        pix = {}
        for vv in ['healpixID', 'pixRa', 'pixDec']:
            if vv in obs.dtype.names:
                pix[vv] = np.unique(obs[vv])[0]
            else:
                pix[vv] = np.mean(obs[self.defname[vv]])

        # Metadata
        index = '{}_{}_{}'.format(pix['healpixID'], int(season), index_hdf5)

        names_meta = ['SNID', 'Ra', 'Dec',
                      'x0', 'epsilon_x0',
                      'x1', 'epsilon_x1',
                      'color', 'epsilon_color',
                      'daymax', 'epsilon_daymax',
                      'z', 'survey_area', 'index_hdf5',
                      'pixID', 'pixRa', 'pixDec',
                      'season', 'dL']
        val_meta = [self.SNID, ra, dec,
                    self.X0, self.gen_parameters['epsilon_x0'],
                    self.sn_parameters['x1'], self.gen_parameters['epsilon_x1'],
                    self.sn_parameters['color'], self.gen_parameters['epsilon_color'],
                    self.sn_parameters['daymax'], self.gen_parameters['epsilon_daymax'],
                    self.sn_parameters['z'], area, index,
                    pix['healpixID'], pix['pixRa'], pix['pixDec'],
                    season, self.dL]

        metadata = dict(zip(names_meta, val_meta))

        # Select obs depending on min and max phases
        obs = self.cutoff(obs, self.sn_parameters['daymax'],
                          self.sn_parameters['z'],
                          self.sn_parameters['min_rf_phase'],
                          self.sn_parameters['max_rf_phase'],
                          self.sn_parameters['blue_cutoff'],
                          self.sn_parameters['red_cutoff'])

        """
        print('after sel',obs.dtype)
        for band in np.unique(obs['filter']):
            idx = obs['filter']==band
            sel = obs[idx]
            phase = (sel['observationStartMJD']-self.sn_parameters['daymax'])/(1.+self.sn_parameters['z'])
            print(band,np.min(phase),np.max(phase))
        """
        for band in 'grizy':
            idb = obs[self.filterCol] == band
        if len(obs) == 0:
            return None, metadata

        # output table
        table_lc = Table()

        # set metadata
        table_lc.meta = metadata

        # Sort data according to mjd
        obs.sort(order=self.mjdCol)

        # apply dust here since Ra, Dec is known
        """
        ebvofMW = self.lsstmwebv.calculateEbv(                                                           
            equatorialCoordinates=np.array(
                [[ra], [dec]]))[0]
        self.SN.set(mwebv = ebvofMW)
        """
        # Get the fluxes (vs wavelength) for each obs
        fluxes = 10.*self.SN.flux(obs[self.mjdCol], self.wave)

        wavelength = self.wave/10.

        wavelength = np.repeat(wavelength[np.newaxis, :], len(fluxes), 0)
        SED_time = Sed(wavelen=wavelength, flambda=fluxes)

        fluxes = []
        transes = []
        nvals = range(len(SED_time.wavelen))
        # Arrays of SED, transmissions to estimate integrated fluxes
        seds = [Sed(wavelen=SED_time.wavelen[i], flambda=SED_time.flambda[i])
                for i in nvals]
        transes = np.asarray([self.telescope.atmosphere[obs[self.filterCol][i][-1]]
                              for i in nvals])
        int_fluxes = np.asarray(
            [seds[i].calcFlux(bandpass=transes[i]) for i in nvals])

        #
        #idx = int_fluxes > 0
        int_fluxes[int_fluxes < 0.] = 1.e-10
        """
        int_fluxes = int_fluxes[idx]
        transes = transes[idx]
        obs = obs[idx]
        """
        nvals = range(len(obs))

        # Get photometric parameters to estimate SNR
        photParams = [PhotometricParameters(exptime=obs[self.exptimeCol][i]/obs[self.nexpCol][i],
                                            nexp=obs[self.nexpCol][i]) for i in nvals]
        # magnitude - integrated fluxes are in Jy
        mag_SN = -2.5 * np.log10(int_fluxes / 3631.0)  # fluxes are in Jy
        # estimate SNR
        calc = [SignalToNoise.calcSNR_m5(
            mag_SN[i], transes[i], obs[self.m5Col][i],
            photParams[i]) for i in nvals]
        snr_m5_opsim = [calc[i][0] for i in nvals]
        gamma_opsim = [calc[i][1] for i in nvals]
        exptime = [obs[self.exptimeCol][i] for i in nvals]

        #print('Exposure time',exptime)
        # estimate the flux in elec.sec-1
        e_per_sec = self.telescope.mag_to_flux_e_sec(
            mag_SN, obs[self.filterCol], [30.]*len(mag_SN))

        # Fill the astopy table
        table_lc.add_column(Column(int_fluxes, name='flux'))
        table_lc.add_column(Column(int_fluxes/snr_m5_opsim, name='fluxerr'))
        table_lc.add_column(Column(snr_m5_opsim, name='snr_m5'))
        table_lc.add_column(Column(gamma_opsim, name='gamma'))
        table_lc.add_column(Column(obs[self.m5Col], name='m5'))
        if self.airmassCol in obs.dtype.names:
            table_lc.add_column(
                Column(obs[self.airmassCol], name=self.airmassCol))
        if self.skyCol in obs.dtype.names:
            table_lc.add_column(Column(obs[self.skyCol], name=self.skyCol))
        if self.moonCol in obs.dtype.names:
            table_lc.add_column(Column(obs[self.moonCol], name=self.moonCol))
        table_lc.add_column(Column(obs[self.nexpCol], name=self.nexpCol))
        table_lc.add_column(
            Column(obs[self.seeingEffCol], name=self.seeingEffCol))
        table_lc.add_column(
            Column(obs[self.seeingGeomCol], name=self.seeingGeomCol))
        table_lc.add_column(Column(e_per_sec[:, 1], name='flux_e_sec'))
        table_lc.add_column(Column(mag_SN, name='mag'))
        table_lc.add_column(Column(exptime, name='exptime'))

        table_lc.add_column(
            Column((2.5/np.log(10.))/snr_m5_opsim, name='magerr'))
        table_lc.add_column(
            Column(['LSST::'+obs[self.filterCol][i][-1]
                    for i in range(len(obs[self.filterCol]))], name='band',
                   dtype=h5py.special_dtype(vlen=str)))
        # table_lc.add_column(Column([obs['band'][i][-1]
        # for i in range(len(obs['band']))], name='band'))
        table_lc.add_column(Column([2.5*np.log10(3631)]*len(obs),
                                   name='zp'))
        table_lc.add_column(
            Column(['ab']*len(obs), name='zpsys',
                   dtype=h5py.special_dtype(vlen=str)))
        table_lc.add_column(Column(obs[self.mjdCol], name='time'))
        phases = (table_lc['time']-self.sn_parameters['daymax']
                  )/(1.+self.sn_parameters['z'])
        table_lc.add_column(Column(phases, name='phase'))

        # if the user chooses to display the results...
        if display:
            self.plotLC(table_lc['time', 'band',
                                 'flux', 'fluxerr', 'zp', 'zpsys'], time_display)

        return table_lc, metadata


"""
    def X0_norm(self):
        #Extimate X0 from flux at 10pc
        using Vega spectrum

        Parameters
        --------------

        Returns
        ----------
        x0: float
          x0 from flux at 10pc
        #

        from lsst.sims.photUtils import Sed

        name = 'STANDARD'
        band = 'B'
        #thedir = os.getenv('SALT2_DIR')
        thedir = self.salt2Dir

        os.environ[name] = thedir+'/Instruments/Landolt'

        trans_standard = Throughputs(through_dir='STANDARD',
                                     telescope_files=[],
                                     filter_files=['sb_-41A.dat'],
                                     atmos=False,
                                     aerosol=False,
                                     filterlist=('A'),
                                     wave_min=3559,
                                     wave_max=5559)

        mag, spectrum_file = self.getMag(
            thedir+'/MagSys/VegaBD17-2008-11-28.dat',
            np.string_(name),
            np.string_(band))

        sourcewavelen, sourcefnu = self.readSED_fnu(
            filename=thedir+'/'+spectrum_file)
        CLIGHT_A_s = 2.99792458e18         # [A/s]
        HPLANCK = 6.62606896e-27

        sedb = Sed(wavelen=sourcewavelen, flambda=sourcewavelen *
                   sourcefnu/(CLIGHT_A_s * HPLANCK))

        flux = self.calcInteg(
            bandpass=trans_standard.system['A'],
            signal=sedb.flambda,
            wavelen=sedb.wavelen)

        zp = 2.5*np.log10(flux)+mag
        flux_at_10pc = np.power(10., -0.4 * (self.sn_parameters['absmag']-zp))

        source = sncosmo.get_source(self.model, version=self.version)
        SN = sncosmo.Model(source=source)

        SN.set(z=0.)
        SN.set(t0=0)
        SN.set(c=self.sn_parameters['color'])
        SN.set(x1=self.sn_parameters['x1'])
        SN.set(x0=1)

        fluxes = 10.*SN.flux(0., self.wave)

        wavelength = self.wave/10.
        SED_time = Sed(wavelen=wavelength, flambda=fluxes)

        expTime = 30.
        photParams = PhotometricParameters(nexp=expTime/15.)
        trans = Bandpass(
            wavelen=trans_standard.system['A'].wavelen/10.,
            sb=trans_standard.system['A'].sb)
        # number of ADU counts for expTime
        e_per_sec = SED_time.calcADU(bandpass=trans, photParams=photParams)
        # e_per_sec = sed.calcADU(bandpass=self.transmission.lsst_atmos[filtre], photParams=photParams)
        e_per_sec /= expTime/photParams.gain*photParams.effarea

        return flux_at_10pc * 1.E-4 / e_per_sec

    def getMag(self, filename, name, band):
        #Get magnitude in filename

        Parameters
        --------------
        filename: str
          name of the file to scan
        name: str
           throughtput used
        band: str
          band to consider

        Returns
        ----------
        mag: float
         mag
        spectrum_file: str
         spectrum file

        
        sfile = open(filename, 'rb')
        spectrum_file = 'unknown'
        for line in sfile.readlines():
            if np.string_('SPECTRUM') in line:
                spectrum_file = line.decode().split(' ')[1].strip()
            if name in line and band in line:
                return float(line.decode().split(' ')[2]), spectrum_file

        sfile.close()

    def calcInteg(self, bandpass, signal, wavelen):
        #Estimate integral of signal
        over wavelength using bandpass

        Parameters
        --------------
        bandpass: list(float)
          bandpass
        signal:  list(float)
          signal to integrate (flux)
        wavelength: list(float)
          wavelength used for integration

        Returns
        -----------
        integrated signal (float)
        #

        fa = interpolate.interp1d(bandpass.wavelen, bandpass.sb)
        fb = interpolate.interp1d(wavelen, signal)

        min_wave = np.max([np.min(bandpass.wavelen), np.min(wavelen)])
        max_wave = np.min([np.max(bandpass.wavelen), np.max(wavelen)])

        wavelength_integration_step = 5
        waves = np.arange(min_wave, max_wave, wavelength_integration_step)

        integrand = fa(waves) * fb(waves)

        range_inf = min_wave
        range_sup = max_wave
        n_steps = int((range_sup-range_inf) / wavelength_integration_step)

        x = np.core.function_base.linspace(range_inf, range_sup, n_steps)

        return integrate.simps(integrand, x=waves)

    def readSED_fnu(self, filename, name=None):
        
        Read a file containing [lambda Fnu] (lambda in nm) (Fnu in Jansky).
        Extracted from sims/photUtils/Sed.py which does not seem to work

        Parameters
        --------------
        filename: str
          name of the file to process
        name: str,opt
          default: None

        Returns
        ----------
        sourcewavelen: list(float)
         wavelength with lambda in nm
        sourcefnu: list(float)
         signal with Fnu in Jansky
        
        # Try to open the data file.
        try:
            if filename.endswith('.gz'):
                f = gzip.open(filename, 'rt')
            else:
                f = open(filename, 'r')
        # if the above fails, look for the file with and without the gz
        except IOError:
            try:
                if filename.endswith(".gz"):
                    f = open(filename[:-3], 'r')
                else:
                    f = gzip.open(filename+".gz", 'rt')
            except IOError:
                raise IOError(
                    "The throughput file %s does not exist" % (filename))
        # Read source SED from file
        # lambda, fnu should be first two columns in the file.
        # lambda should be in nm and fnu should be in Jansky.
        sourcewavelen = []
        sourcefnu = []
        for line in f:
            if line.startswith("#"):
                continue
            values = line.split()
            sourcewavelen.append(float(values[0]))
            sourcefnu.append(float(values[1]))
        f.close()
        # Convert to numpy arrays.
        sourcewavelen = np.array(sourcewavelen)
        sourcefnu = np.array(sourcefnu)
        return sourcewavelen, sourcefnu
"""
