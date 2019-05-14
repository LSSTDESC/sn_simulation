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

from sn_simulation.sn_object import SN_Object
from sn_tools.sn_throughputs import Throughputs


class SN(SN_Object):
    """ SN class - inherits from SN_Object
          Input parameters (as given in the input yaml file):
          - SN parameters (x1, color, daymax, z, ...)
          - simulation parameters

         Output:
         - astropy table with the simulated light curve:
               - columns : band, flux, fluxerr, snr_m5,flux_e,zp,zpsys,time
               - metadata : SNID,Ra,Dec,daymax,x1,color,z

    """

    def __init__(self, param, simu_param):
        super().__init__(param.name, param.sn_parameters, param.gen_parameters,
                         param.cosmology, param.telescope, param.SNID, param.area,
                         mjdCol=param.mjdCol, RaCol=param.RaCol, DecCol=param.DecCol,
                         filterCol=param.filterCol, exptimeCol=param.exptimeCol,
                         nexpCol=param.nexpCol,
                         m5Col=param.m5Col, seasonCol=param.seasonCol,
                         seeingEffCol=param.seeingEffCol, seeingGeomCol=param.seeingGeomCol)

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
        self.SN.set(t0=self.sn_parameters['daymax'])
        self.SN.set(c=self.sn_parameters['color'] +
                    self.gen_parameters['epsilon_color'])
        self.SN.set(x1=self.sn_parameters['x1'] +
                    self.gen_parameters['epsilon_x1'])
        # need to correct X0 for alpha and beta
        lumidist = self.cosmology.luminosity_distance(
            self.sn_parameters['z']).value*1.e3
        X0 = self.X0_norm() / lumidist ** 2
        alpha = 0.13
        beta = 3.
        X0 *= np.power(10., 0.4*(alpha *
                                 self.sn_parameters['x1'] - beta *
                                 self.sn_parameters['color']))
        X0 += self.gen_parameters['epsilon_x0']
        self.X0 = X0
        self.SN.set(x0=X0)
        """
        self.SN.set_source_peakabsmag(self.sn_parameters['absmag'],
        self.sn_parameters['band'], self.sn_parameters['magsys'])

        self.X0=self.SN.get('x0')
        """

    def __call__(self, obs, index_hdf5, display=False, time_display=0.):
        """ Simulation of the light curve

        Input
        ---------
        a set of observations


        Returns
        ---------
        astropy table with:
        columns: band, flux, fluxerr, snr_m5,flux_e,zp,zpsys,time
        metadata : SNID,Ra,Dec,DayMax,X1,Color,z
        """
        #assert (len(np.unique(obs[self.RaCol])) == 1)
        #assert (len(np.unique(obs[self.DecCol])) == 1)
        ra = np.mean(obs[self.RaCol])
        dec = np.mean(obs[self.DecCol])
        area = self.area

        metadata = dict(zip(['SNID', 'Ra', 'Dec',
                             'daymax', 'x0', 'epsilon_x0', 'x1', 'epsilon_x1', 'color', 'epsilon_color', 'z', 'survey_area', 'index_hdf5'], [
            self.SNID, ra, dec, self.sn_parameters['daymax'],
            self.X0, self.gen_parameters['epsilon_x0'], self.sn_parameters['x1'], self.gen_parameters[
                'epsilon_x1'], self.sn_parameters['color'], self.gen_parameters['epsilon_color'],
            self.sn_parameters['z'], area, index_hdf5]))

        # print('Simulating SNID', self.SNID)
        obs = self.cutoff(obs, self.sn_parameters['daymax'],
                          self.sn_parameters['z'],
                          self.sn_parameters['min_rf_phase'],
                          self.sn_parameters['max_rf_phase'])

        for band in 'grizy':
            idb = obs[self.filterCol] == band
        if len(obs) == 0:
            return None, metadata

        # output table
        table_lc = Table()

        # set metadata
        table_lc.meta = metadata

        obs.sort(order=self.mjdCol)

        # apply dust here since Ra, Dec is known
        """
        ebvofMW = self.lsstmwebv.calculateEbv(                                                           
            equatorialCoordinates=np.array(
                [[ra], [dec]]))[0]
        self.SN.set(mwebv = ebvofMW)
        """
        fluxes = 10.*self.SN.flux(obs[self.mjdCol], self.wave)

        wavelength = self.wave/10.

        wavelength = np.repeat(wavelength[np.newaxis, :], len(fluxes), 0)
        SED_time = Sed(wavelen=wavelength, flambda=fluxes)

        fluxes = []
        transes = []
        nvals = range(len(SED_time.wavelen))
        seds = [Sed(wavelen=SED_time.wavelen[i], flambda=SED_time.flambda[i])
                for i in nvals]
        transes = np.asarray([self.telescope.atmosphere[obs[self.filterCol][i][-1]]
                              for i in nvals])
        fluxes = np.asarray(
            [seds[i].calcFlux(bandpass=transes[i]) for i in nvals])

        idx = fluxes > 0
        fluxes = fluxes[idx]
        transes = transes[idx]
        obs = obs[idx]
        nvals = range(len(obs))

        photParams = [PhotometricParameters(exptime=obs[self.exptimeCol][i]/obs[self.nexpCol][i],
                                            nexp=obs[self.nexpCol][i]) for i in nvals]
        mag_SN = -2.5 * np.log10(fluxes / 3631.0)  # fluxes are in Jy
        #print('mags',mag_SN,photParams, obs[self.m5Col])
        calc = [SignalToNoise.calcSNR_m5(
            mag_SN[i], transes[i], obs[self.m5Col][i],
            photParams[i]) for i in nvals]
        snr_m5_opsim = [calc[i][0] for i in nvals]
        gamma_opsim = [calc[i][1] for i in nvals]
        exptime = [obs[self.exptimeCol][i] for i in nvals]
        """
        e_per_sec = [seds[i].calcADU(bandpass=transes[i],
                                     photParams=photParams[i]) /
                     obs[self.exptimeCol][i]*photParams[i].gain for i in nvals]
        """
        e_per_sec = self.telescope.mag_to_flux_e_sec(
            mag_SN, obs[self.filterCol], [30.]*len(mag_SN))
        # print(e_per_sec,e_per_sec_b)
        # table_lc=Table(obs)

        # table_lc.remove_column('band')
        table_lc.add_column(Column(fluxes, name='flux'))
        table_lc.add_column(Column(fluxes/snr_m5_opsim, name='fluxerr'))
        table_lc.add_column(Column(snr_m5_opsim, name='snr_m5'))
        table_lc.add_column(Column(gamma_opsim, name='gamma'))
        table_lc.add_column(Column(obs[self.m5Col], name='m5'))
        table_lc.add_column(
            Column(obs[self.seeingEffCol], name=self.seeingEffCol))
        table_lc.add_column(
            Column(obs[self.seeingGeomCol], name=self.seeingGeomCol))
        table_lc.add_column(Column(e_per_sec[:, 1], name='flux_e'))
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
        #idx = table_lc['flux'] >= 0.
        #table_lc = table_lc[idx]

        # print(table_lc.dtype,table_lc['band'])
        if display:
            self.plotLC(table_lc['time', 'band',
                                 'flux', 'fluxerr', 'zp', 'zpsys'], time_display, self.sn_parameters['z'], self.sn_parameters['daymax'], 1)

        return table_lc, metadata

    def X0_norm(self):
        """ Extimate X0 from flux at 10pc
        using Vega spectrum
        """

        from lsst.sims.photUtils import Sed

        name = 'STANDARD'
        band = 'B'
        #thedir = os.getenv('SALT2_DIR')
        thedir = 'SALT2_Files'

        os.environ[name] = thedir+'/Instruments/Landolt'

        trans_standard = Throughputs(through_dir='STANDARD',
                                     telescope_files=[],
                                     filter_files=['sb_-41A.dat'],
                                     atmos=False,
                                     aerosol=False,
                                     filterlist=('A'),
                                     wave_min=3559,
                                     wave_max=5559)

        mag, spectrum_file = self.Get_Mag(
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

    def Get_Mag(self, filename, name, band):
        """ Get magnitude in filename
        """
        sfile = open(filename, 'rb')
        spectrum_file = 'unknown'
        for line in sfile.readlines():
            if np.string_('SPECTRUM') in line:
                spectrum_file = line.decode().split(' ')[1].strip()
            if name in line and band in line:
                return float(line.decode().split(' ')[2]), spectrum_file

        sfile.close()

    def calcInteg(self, bandpass, signal, wavelen):
        """ Estimate integral of signal
        over wavelength using bandpass
        """

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
        """
        Read a file containing [lambda Fnu] (lambda in nm) (Fnu in Jansky).

        Extracted from sims/photUtils/Sed.py which does not seem to work
        """
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
