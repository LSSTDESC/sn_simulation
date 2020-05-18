from builtins import zip
import numpy as np
import unittest
import lsst.utils.tests
import os
from numpy.testing import assert_almost_equal, assert_equal
import h5py
from astropy.table import Table, vstack
from sn_wrapper.sn_simu import SNSimulation


main_repo = 'https://me.lsst.eu/gris/Reference_Files'
m5_ref = dict(zip('ugrizy', [23.60, 24.83, 24.38, 23.92, 23.35, 22.44]))


def getFile(refdir, fname):
    fullname = '{}/{}/{}'.format(main_repo, refdir, fname)

    # check whether the file is available; if not-> get it!
    if not os.path.isfile(fname):
        print('wget path:', fullname)
        cmd = 'wget --no-clobber --no-verbose {}'.format(fullname)
        os.system(cmd)


def getRefDir(dirname):
    fullname = '{}/{}'.format(main_repo, dirname)

    if not os.path.exists(dirname):
        print('wget path:', fullname)
        cmd = 'wget --no-verbose --recursive {} --directory-prefix={} --no-clobber --no-parent -nH --cut-dirs=3 -R \'index.html*\''.format(
            fullname+'/', dirname)
        os.system(cmd)


def getconfig(prodid,
              x1Type, x1min, x1max, x1step,
              colorType, colormin, colormax, colorstep,
              zType, zmin, zmax, zstep,
              daymaxType, daymaxstep, diffflux,
              fulldbName, fieldType, fcoadd, seasval,
              simulator='sn_cosmo', nside=64, nproc=1, outputDir='.'):

    config = {}
    config['ProductionID'] = prodid+'_'+simulator

    # -------------- Supernova parameters ----------------------------------------
    config['SN parameters'] = {}
    config['SN parameters']['Id'] = 100                   # Id of the first SN
    # stretch and color
    config['SN parameters']['x1'] = {}
    config['SN parameters']['x1']['type'] = x1Type  # unique, uniform or random
    config['SN parameters']['x1']['min'] = x1min
    config['SN parameters']['x1']['max'] = x1max
    config['SN parameters']['x1']['step'] = 1.
    config['SN parameters']['color'] = {}
    # unique, uniform or random
    config['SN parameters']['color']['type'] = colorType
    config['SN parameters']['color']['min'] = colormin
    config['SN parameters']['color']['max'] = colormax
    config['SN parameters']['color']['step'] = 0.05

    config['SN parameters']['x1_color'] = {}
    """
    config['SN parameters']['x1_color']['type'] = x1colorType  # random or fixed
    config['SN parameters']['x1_color']['min'] = [x1min, colormin]
    config['SN parameters']['x1_color']['max'] = [0.2, 0.2]
    """
    config['SN parameters']['x1_color']['rate'] = 'JLA'
    config['SN parameters']['x1_color']['dirFile'] = 'reference_files'
    config['SN parameters']['z'] = {}               # redshift
    config['SN parameters']['z']['type'] = zType
    config['SN parameters']['z']['min'] = zmin
    config['SN parameters']['z']['max'] = zmax
    config['SN parameters']['z']['step'] = zstep
    # Type Ia volumetric rate = Perrett, Ripoche, Dilday.
    config['SN parameters']['z']['rate'] = 'Perrett'
    config['SN parameters']['daymax'] = {}                 # Tmax (obs. frame)
    # unique, uniform or random
    config['SN parameters']['daymax']['type'] = daymaxType
    # if uniform: step (in day) between Tmin(obs) and Tmax(obs)
    config['SN parameters']['daymax']['step'] = daymaxstep
    # obs min and max phase (rest frame) for LC points
    config['SN parameters']['min_rf_phase'] = - 20.
    config['SN parameters']['max_rf_phase'] = 60.
    # obs min and max phase (rest frame) for T0 estimation
    config['SN parameters']['min_rf_phase_qual'] = -15
    config['SN parameters']['max_rf_phase_qual'] = 45
    config['SN parameters']['absmag'] = -19.0906          # peak abs mag
    config['SN parameters']['band'] = 'bessellB'             # band for absmag
    config['SN parameters']['magsys'] = 'vega'              # magsys for absmag
    # to estimate differential flux
    config['SN parameters']['differential_flux'] = diffflux
    # dir where SALT2 files are located
    config['SN parameters']['salt2Dir'] = 'SALT2_Files'
    config['SN parameters']['blue_cutoff'] = 380.
    config['SN parameters']['red_cutoff'] = 800.

    # ------------------cosmology ----------------------
    config['Cosmology'] = {}
    config['Cosmology']['Model'] = 'w0waCDM'      # Cosmological model
    config['Cosmology']['Omega_m'] = 0.30             # Omega_m
    config['Cosmology']['Omega_l'] = 0.70             # Omega_l
    config['Cosmology']['H0'] = 72.0                  # H0
    config['Cosmology']['w0'] = -1.0                  # w0
    config['Cosmology']['wa'] = 0.0                   # wa

    # -------------------instrument -----------------------
    config['Instrument'] = {}
    config['Instrument']['name'] = 'LSST'  # name of the telescope (internal)
    # dir of throughput
    config['Instrument']['throughput_dir'] = 'LSST_THROUGHPUTS_BASELINE'
    config['Instrument']['atmos_dir'] = 'THROUGHPUTS_DIR'  # dir of atmos
    config['Instrument']['airmass'] = 1.2  # airmass value
    config['Instrument']['atmos'] = True  # atmos
    config['Instrument']['aerosol'] = False  # aerosol

    # -------------------observations ------------------------
    config['Observations'] = {}
    # filename: /sps/lsst/cadence/LSST_SN_PhG/cadence_db/opsim_db/kraken_2026.db # Name of db obs file (full path)
    config['Observations']['filename'] = fulldbName
    config['Observations']['fieldtype'] = fieldType  # DD or WFD
    # this is the coaddition per night
    config['Observations']['coadd'] = fcoadd
    # season to simulate (-1 = all seasons)
    config['Observations']['season'] = seasval

    # --------------------simulator -------------------------
    config['Simulator'] = {}
    config['Simulator']['name'] = 'sn_simulator.{}'.format(
        simulator)    # Simulator name= sn_cosmo,sn_sim,sn_ana, sn_fast
    config['Simulator']['model'] = 'salt2-extended'   # spectra model
    config['Simulator']['version'] = 1.0  # version
    # Reference File= SN_MAF/Reference_Files/LC_Ref_-2.0_0.2.hdf5
    config['Simulator']['Template Dir'] = '.'
    config['Simulator']['Gamma File'] = 'reference_files/gamma.hdf5'

    # -------------------------host ---------------------
    config['Host Parameters'] = None         # Host parameters

    # --------------------miscellanea ----------------
    config['Display_LC'] = {}  # display during LC simulations
    config['Display_LC']['display'] = False
    # display during time (sec) before closing
    config['Display_LC']['time'] = 30

    config['Output'] = {}
    config['Output']['directory'] = 'Output_Simu'
    config['Output']['save'] = True

    config['Multiprocessing'] = {}
    config['Multiprocessing']['nproc'] = nproc

    config['Metric'] = 'sn_mafsim.sn_maf_simulation'

    config['Pixelisation'] = {}
    config['Pixelisation']['nside'] = nside

    return config


def Observations_band(day0=59000, daymin=59000, cadence=3., season_length=140., band='r'):
    # Define fake data
    names = ['observationStartMJD', 'fieldRA', 'fieldDec',
             'fiveSigmaDepth', 'visitExposureTime', 'numExposures',
             'visitTime', 'seeingFwhmEff', 'seeingFwhmGeom',
             'pixRA', 'pixDec', 'RA', 'Dec', 'airmass', 'sky', 'moonPhase']
    types = ['f8']*len(names)
    names += ['night', 'healpixID']
    types += ['i2', 'i2']
    names += ['filter']
    types += ['O']

    daylast = daymin+season_length
    cadence = cadence
    dayobs = np.arange(daymin, daylast, cadence)
    npts = len(dayobs)
    data = np.zeros(npts, dtype=list(zip(names, types)))
    data['observationStartMJD'] = dayobs
    data['night'] = np.floor(data['observationStartMJD']-day0+1)
    # data['night'] = 10
    data['fiveSigmaDepth'] = m5_ref[band]
    data['visitExposureTime'] = 30.
    data['numExposures'] = 1
    data['visitTime'] = 34
    data['filter'] = band
    # data['season'] = 1.
    data['seeingFwhmEff'] = 0.8
    data['seeingFwhmGeom'] = 0.8
    data['healpixID'] = 10
    data['pixRA'] = 0.0
    data['pixDec'] = 0.0
    data['RA'] = 0.0
    data['Dec'] = 0.0
    data['airmass'] = 1.2
    data['sky'] = 1.2
    data['moonPhase'] = 0.5
    return data


def Observations_season(day0=59000, mjdmin=59000, cadence=3.):
    bands = 'grizy'
    Nvisits = dict(zip(bands, [10, 20, 20, 26, 20]))
    rat = 34./3600./24.
    shift_visits = {}
    shift_visits['g'] = 0
    shift_visits['r'] = rat*Nvisits['g']
    shift_visits['i'] = rat*Nvisits['r']
    shift_visits['z'] = rat*Nvisits['i']
    shift_visits['y'] = rat*Nvisits['z']

    # get data
    data = None
    season_length = 180
    shift = 30./(3600.*24)
    for band in bands:

        mjd = mjdmin+shift_visits[band]
        for i in range(Nvisits[band]):
            mjd += shift
            dat = Observations_band(
                daymin=mjd, season_length=season_length, cadence=cadence, band=band)
            if data is None:
                data = dat
            else:
                data = np.concatenate((data, dat))

    return data


class TestSNsimulation(unittest.TestCase):

    def testSimuSNCosmo(self):
        # set simulation parameters
        prodid = 'Fake'
        # x1colorType = 'unique'
        x1Type = 'unique'
        x1min = -2.0
        x1max = 2.0
        x1step = 0.1
        colorType = 'unique'
        colormin = 0.2
        colormax = 0.3
        colorstep = 0.02
        zType = 'uniform'
        zmin = 0.1
        zmax = 0.8
        zstep = 0.1
        daymaxtype = 'unique'
        daymaxstep = 1.
        difflux = 0
        fulldbName = 'data_from_fake'
        fieldType = 'Fake'
        fcoadd = 1
        seasval = [1]

        # get the config file from these
        conf = getconfig(prodid,
                         x1Type, x1min, x1max, x1step,
                         colorType, colormin, colormax, colorstep,
                         zType, zmin, zmax, zstep,
                         daymaxtype, daymaxstep, difflux,
                         fulldbName, fieldType, fcoadd, seasval)

        # SN_Simulation instance
        getRefDir('SALT2_Files')
        getRefDir('reference_files')

        absMag = conf['SN parameters']['absmag']
        x0normFile = 'reference_files/X0_norm_{}.npy'.format(absMag)

        x0_norm = np.load(x0normFile)

        area = 9.6  # survey area (deg2)

        simu = SNSimulation(mjdCol='observationStartMJD',
                            filterCol='filter',
                            nexpCol='numExposures',
                            exptimeCol='visitExposureTime',
                            config=conf, x0_norm=x0_norm)

        # Generate fake data
        day0 = 59000
        data = None

        diff_season = 280.
        nseasons = 1
        for val in np.arange(59000, 59000+nseasons*diff_season, diff_season):
            dat = Observations_season(day0, val)
            if data is None:
                data = dat
            else:
                data = np.concatenate((data, dat))

        # now simulate LC on this data

        simu.run(data)

        # save metadata

        simu.save_metadata()

        # check what we have inside the data

        simu_name = '{}/Simu_{}.hdf5'.format(
            conf['Output']['directory'], conf['ProductionID'])
        lc_name = '{}/LC_{}.hdf5'.format(
            conf['Output']['directory'], conf['ProductionID'])

        f = h5py.File(simu_name, 'r')
        # reading the simu file
        for i, key in enumerate(f.keys()):
            simul = Table.read(simu_name, path=key)

        # first check on simulation parameters
        """
        for cc in simul.columns:
            print('RefDict[\'{}\']='.format(cc), simul[cc].tolist())
        """

        RefDict = {}
        RefDict['SNID'] = [101, 102, 103, 104, 105, 106, 107, 108]
        RefDict['RA'] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        RefDict['Dec'] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        RefDict['x0'] = [0.0001740043427228556, 3.838217653816398e-05, 1.5291775006387726e-05, 7.813694452227428e-06,
                         4.593754146993177e-06, 2.958319253169902e-06, 2.031812357572594e-06, 1.4642465313185475e-06]
        RefDict['epsilon_x0'] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        RefDict['x1'] = [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0]
        RefDict['epsilon_x1'] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        RefDict['color'] = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
        RefDict['epsilon_color'] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        RefDict['daymax'] = [59023.10190972222, 59025.20190972222, 59027.30190972223, 59029.401909722226,
                             59031.501909722225, 59033.60190972222, 59035.70190972222, 59037.80190972223]
        RefDict['epsilon_daymax'] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        RefDict['z'] = [0.1, 0.2, 0.30000000000000004,
                        0.4, 0.5, 0.6, 0.7000000000000001, 0.8]
        RefDict['id_hdf5'] = ['10_1_101', '10_1_102', '10_1_103',
                              '10_1_104', '10_1_105', '10_1_106', '10_1_107', '10_1_108']
        RefDict['season'] = [1, 1, 1, 1, 1, 1, 1, 1]
        RefDict['fieldname'] = ['unk', 'unk', 'unk',
                                'unk', 'unk', 'unk', 'unk', 'unk']
        RefDict['fieldid'] = [0, 0, 0, 0, 0, 0, 0, 0]
        RefDict['n_lc_points'] = [116, 128, 140, 148, 160, 172, 135, 144]
        RefDict['survey_area'] = [0.8392936452111668, 0.8392936452111668, 0.8392936452111668,
                                  0.8392936452111668, 0.8392936452111668, 0.8392936452111668, 0.8392936452111668, 0.8392936452111668]
        RefDict['pixID'] = [10, 10, 10, 10, 10, 10, 10, 10]
        RefDict['pixRA'] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        RefDict['pixDec'] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        RefDict['dL'] = [447513.8270462983, 952843.778897064, 1509584.9325567451, 2111826.765549938,
                         2754245.369069551, 3432131.997266213, 4141376.3917280077, 4878422.389153463]

        for key, vv in RefDict.items():
            if key not in ['id_hdf5', 'fieldname']:
                assert(np.isclose(vv, simul[key].tolist()).all())

        # now grab LC

        vars = ['snr_m5', 'flux_e_sec', 'mag',
                'exptime', 'magerr', 'band', 'phase']

        for simu in simul:
            lc = Table.read(lc_name, path='lc_{}'.format(simu['id_hdf5']))
            idx = lc['snr_m5'] >= 5.
            lc = lc[idx][:20]
            break

        """
        for cc in vars:
            print('RefDict[\'{}\']='.format(cc), lc[cc].tolist())
        """
        RefDict = {}

        RefDict['snr_m5'] = [27.772923620356103, 18.983342162107302, 72.79593984362035, 152.87977223849938, 156.0685865832734, 29.621406914295484, 285.0462993006308, 367.0592232519814, 359.1614412531729,
                             171.59490954768887, 506.5917937548474, 646.3234598972053, 598.4834333799382, 288.7282674337122, 661.327366034326, 850.0564702893679, 734.2166561609322, 389.3250502768923, 745.0477332694176, 967.0798070200309]
        RefDict['flux_e_sec'] = [38.63048481688093, 30.05326359299371, 121.88545019060624, 229.46055195746084, 265.7354374367385, 47.61100427319823, 597.0568637060227, 626.6704052212074, 680.5620770907243,
                                 292.05747499012944, 1321.1124849045573, 1298.1026162913374, 1282.7999663138387, 515.1045023970963, 1985.7957890733053, 1911.397717772466, 1684.5021456227396, 723.0824208367617, 2403.864481616118, 2314.5019489575334]
        RefDict['mag'] = [24.1975402954205, 24.157655378648695, 23.160953686205854, 22.263095607161585, 21.791260794726814, 23.244285332871133, 21.4357941123162, 21.172267728375726, 20.77021511674539, 21.27488287316678,
                          20.573483858618765, 20.381593128907532, 20.081987130304093, 20.65881532367227, 20.13099688702663, 19.96148803417519, 19.78620555869124, 20.290584180196035, 19.923558399162097, 19.753721814921928]
        RefDict['exptime'] = [600.0, 600.0, 300.0, 600.0, 600.0, 780.0, 300.0, 600.0, 600.0,
                              780.0, 300.0, 600.0, 600.0, 780.0, 300.0, 600.0, 600.0, 780.0, 300.0, 600.0]
        RefDict['magerr'] = [0.039093334918558646, 0.057194154511178236, 0.014914790675008787, 0.007101895750239159, 0.006956788861407506, 0.03665376894147948, 0.003808981935292667, 0.002957931952067543, 0.0030229754089687874, 0.00632732175808739,
                             0.002143217119074662, 0.0016798650708591183, 0.0018141457961942716, 0.0037604084089459607, 0.001641753026596777, 0.0012772518564425886, 0.0014787681478587269, 0.0027887653362811916, 0.0014572706637113076, 0.00112269555922559]
        RefDict['band'] = ['LSST::r', 'LSST::i', 'LSST::g', 'LSST::r', 'LSST::i', 'LSST::z', 'LSST::g', 'LSST::r', 'LSST::i',
                           'LSST::z', 'LSST::g', 'LSST::r', 'LSST::i', 'LSST::z', 'LSST::g', 'LSST::r', 'LSST::i', 'LSST::z', 'LSST::g', 'LSST::r']
        RefDict['phase'] = [-15.54029882153951, -15.536721380459229, -12.818181818180495, -12.813026094266784, -12.809448653186502, -12.808501683486444, -10.090909090907767, -10.085753366994057, -10.082175925913774, -
                            10.081228956213717, -7.36363636363504, -7.35848063972133, -7.354903198641047, -7.353956228940991, -4.636363636362313, -4.631207912448603, -4.62763047136832, -4.626683501668263, -1.909090909089586, -1.9039351851758757]

        for key, vv in RefDict.items():
            if key not in ['band']:
                assert(np.isclose(vv, lc[key].tolist()).all())
            else:
                assert(set(vv) == set(lc[key].tolist()))

    def testSimuSNFast(self):
        # set simulation parameters
        prodid = 'Fake'
        # x1colorType = 'unique'
        x1Type = 'unique'
        x1min = -2.0
        x1max = 2.0
        x1step = 0.1
        colorType = 'unique'
        colormin = 0.2
        colormax = 0.3
        colorstep = 0.02
        zType = 'uniform'
        zmin = 0.1
        zmax = 0.8
        zstep = 0.1
        daymaxtype = 'unique'
        daymaxstep = 1.
        difflux = 0
        fulldbName = 'data_from_fake'
        fieldType = 'Fake'
        fcoadd = 1
        seasval = [1]

        # get the config file from these
        conf = getconfig(prodid,
                         x1Type, x1min, x1max, x1step,
                         colorType, colormin, colormax, colorstep,
                         zType, zmin, zmax, zstep,
                         daymaxtype, daymaxstep, difflux,
                         fulldbName, fieldType, fcoadd, seasval,
                         simulator='sn_fast')

        # get the reference LC file

        referenceName = 'LC_{}_{}_vstack.hdf5'.format(x1min, colormin)
        getFile('Templates', referenceName)

        # instance of SNSimulation
        simu = SNSimulation(mjdCol='observationStartMJD',
                            filterCol='filter',
                            nexpCol='numExposures',
                            exptimeCol='visitExposureTime',
                            config=conf)

        # Generate fake data
        day0 = 59000
        data = None

        diff_season = 280.
        nseasons = 1
        for val in np.arange(59000, 59000+nseasons*diff_season, diff_season):
            dat = Observations_season(day0, val)
            if data is None:
                data = dat
            else:
                data = np.concatenate((data, dat))

        # now simulate LC on this data

        tab = simu.run(data)

        # save metadata

        simu.save_metadata()

        # check what we have inside the data

        simu_name = '{}/Simu_{}.hdf5'.format(
            conf['Output']['directory'], conf['ProductionID'])
        lc_name = '{}/LC_{}.hdf5'.format(
            conf['Output']['directory'], conf['ProductionID'])

        f = h5py.File(simu_name, 'r')
        # reading the simu file
        for i, key in enumerate(f.keys()):
            simul = Table.read(simu_name, path=key)

        # first check on simulation
        """
        for cc in simul.columns:
            print('RefDict[\'{}\']='.format(cc), simul[cc].tolist())
        """
        RefDict = {}
        RefDict['SNID'] = [101, 102, 103, 104, 105, 106, 107, 108]
        RefDict['RA'] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        RefDict['Dec'] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        RefDict['x0'] = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
        RefDict['epsilon_x0'] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        RefDict['x1'] = [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0]
        RefDict['epsilon_x1'] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        RefDict['color'] = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
        RefDict['epsilon_color'] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        RefDict['daymax'] = [59023.10190972222, 59025.20190972222, 59027.30190972223, 59029.401909722226,
                             59031.501909722225, 59033.60190972222, 59035.70190972222, 59037.80190972223]
        RefDict['epsilon_daymax'] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        RefDict['z'] = [0.1, 0.2, 0.30000000000000004,
                        0.4, 0.5, 0.6, 0.7000000000000001, 0.8]
        RefDict['id_hdf5'] = ['101', '102', '103',
                              '104', '105', '106', '107', '108']
        RefDict['season'] = [1, 1, 1, 1, 1, 1, 1, 1]
        RefDict['fieldname'] = ['unk', 'unk', 'unk',
                                'unk', 'unk', 'unk', 'unk', 'unk']
        RefDict['fieldid'] = [0, 0, 0, 0, 0, 0, 0, 0]
        RefDict['n_lc_points'] = [88, 96, 104, 112, 120, 128, 102, 108]
        RefDict['survey_area'] = [0.8392936452111668, 0.8392936452111668, 0.8392936452111668,
                                  0.8392936452111668, 0.8392936452111668, 0.8392936452111668, 0.8392936452111668, 0.8392936452111668]
        RefDict['pixID'] = [10, 10, 10, 10, 10, 10, 10, 10]
        RefDict['pixRA'] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        RefDict['pixDec'] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        RefDict['dL'] = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]

        for key, vv in RefDict.items():
            if key not in ['id_hdf5', 'fieldname']:
                assert(np.isclose(vv, simul[key].tolist()).all())

        # now grab LC

        vars = ['snr_m5', 'flux_e_sec', 'mag',
                'exptime', 'magerr', 'band', 'phase']

        for simu in simul:
            lc = Table.read(lc_name, path='lc_{}'.format(simu['id_hdf5']))
            idx = lc['snr_m5'] >= 5.
            lc = lc[idx][:20]
            break

        """
        for cc in vars:
            print('RefDict[\'{}\']='.format(cc), lc[cc].tolist())
        """
        RefDict = {}
        RefDict['snr_m5'] = [73.43420610434742, 283.33317127966245, 499.9323483273812, 650.0445290499508, 730.9094020380237, 737.1805610088064, 690.0470243527778, 608.6652253891193, 504.1773404482862, 403.6131375689242,
                             313.37158557134967, 238.66226160308508, 187.4911237193539, 158.79633621940252, 125.5922386662526, 100.22733937394999, 90.02593165899779, 86.15059662496986, 82.41581639581872, 78.79113804120628]
        RefDict['flux_e_sec'] = [123.6580798711139, 604.0931661593202, 1337.3476393884694, 2008.2103610697675, 2428.626413721442, 2463.0016660331135, 2210.91986650868, 1809.2730842763335, 1354.4292762650773,
                                 978.6279463735051, 690.3267842547574, 484.4212603063286, 359.4189128946231, 294.71705055807524, 224.47205877381782, 174.03032407514914, 154.50456888891176, 147.19958383247968, 140.2167124565155, 133.49508096240703]
        RefDict['mag'] = [23.14528858064343, 21.423083353285417, 20.56022356557598, 20.11881516319673, 19.912440047007113, 19.89718092156664, 20.014412194983553, 20.23208061893674, 20.54645306914551, 20.899292352785235,
                          21.278203341148284, 21.662784602641608, 21.986840998240314, 22.202328407512812, 22.497935185203982, 22.77428229900593, 22.90349048802979, 22.956077880337197, 23.00883863800245, 23.062178024888865]
        RefDict['exptime'] = [300.0, 300.0, 300.0, 300.0, 300.0, 300.0, 300.0, 300.0, 300.0,
                              300.0, 300.0, 300.0, 300.0, 300.0, 300.0, 300.0, 300.0, 300.0, 300.0, 300.0]
        RefDict['magerr'] = [0.014785156160268641, 0.0038320123261757423, 0.002171766256755872, 0.0016702489694743652, 0.0014854593493129628, 0.0014728226192947038, 0.0015734235007773332, 0.0017837986457399778, 0.002153480764908541,
                             0.0026900417843131294, 0.003464692571850694, 0.004549258007802665, 0.005790867232644643, 0.006837287500500082, 0.008644930740054348, 0.010832734975705863, 0.012060260691005175, 0.012602770581897977, 0.013173881570785641, 0.013779927943042398]
        RefDict['band'] = ['LSST::g', 'LSST::g', 'LSST::g', 'LSST::g', 'LSST::g', 'LSST::g', 'LSST::g', 'LSST::g', 'LSST::g',
                           'LSST::g', 'LSST::g', 'LSST::g', 'LSST::g', 'LSST::g', 'LSST::g', 'LSST::g', 'LSST::g', 'LSST::g', 'LSST::g', 'LSST::g']
        RefDict['phase'] = [-12.818181818180495, -10.090909090907767, -7.36363636363504, -4.636363636362313, -1.909090909089586, 0.8181818181831411, 3.545454545455868, 6.272727272728595, 9.000000000001322,
                            11.72727272727405, 14.454545454546777, 17.181818181819505, 19.90909090909223, 22.63636363636496, 25.363636363637685, 28.09090909091041, 30.818181818183138, 33.54545454545587, 36.272727272728595, 39.00000000000132]

        for key, vv in RefDict.items():
            if key not in ['band']:
                assert(np.isclose(vv, lc[key].tolist()).all())
            else:
                assert(set(vv) == set(lc[key].tolist()))


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main(verbosity=5)
