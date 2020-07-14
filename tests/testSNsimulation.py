from builtins import zip
import numpy as np
import unittest
import lsst.utils.tests
import os
from numpy.testing import assert_almost_equal, assert_equal
import h5py
from astropy.table import Table, vstack
from sn_simu_wrapper.sn_simu import SNSimulation
from sn_tools.sn_io import check_get_file


main_repo = 'https://me.lsst.eu/gris/DESC_SN_pipeline'
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
    # MW dust
    config['SN parameters']['ebvofMW'] = 0
    # NSN factor
    config['SN parameters']['NSN factor'] = 1

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
    config['Simulator']['Template Dir'] = 'Template_LC'
    config['Simulator']['Gamma Dir'] = 'reference_files'
    config['Simulator']['Gamma File'] = 'gamma.hdf5'
    config['Simulator']['DustCorr Dir'] = 'Template_Dust'
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
    config['Web path'] = 'https://me.lsst.eu/gris/DESC_SN_pipeline'

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
        # getRefDir('SALT2_Files')
        # getRefDir('reference_files')

        absMag = conf['SN parameters']['absmag']
        x0normFile = 'reference_files/X0_norm_{}.npy'.format(absMag)

        if not os.path.isfile(x0normFile):
            # if this file does not exist, grab it from a web server
            check_get_file(conf['Web path'], 'reference_files',
                           'X0_norm_{}.npy'.format(absMag))
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

        simu_name = '{}/Simu_{}_1.hdf5'.format(
            conf['Output']['directory'], conf['ProductionID'])
        lc_name = '{}/LC_{}_1.hdf5'.format(
            conf['Output']['directory'], conf['ProductionID'])

        f = h5py.File(simu_name, 'r')
        # reading the simu file
        simul = Table()
        for i, key in enumerate(f.keys()):
            simul = vstack([simul, Table.read(simu_name, path=key)])

        # first check on simulation parameters

        """
        for cc in simul.columns:
            print('RefDict[\'{}\']='.format(cc), simul[cc].tolist())
        """
        RefDict = {}
        RefDict['SNID'] = [11, 12, 13, 14, 15, 16, 17, 18]
        RefDict['index_hdf5'] = ['10_-2.0_0.2_0.1_59023.102_1_0', '10_-2.0_0.2_0.2_59025.202_1_0', '10_-2.0_0.2_0.3_59027.302_1_0', '10_-2.0_0.2_0.4_59029.402_1_0',
                                 '10_-2.0_0.2_0.5_59031.502_1_0', '10_-2.0_0.2_0.6_59033.602_1_0', '10_-2.0_0.2_0.7_59035.702_1_0', '10_-2.0_0.2_0.8_59037.802_1_0']
        RefDict['season'] = [1, 1, 1, 1, 1, 1, 1, 1]
        RefDict['fieldname'] = ['unknown', 'unknown', 'unknown',
                                'unknown', 'unknown', 'unknown', 'unknown', 'unknown']
        RefDict['fieldid'] = [0, 0, 0, 0, 0, 0, 0, 0]
        RefDict['n_lc_points'] = [99, 106, 114, 127, 132, 142, 112, 119]
        RefDict['area'] = [0.8392936452111668, 0.8392936452111668, 0.8392936452111668, 0.8392936452111668,
                           0.8392936452111668, 0.8392936452111668, 0.8392936452111668, 0.8392936452111668]
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
        RefDict['survey_area'] = [0.8392936452111668, 0.8392936452111668, 0.8392936452111668,
                                  0.8392936452111668, 0.8392936452111668, 0.8392936452111668, 0.8392936452111668, 0.8392936452111668]
        RefDict['healpixID'] = [10, 10, 10, 10, 10, 10, 10, 10]
        RefDict['pixRA'] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        RefDict['pixDec'] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        RefDict['dL'] = [447513.8270462983, 952843.778897064, 1509584.9325567451, 2111826.765549938,
                         2754245.369069551, 3432131.997266213, 4141376.3917280077, 4878422.389153463]
        RefDict['snr_fluxsec_meth'] = ['interp', 'interp', 'interp',
                                       'interp', 'interp', 'interp', 'interp', 'interp']
        RefDict['status'] = [1, 1, 1, 1, 1, 1, 1, 1]
        RefDict['ebvofMW'] = [0, 0, 0, 0, 0, 0, 0, 0]

        for key, vv in RefDict.items():
            if key not in ['index_hdf5', 'fieldname', 'snr_fluxsec_meth']:
                assert(np.isclose(vv, simul[key].tolist()).all())
            else:
                assert((vv == simul[key]).all())

        # now grab LC

        vars = ['snr_m5', 'flux_e_sec', 'mag',
                'exptime', 'magerr', 'band', 'phase']

        for simu in simul:
            lc = Table.read(lc_name, path='lc_{}'.format(simu['index_hdf5']))
            idx = lc['snr_m5'] >= 5.
            lc = lc[idx][:20]
            break

        """
        for cc in vars:
            print('RefDict[\'{}\']='.format(cc), lc[cc].tolist())
        """

        RefDict = {}

        RefDict['snr_m5'] = [27.77258180379027, 18.98377096566505, 72.79590570928633, 152.87684647830162, 156.06666693213177, 29.621529233519308, 285.0437939047469, 367.04307444774975, 359.1500294310009,
                             171.59301350412957, 506.58512140687225, 646.2792675349654, 598.4544644527755, 288.72360876534253, 661.3175452293224, 849.9877278958845, 734.1754283838669, 389.3160994765804, 745.0362859344243, 966.9961020792483]
        RefDict['flux_e_sec'] = [38.63207940525376, 30.058075022627055, 121.90767409695435, 229.50791298666263, 265.77760354105226, 47.616478031948624, 597.1868291509262, 626.8365327579263, 680.7371214592486,
                                 292.13548602917916, 1321.4612615401911, 1298.4282257064317, 1283.1179646726964, 515.1858166393667, 1986.2944122972094, 1911.7598017542887, 1684.8650500083472, 723.2044598429111, 2404.500751077161, 2314.677056897036]
        RefDict['mag'] = [24.197549880640725, 24.157628846869798, 23.160950966140664, 22.263092425796074, 21.79125631239912, 23.244278569706797, 21.435793513964224, 21.17226381616802, 20.77021232574782,
                          21.27488067257211, 20.5734837320662, 20.381589712793957, 20.081984493733383, 20.658809391021087, 20.130996540814564, 19.96148438529608, 19.7862032720865, 20.290578693537636, 19.92355764391396, 19.75371788930898]
        RefDict['exptime'] = [600.0, 600.0, 300.0, 600.0, 600.0, 780.0, 300.0, 600.0, 600.0,
                              780.0, 300.0, 600.0, 600.0, 780.0, 300.0, 600.0, 600.0, 780.0, 300.0, 600.0]
        RefDict['magerr'] = [0.039093816067541594, 0.05719286261522241, 0.014914797668622531, 0.007102031666464496, 0.006956874431298518, 0.03665361758330578, 0.003809015414385587, 0.0029580620922809125, 0.0030230714625814016, 0.0063273916728084035,
                             0.0021432453478753125, 0.0016799799394762239, 0.0018142336121611566, 0.0037604690846066266, 0.0016417774072236856, 0.0012773551536394909, 0.0014788511883980504, 0.0027888294530276484, 0.001457293054386471, 0.001122792741794475]
        RefDict['band'] = ['LSST::r', 'LSST::i', 'LSST::g', 'LSST::r', 'LSST::i', 'LSST::z', 'LSST::g', 'LSST::r', 'LSST::i',
                           'LSST::z', 'LSST::g', 'LSST::r', 'LSST::i', 'LSST::z', 'LSST::g', 'LSST::r', 'LSST::i', 'LSST::z', 'LSST::g', 'LSST::r']
        RefDict['phase'] = [-15.54029882153951, -15.536721380459229, -12.818181818180495, -12.813026094266784, -12.809448653186502, -12.808501683486444, -10.090909090907767, -10.085753366994057, -10.082175925913774, -
                            10.081228956213717, -7.36363636363504, -7.35848063972133, -7.354903198641047, -7.353956228940991, -4.636363636362313, -4.631207912448603, -4.62763047136832, -4.626683501668263, -1.909090909089586, -1.9039351851758757]

        for key, vv in RefDict.items():
            #print('testing', key)
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

        #referenceName = 'LC_{}_{}_vstack.hdf5'.format(x1min, colormin)
        #getFile('Templates', referenceName)

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

        simu_name = '{}/Simu_{}_1.hdf5'.format(
            conf['Output']['directory'], conf['ProductionID'])
        lc_name = '{}/LC_{}_1.hdf5'.format(
            conf['Output']['directory'], conf['ProductionID'])

        f = h5py.File(simu_name, 'r')
        # reading the simu file
        simul = Table()
        for i, key in enumerate(f.keys()):
            simul = vstack([simul, Table.read(simu_name, path=key)])

        # first check on simulation
        """
        for cc in simul.columns:
            print('RefDict[\'{}\']='.format(cc), simul[cc].tolist())
        """
        RefDict = {}

        RefDict['SNID'] = [11, 12, 13, 14, 15, 16, 17, 18]
        RefDict['index_hdf5'] = ['10_-2.0_0.2_0.1_59023.1_1', '10_-2.0_0.2_0.2_59025.2_1', '10_-2.0_0.2_0.3_59027.3_1', '10_-2.0_0.2_0.4_59029.4_1',
                                 '10_-2.0_0.2_0.5_59031.5_1', '10_-2.0_0.2_0.6_59033.6_1', '10_-2.0_0.2_0.7_59035.7_1', '10_-2.0_0.2_0.8_59037.8_1']
        RefDict['season'] = [1, 1, 1, 1, 1, 1, 1, 1]
        RefDict['fieldname'] = ['unknown', 'unknown', 'unknown',
                                'unknown', 'unknown', 'unknown', 'unknown', 'unknown']
        RefDict['fieldid'] = [0, 0, 0, 0, 0, 0, 0, 0]
        RefDict['n_lc_points'] = [88, 96, 104, 112, 120, 128, 102, 108]
        RefDict['area'] = [0.8392936452111668, 0.8392936452111668, 0.8392936452111668, 0.8392936452111668,
                           0.8392936452111668, 0.8392936452111668, 0.8392936452111668, 0.8392936452111668]
        RefDict['z'] = [0.1, 0.2, 0.30000000000000004,
                        0.4, 0.5, 0.6, 0.7000000000000001, 0.8]
        RefDict['daymax'] = [59023.10190972222, 59025.20190972222, 59027.30190972223, 59029.401909722226,
                             59031.501909722225, 59033.60190972222, 59035.70190972222, 59037.80190972223]
        RefDict['x1'] = [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0]
        RefDict['color'] = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
        RefDict['x0'] = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
        RefDict['epsilon_x0'] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        RefDict['epsilon_x1'] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        RefDict['epsilon_color'] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        RefDict['epsilon_daymax'] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        RefDict['RA'] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        RefDict['Dec'] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        RefDict['pixRA'] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        RefDict['pixDec'] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        RefDict['healpixID'] = [10, 10, 10, 10, 10, 10, 10, 10]
        RefDict['dL'] = [-1, -1, -1, -1, -1, -1, -1, -1]

        for key, vv in RefDict.items():
            if key not in ['index_hdf5', 'fieldname']:
                assert(np.isclose(vv, simul[key].tolist()).all())

        # now grab LC

        vars = ['snr_m5', 'flux_e_sec', 'mag',
                'visitExposureTime', 'magerr', 'band', 'phase']

        for simu in simul:
            lc = Table.read(lc_name, path='lc_{}'.format(simu['index_hdf5']))
            idx = lc['snr_m5'] >= 5.
            lc = lc[idx][:20]
            break

        """
        for cc in vars:
            print('RefDict[\'{}\']='.format(cc), lc[cc].tolist())
        """

        RefDict = {}

        RefDict['snr_m5'] = [72.76621556433962, 284.91155929878005, 506.3382490861352, 661.0244890696117, 744.7374346856301, 750.9560627892503, 701.5801119418779, 617.0362767220539, 509.49758960144334,
                             405.6704063879359, 312.1587215082942, 236.20474351258605, 185.20525529387027, 156.7007460981387, 123.82556613113591, 98.86188081087201, 89.07300629269548, 85.39942468306856, 81.6029185267763, 77.71841283547786]
        RefDict['flux_e_sec'] = [121.85475114199077, 596.8249003571814, 1320.5101003419977, 1984.8958672777107, 2402.9349000717466, 2435.4188847893047, 2182.188549348978, 1781.5190648980197, 1332.6543244852983,
                                 959.9063595202888, 672.3614552092635, 470.38517751045663, 349.6450546810812, 287.0414484278593, 219.024995436545, 170.22546308092205, 151.80546033037837, 144.97271315775106, 137.96525588832066, 130.84706269377557]
        RefDict['mag'] = [23.161428491773833, 21.436445105957933, 20.574266452603773, 20.131756917960203, 19.9242657576835, 19.90957575410812, 20.028884191267565, 20.248882689684862, 20.564272843036967, 20.920539051651428,
                          21.30696280634007, 21.694807732877024, 22.017022106232524, 22.23124270951677, 22.524886922545807, 22.798310177888723, 22.922900267535, 22.972903089242255, 23.02669579575968, 23.084173861687447]
        RefDict['visitExposureTime'] = [300.0, 300.0, 300.0, 300.0, 300.0, 300.0, 300.0, 300.0,
                                        300.0, 300.0, 300.0, 300.0, 300.0, 300.0, 300.0, 300.0, 300.0, 300.0, 300.0, 300.0]
        RefDict['magerr'] = [0.014920883219467768, 0.003810783274045906, 0.0021442903172290873, 0.0016425052667659818, 0.00145787784283523, 0.0014458052322334502, 0.0015475584131838627, 0.0017595986584873079, 0.0021309938003974682,
                             0.00267639982523117, 0.0034781543168553785, 0.004596589334372434, 0.0058623401535521326, 0.006928723900766582, 0.008768271679923466, 0.010982354329624782, 0.012189284385331971, 0.012713624345684724, 0.013305114870393606, 0.013970128379442394]
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
