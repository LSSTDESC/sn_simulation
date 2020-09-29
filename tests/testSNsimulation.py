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
import yaml

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


def getconfig(prodid,sn_type,sn_model,sn_model_version,
              x1Type, x1min, x1max, x1step,
              colorType, colormin, colormax, colorstep,
              zType, zmin, zmax, zstep,
              daymaxType, daymaxstep, diffflux,
              fulldbName, fieldType, fcoadd, seasval,ebvofMW=0.0,bluecutoff=380.,redcutoff=800.,error_model=0,
              simu='sn_cosmo', nside=64, nproc=1, outputDir='Output_Simu',config_orig='param_simulation_gen.yaml'):

    prodid = prodid+'_'+simu
    with open(config_orig, 'r') as file:
            filedata = file.read()
            filedata = filedata.replace('prodid', prodid)
            filedata = filedata.replace('sn_type', sn_type)
            filedata = filedata.replace('x1Type', x1Type)
            filedata = filedata.replace('x1min', str(x1min))
            filedata = filedata.replace('x1max', str(x1max))
            filedata = filedata.replace('x1step', str(x1step))
            filedata = filedata.replace('colorType', colorType)
            filedata = filedata.replace('colormin', str(colormin))
            filedata = filedata.replace('colormax', str(colormax))
            filedata = filedata.replace('colorstep', str(colorstep))
            filedata = filedata.replace('zmin', str(zmin))
            filedata = filedata.replace('zmax', str(zmax))
            filedata = filedata.replace('zstep', str(zstep))
            filedata = filedata.replace('zType', zType)
            filedata = filedata.replace('daymaxType', daymaxType)
            filedata = filedata.replace('daymaxstep', str(daymaxstep))
            filedata = filedata.replace('fcoadd', str(fcoadd))
            filedata = filedata.replace('seasval', str(seasval))
            filedata = filedata.replace('mysimu', simu)
            filedata = filedata.replace('ebvofMWval', str(ebvofMW))
            filedata = filedata.replace('bluecutoffval', str(bluecutoff))
            filedata = filedata.replace('redcutoffval', str(redcutoff))
            filedata = filedata.replace('errmod', str(error_model))
            filedata = filedata.replace('sn_model', sn_model)
            filedata = filedata.replace('sn_mod_version', sn_model_version)
            filedata = filedata.replace('nnside', str(nside))
            filedata = filedata.replace('nnproc', str(nproc))
            filedata = filedata.replace('outputDir', outputDir)
            filedata = filedata.replace('diffflux', str(diffflux))

    return yaml.load(filedata, Loader=yaml.FullLoader)

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

def fake_data(day0 = 59000, diff_season = 280.,nseasons = 1):

    # Generate fake data
   
    data = None
    
    for val in np.arange(59000, 59000+nseasons*diff_season, diff_season):
        dat = Observations_season(day0, val)
        if data is None:
            data = dat
        else:
            data = np.concatenate((data, dat))

    return data

def getSimu(prodid,sn_type,sn_model,sn_model_version,
         x1Type, x1min, x1max, x1step,
         colorType, colormin, colormax, colorstep,
         zType, zmin, zmax, zstep,
         daymaxtype, daymaxstep, difflux,
            fulldbName, fieldType, fcoadd, seasval,error_model):
    
    # get the config file from these
    conf = getconfig(prodid,sn_type,sn_model,sn_model_version,
                     x1Type, x1min, x1max, x1step,
                     colorType, colormin, colormax, colorstep,
                     zType, zmin, zmax, zstep,
                     daymaxtype, daymaxstep, difflux,
                     fulldbName, fieldType, fcoadd, seasval,error_model=error_model)

    print('hello conf',conf,error_model)
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

    return simu,conf
        
def testSimu(data,prodid,sn_type,sn_model,sn_model_version,
                        x1Type, x1min, x1max, x1step,
                        colorType, colormin, colormax, colorstep,
                        zType, zmin, zmax, zstep,
                        daymaxtype, daymaxstep, difflux,
                        fulldbName, fieldType, fcoadd, seasval,error_model):
    
    simu,conf = getSimu(prodid,sn_type,sn_model,sn_model_version,
                        x1Type, x1min, x1max, x1step,
                        colorType, colormin, colormax, colorstep,
                        zType, zmin, zmax, zstep,
                        daymaxtype, daymaxstep, difflux,
                        fulldbName, fieldType, fcoadd, seasval,error_model)

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
        
    ref_simul = 'data_tests/ref_simu_{}_error_model_{}.hdf5'.format(sn_type,error_model)

    if not os.path.exists(ref_simul):
        simul.write(ref_simul,'simu_parameters', append=True, compression=True)

    # load reference parameters
    tab_simu_ref = Table.read(ref_simul,path='simu_parameters')
    if 'ptime' in tab_simu_ref.columns:
        tab_simu_ref.remove_column('ptime')
            
    for key in tab_simu_ref.columns:
        if key not in ['index_hdf5', 'fieldname', 'snr_fluxsec_meth']:
            assert(np.isclose(tab_simu_ref[key].tolist(), simul[key].tolist()).all())
        else:
            assert((tab_simu_ref[key]== simul[key]).all())
                
    # now grab LC

    vars = ['snr_m5', 'flux_e_sec', 'mag',
            'exptime', 'magerr', 'band', 'phase']

    for simu in simul:
        lc = Table.read(lc_name, path='lc_{}'.format(simu['index_hdf5']))
        idx = lc['snr_m5'] >= 5.
        lc = lc[idx][:20]
        break

    ref_lc =  'data_tests/ref_lc_{}_error_model_{}.hdf5'.format(sn_type,error_model)
    if not os.path.exists(ref_lc):
        lc.write(ref_lc,'lc_points', append=True, compression=True)

    # load lc reference points
    tab_lc_ref = Table.read(ref_lc,path='lc_points')
    if 'index' in tab_lc_ref.columns:
         tab_lc_ref.remove_column('index')
    
    for key in tab_lc_ref.columns:
        if key not in ['band','filter_cosmo','zpsys']:
            assert(np.isclose(tab_lc_ref[key].tolist(), lc[key].tolist()).all())
        else:
            assert(set(tab_lc_ref[key].tolist()) == set(lc[key].tolist()))


class TestSNsimulation(unittest.TestCase):

    def testSimuSNCosmo(self):

        # fake data
        data = fake_data()

        # test Ia simulation
        
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
        sn_type='SN_Ia'
        sn_model = 'salt2-extended'
        sn_model_version = '1.0'

        """
        # test Ia - no error model
        error_model = 0
        testSimu(data,prodid,sn_type,sn_model,sn_model_version,
                       x1Type, x1min, x1max, x1step,
                       colorType, colormin, colormax, colorstep,
                       zType, zmin, zmax, zstep,
                       daymaxtype, daymaxstep, difflux,
                       fulldbName, fieldType, fcoadd, seasval,error_model)

        # test Ia - with error model
        error_model = 1
        testSimu(data,prodid,sn_type,sn_model,sn_model_version,
                       x1Type, x1min, x1max, x1step,
                       colorType, colormin, colormax, colorstep,
                       zType, zmin, zmax, zstep,
                       daymaxtype, daymaxstep, difflux,
                       fulldbName, fieldType, fcoadd, seasval,error_model)
        """

        # test non Ia - error_model=0
        error_model = 0
        sn_type='SN_Ib'
        sn_model = 'nugent-sn2p'
        sn_model_version = '1.2'

        # test Ib
        testSimu(data,prodid,sn_type,sn_model,sn_model_version,
                       x1Type, x1min, x1max, x1step,
                       colorType, colormin, colormax, colorstep,
                       zType, zmin, zmax, zstep,
                       daymaxtype, daymaxstep, difflux,
                       fulldbName, fieldType, fcoadd, seasval,error_model)
        

    def testSimuSNFast():
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
        error_model = 0

        # get the config file from these
        conf = getconfig(prodid,
                         x1Type, x1min, x1max, x1step,
                         colorType, colormin, colormax, colorstep,
                         zType, zmin, zmax, zstep,
                         daymaxtype, daymaxstep, difflux,
                         fulldbName, fieldType, fcoadd, seasval,
                         error_model,simulator='sn_fast')

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
