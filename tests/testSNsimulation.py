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
from sn_simu_wrapper.config_simulation import ConfigSimulation

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

"""
def getconfig(prodid,sn_type,sn_model,sn_model_version,
              x1Type, x1min, x1max, x1step,
              colorType, colormin, colormax, colorstep,
              zType, zmin, zmax, zstep,
              daymaxType, daymaxstep, diffflux,
              fulldbName, fieldType, fcoadd, seasval,ebvofMW=0.0,bluecutoff=380.,redcutoff=800.,error_model=0,display=False,
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
            filedata = filedata.replace('thedisp', str(display))
            
    return yaml.load(filedata, Loader=yaml.FullLoader)
"""
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

def getSimu(config_name):
    
    # get the config file from these
    ffi = open(config_name)
    conf =yaml.load(ffi, Loader=yaml.FullLoader)
    ffi.close()
    
    #print('conf',conf)
    absMag = conf['SN']['absmag']
    x0normFile = 'reference_files/X0_norm_{}.npy'.format(absMag)

    if not os.path.isfile(x0normFile):
        # if this file does not exist, grab it from a web server
        check_get_file(conf['WebPath'], 'reference_files',
                       'X0_norm_{}.npy'.format(absMag))
    x0_norm = np.load(x0normFile)
    
    area = 9.6  # survey area (deg2)

    simu = SNSimulation(mjdCol='observationStartMJD',
                        filterCol='filter',
                        nexpCol='numExposures',
                        exptimeCol='visitExposureTime',
                        config=conf, x0_norm=x0_norm)

    return simu,conf

def dump(fname, thedict):

    #with open(fname, 'w') as ffile:
    ffile = open(fname,'w')
    documents = yaml.dump(thedict, ffile)
    ffile.close()
    
def testSimu(data,config_name):
    
    simu,conf = getSimu(config_name)
    name_config = config_name.split('/')[-1].split('.')[0]
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
        
    ref_simul = 'data_tests/ref_simu_{}.hdf5'.format(name_config)

    if not os.path.exists(ref_simul):
        simul.write(ref_simul,'simu_parameters', append=True, compression=True)

    # load reference parameters
    tab_simu_ref = Table.read(ref_simul,path='simu_parameters')
    if 'ptime' in tab_simu_ref.columns:
        tab_simu_ref.remove_column('ptime')

   
    for key in tab_simu_ref.columns:
        if key not in ['index_hdf5', 'fieldname', 'snr_fluxsec_meth','sn_type','sn_model','sn_version']:
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
  
    ref_lc =  'data_tests/ref_lc_{}.hdf5'.format(name_config)
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
        # test Ia - no error model
        # get configuration file
        config = ConfigSimulation('Ia','salt2','../sn_simu_input/config_simulation.txt').conf_dict
        config['ProductionID'] = 'prod_Ia_sncosmo_errormodel_0'
        config['Simulator']['errorModel']=0
        config['SN']['z']['type'] = 'uniform'
        config['SN']['z']['step'] = 0.1
        config['SN']['z']['min'] = 0.1
        config['SN']['z']['max'] = 1.0
        """
        config['SN']['x1']['type'] = 'random'
        config['SN']['color']['type'] = 'random'
        config['Display']['LC']['display'] = 1
        """
        fname = 'config2.yaml'
        dump(fname,config)
        testSimu(data,fname)

        # test Ia - with error model
        config['Simulator']['errorModel']=1
        config['ProductionID'] = 'prod_Ia_sncosmo_errormodel_1'
        fname = 'config1.yaml'
        dump(fname,config)
        testSimu(data,fname)
        

        
        # test non Ia - error_model=0
        config['ProductionID'] = 'prod_Ib_sncosmo_errormodel_0'
        error_model = 0
        sn_type='SN_Ib'
        sn_model = 'nugent-sn2p'
        sn_model_version = '1.2'
        config = ConfigSimulation(sn_type,sn_model,'../sn_simu_input/config_simulation.txt').conf_dict
        config['Simulator']['model'] = sn_model
        config['Simulator']['version'] = sn_model_version
        config['SN']['type'] = sn_type
        config['SN']['z']['type'] = 'uniform'
        config['SN']['z']['step'] = 0.1
        config['SN']['z']['min'] = 0.01
        config['SN']['z']['max'] = 1.0
        # test Ib
        fname = 'config3.yaml'
        dump(fname,config)
        testSimu(data,fname)
        

    def testSimuSNFast(self):

        # fake data
        data = fake_data()

        config = ConfigSimulation('Ia','salt2','../sn_simu_input/config_simulation.txt').conf_dict
        config['ProductionID'] = 'prod_Ia_snfast_errormodel_0'
        config['Simulator']['errorModel']=0
        config['SN']['z']['type'] = 'uniform'
        config['SN']['z']['step'] = 0.1
        config['SN']['z']['min'] = 0.1
        config['SN']['z']['max'] = 1.0

        config['Simulator']['name'] = 'sn_simulator.sn_fast'
        """
        config['SN']['x1']['type'] = 'random'
        config['SN']['color']['type'] = 'random'
       
        config['Display']['LC']['display'] = 1
        """
        fname = 'config4.yaml'
        dump(fname,config)
        testSimu(data,fname)


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main(verbosity=5)
