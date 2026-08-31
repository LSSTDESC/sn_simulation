import pytest
from builtins import zip
import numpy as np
import unittest
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
    """
    Function to get file (web) named fname and located in refdir

    Parameters
    ----------
    refdir : str
        directory location of the file.
    fname : str
        file name.

    Returns
    -------
    None.

    """
    fullname = '{}/{}/{}'.format(main_repo, refdir, fname)

    # check whether the file is available; if not-> get it!
    if not os.path.isfile(fname):
        print('wget path:', fullname)
        cmd = 'wget --no-clobber --no-verbose {}'.format(fullname)
        os.system(cmd)


def getRefDir(dirname):
    """
    Function to grab a directory.

    Parameters
    ----------
    dirname : str
        directory name.

    Returns
    -------
    None.

    """
    fullname = '{}/{}'.format(main_repo, dirname)

    if not os.path.exists(dirname):
        print('wget path:', fullname)
        cmd = 'wget - -no-verbose - -recursive {} - -directory-prefix = {} \
        --no-clobber - -no-parent - nH - -cut-dirs = 3 - R \'index.html*\''.format(
            fullname+'/', dirname)
        os.system(cmd)


def Observations_band(day0=59000, daymin=59000, cadence=3.,
                      season_length=140., band='r'):
    """
    Function to generate observations per band

    Parameters
    ----------
    day0 : float, optional
        Min day for obs. The default is 59000.
    daymin : float, optional
        Min day for obs. The default is 59000.
    cadence : float. optional
        cadence of observation (in days). The default is 3..
    season_length : float, optional
        Season length (in days). The default is 140..
    band : str, optional
        filter for obs. The default is 'r'.

    Returns
    -------
    data : array
        array of obs.

    """
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
    """
    Function to generate observations per season

    Parameters
    ----------
    day0 : float, optional
        First day of obs. The default is 59000.
    mjdmin : float, optional
        min day for obs. The default is 59000.
    cadence : float, optional
        cadence of observations (in days). The default is 3..

    Returns
    -------
    data : array
        array of observations.

    """
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
                daymin=mjd, season_length=season_length,
                cadence=cadence, band=band)
            if data is None:
                data = dat
            else:
                data = np.concatenate((data, dat))

    return data


def fake_data(day0=59000, diff_season=280., nseasons=1):
    """
    Function to generate fake data

    Parameters
    ----------
    day0 : float, optional
        First day of obs. The default is 59000.
    diff_season : float, optional
        delta time (in days) between two seasons. The default is 280..
    nseasons : int, optional
        Number of seasons. The default is 1.

    Returns
    -------
    data : array
        array of obs.

    """

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
    """
    Function to perform simulations

    Parameters
    ----------
    config_name : yaml file
        Simulation parameters.

    Returns
    -------
    simu : class instance
        instance of SNSimulation.
    conf : dict
        suimulation parameters.

    """

    # get the config file from these
    ffi = open(config_name)
    conf = yaml.load(ffi, Loader=yaml.FullLoader)
    ffi.close()

    # print('conf',conf)
    absMag = conf['SN']['absmag']
    x0normFile = 'reference_files/X0_norm_{}.npy'.format(absMag)

    if not os.path.isfile(x0normFile):
        # if this file does not exist, grab it from a web server
        check_get_file(conf['WebPathSimu'], 'reference_files',
                       'X0_norm_{}.npy'.format(absMag))
    x0_norm = np.load(x0normFile)

    simu = SNSimulation(mjdCol='observationStartMJD',
                        filterCol='filter',
                        nexpCol='numExposures',
                        exptimeCol='visitExposureTime',
                        config=conf, x0_norm=x0_norm)

    return simu, conf


def dump(fname, thedict):
    """
    Function to dump a dict in a yaml file.

    Parameters
    ----------
    fname : str
        output file name.
    thedict : dict
        dict to dum in the yaml file.

    Returns
    -------
    None.

    """

    # with open(fname, 'w') as ffile:
    ffile = open(fname, 'w')
    documents = yaml.dump(thedict, ffile)
    ffile.close()


def simulation(data, config_name):
    """
    Functio to perform LC simulations

    Parameters
    ----------
    data : array
        observations used to simulate LCs.
    config_name : str
        yaml config name.

    Returns
    -------
    None.

    """

    simu, conf = getSimu(config_name)

    # now simulate LC on this data

    simu.run(data)

    # save metadata

    simu.save_metadata()


def gimeSimu(simu_name):

    f = h5py.File(simu_name, 'r')
    # reading the simu file
    simul = Table()
    for i, key in enumerate(f.keys()):
        simul = vstack([simul, Table.read(simu_name, path=key)])

    return simul


def gimerefSimu(config_name):
    """
    Functio to get reference simu params

    Parameters
    ----------
    config_name : str
        config file name.

    Returns
    -------
    tab_simu_ref : astropy table
        Reference simulation parameters.

    """

    name_config = config_name.split('/')[-1].split('.')[0]

    ref_simul = 'data_tests/ref_simu_{}.hdf5'.format(name_config)
    print('alors:::', ref_simul)
    # load reference parameters
    tab_simu_ref = Table.read(ref_simul, path='simu_parameters')
    if 'ptime' in tab_simu_ref.columns:
        tab_simu_ref.remove_column('ptime')

    return tab_simu_ref


def simuName(conf):
    """
    get the name of the simu parameter file

    Parameters
    ----------
    conf : dict
        simu parameters.

    Returns
    -------
    simu_name : str
        simu file name.

    """

    simu_name = '{}/Simu_{}_1.hdf5'.format(
        conf['OutputSimu']['directory'], conf['ProductionIDSimu'])

    return simu_name


def lcName(conf):
    """
    get the name of the lc file

    Parameters
    ----------
    conf : dict
        simu params.

    Returns
    -------
    lc_name : astropy table
        light curve points

    """

    lc_name = '{}/LC_{}_1.hdf5'.format(
        conf['OutputSimu']['directory'], conf['ProductionIDSimu'])

    return lc_name


def conftest(data, config, fname,
             cols_simpars=['sn_model', 'sn_version', 'daymax',
                           'z', 'ebvofMW', 'x0', 'x1', 'color'],
             cols_lc=['snr_m5', 'flux', 'mag',
                      'exptime', 'magerr', 'band', 'phase']):
    """
    Function to perform unit tests

    Parameters
    ----------
    data : array
        observations used in simulations.
    config : dict
        simu parameters.
    fname : str
        yaml output file name.
    cols_simpars : list(str), optional
        list of columns for comparison of simu params.
        The default is ['sn_model', 'sn_version', 'daymax',
                        'z', 'ebvofMW', 'x0', 'x1', 'color'].
    cols_lc : list(str), optional
        list of columns for LC points comparison.
        The default is ['snr_m5', 'flux', 'mag',
                        'exptime', 'magerr', 'band', 'phase'].

    Returns
    -------
    None.

    """

    dump(fname, config)
    name_config = fname.split('/')[-1].split('.')[0]

    #####################
    #  Simu parameters  #
    #####################

    # load simu data
    simulation(data, fname)
    simu_name = simuName(config)
    simu_data = gimeSimu(simu_name)
    print('kkk', simu_data)
    ref_simul = 'data_tests/ref_simu_{}.hdf5'.format(name_config)

    if not os.path.exists(ref_simul):
        simu_data.write(ref_simul, 'simu_parameters',
                        append=True, compression=True)

    refSimu = gimerefSimu(fname)

    # check simu parameters here
    @ pytest.mark.parametrize("simu_data,expected", refSimu)
    def testsimu_data(simu_data, expected):
        print('ccol', expected.columns)
        print(simu_data['n_lc_points'], expected['n_lc_points'])
        print(type(simu_data), type(expected))
        for key in cols_simpars:
            if key not in ['index_hdf5', 'fieldname', 'snr_fluxsec_meth',
                           'sn_type', 'sn_model', 'sn_version', 'SNID']:
                assert(np.isclose(
                    expected[key].tolist(), simu_data[key].tolist()).all())
            else:
                assert((expected[key] == simu_data[key]).all())
    testsimu_data(simu_data, refSimu)

    ####################
    #    LC data       #
    ####################

    lc_name = lcName(config)

    for simu in simu_data:
        lc = Table.read(lc_name, path='lc_{}'.format(simu['index_hdf5']))
        idx = lc['snr_m5'] >= 5.
        lc = lc[idx][: 20]
        break

    ref_lc = 'data_tests/ref_lc_{}.hdf5'.format(name_config)
    if not os.path.exists(ref_lc):
        lc.write(ref_lc, 'lc_points', append=True, compression=True)

    # load lc reference points
    tab_lc_ref = Table.read(ref_lc, path='lc_points')

    if 'index' in tab_lc_ref.columns:
        tab_lc_ref.remove_column('index')

    # check lc data here

    @ pytest.mark.parametrize("lc_data,expected", tab_lc_ref)
    def testsimu_lc(lc_data, expected):

        for key in cols_lc:
            if key not in ['band', 'filter_cosmo', 'zpsys']:
                assert(np.isclose(
                    expected[key].tolist(), lc_data[key].tolist()).all())
            else:
                assert(set(expected[key].tolist())
                       == set(lc_data[key].tolist()))
    testsimu_lc(lc, tab_lc_ref)


class TestSNsimulation(unittest.TestCase):
    """
    class to perform unit tests.
    """

    def testSimuSNCosmo(self):
        """
        Method to test simulated data with sncosmo

        Returns
        -------
        None.

        """

        # fake data
        data = fake_data()

        # test Ia simulation
        # test Ia - no error model
        # get configuration file
        config = ConfigSimulation(
            'Ia', 'salt3', '../sn_simu_input/config_simulation.txt').conf_dict
        config['ProductionIDSimu'] = 'prod_Ia_sncosmo_errormodel_0'
        config['Simulator']['errorModel'] = 0
        config['SN']['z']['type'] = 'uniform'
        config['SN']['z']['step'] = 0.1
        config['SN']['z']['min'] = 0.1
        config['SN']['z']['max'] = 1.0
        config['SN']['NSNabsolute'] = 1
        config['SN']['ebvofMW'] = 0.0
        config['ReferenceFiles']['GammaFile'] = 'gamma_DDF.hdf5'
        """
        config['SN']['x1']['type'] = 'random'
        config['SN']['color']['type'] = 'random'
        config['Display']['LC']['display'] = 1
        """
        fname = 'config2.yaml'
        conftest(data, config, fname)

        print('running with error model')
        # test Ia - with error model
        config['Simulator']['errorModel'] = 1
        config['ProductionIDSimu'] = 'prod_Ia_sncosmo_errormodel_1'
        fname = 'config1.yaml'
        conftest(data, config, fname)

        # test non Ia - error_model=0
        config['ProductionIDSimu'] = 'prod_Ib_sncosmo_errormodel_0'
        error_model = 0
        sn_type = 'SN_Ib'
        sn_model = 'nugent-sn2p'
        sn_model_version = '1.2'
        config['Simulator']['model'] = sn_model
        config['Simulator']['version'] = sn_model_version
        config['SN']['type'] = sn_type
        config['SN']['z']['type'] = 'uniform'
        config['SN']['z']['step'] = 0.1
        config['SN']['z']['min'] = 0.01
        config['SN']['z']['max'] = 1.0
        config['Simulator']['errorModel'] = error_model

        # test Ib
        fname = 'config3.yaml'
        conftest(data, config, fname,
                 cols_simpars=['sn_model', 'sn_version', 'daymax',
                               'z', 'ebvofMW'])

    def testSimuSNFast(self):
        """
        Method to test SN fast simu output

        Returns
        -------
        None.

        """

        # fake data
        data = fake_data()
        print('data', data)

        config = ConfigSimulation(
            'Ia', 'salt2', '../sn_simu_input/config_simulation.txt').conf_dict
        config['ProductionIDSimu'] = 'prod_Ia_snfast_errormodel_0'
        config['Simulator']['errorModel'] = 0
        config['SN']['z']['type'] = 'uniform'
        config['SN']['z']['step'] = 0.1
        config['SN']['z']['min'] = 0.1
        config['SN']['z']['max'] = 1.0
        config['SN']['ebvofMW'] = 0.0
        config['SN']['NSNabsolute'] = 1

        config['Simulator']['name'] = 'sn_simulator.sn_fast'
        config['ReferenceFiles']['GammaFile'] = 'gamma_DDF.hdf5'

        print('ccc', config)
        fname = 'config4.yaml'
        conftest(data, config, fname, cols_simpars=['season', 'daymax',
                                                    'z', 'ebvofMW'],
                 cols_lc=['snr_m5', 'flux', 'mag',
                          'magerr', 'band', 'phase'])


if __name__ == "__main__":
    unittest.main(verbosity=5)
