from builtins import zip
import numpy as np
import unittest
import lsst.utils.tests
import os
from numpy.testing import assert_almost_equal, assert_equal
# import pandas as pd
import h5py
from astropy.table import Table, vstack
# import glob
from sn_simulation.sn_simclass import SN_Simulation

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


def getconfig(prodid, x1colorType,
              x1Type, x1min, x1max, x1step,
              colorType, colormin, colormax, colorstep,
              zType, zmin, zmax, zstep,
              daymaxType, daymaxstep, diffflux,
              fulldbName, fieldType, fcoadd, seasval,
              simulator='sn_cosmo', nside=64, nproc=1, outputDir='.'):

    config = {}
    config['ProductionID'] = prodid

    # -------------- Supernova parameters ----------------------------------------
    config['SN parameters'] = {}
    config['SN parameters']['Id'] = 100                   # Id of the first SN
    # stretch and color
    config['SN parameters']['x1'] = {}
    config['SN parameters']['x1']['type'] = x1Type  # unique, uniform or random
    config['SN parameters']['x1']['min'] = x1min
    config['SN parameters']['x1']['max'] = x1max
    config['SN parameters']['color'] = {}
    # unique, uniform or random
    config['SN parameters']['color']['type'] = colorType
    config['SN parameters']['color']['min'] = colormin
    config['SN parameters']['color']['max'] = colormax

    config['SN parameters']['x1_color'] = {}
    config['SN parameters']['x1_color']['type'] = x1colorType  # random or fixed
    config['SN parameters']['x1_color']['min'] = [x1min, colormin]
    config['SN parameters']['x1_color']['max'] = [0.2, 0.2]
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
    config['SN parameters']['min_rf_phase'] = - \
        20.        # obs min phase (rest frame)
    # obs max phase (rest frame)
    config['SN parameters']['max_rf_phase'] = 60.
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
    config['Simulator']['Reference File'] = '/sps/lsst/data/dev/pgris/Templates_final_new/LC_{}_{}_vstack.hdf5'.format(
        x1min, colormin)
    config['SN parameters']['Gamma File'] = 'reference_files/gamma.hdf5'

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
             'visitTime', 'season', 'seeingFwhmEff', 'seeingFwhmGeom',
             'pixRA', 'pixDec', 'RA', 'Dec']
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

    def testSimuObject(self):
        # set simulation parameters
        prodid = 'Fake'
        x1colorType = 'unique'
        x1Type = 'unique'
        x1min = -2.0
        x1max = 2.0
        x1step = 0.1
        colorType = 'unique'
        colormin = 0.2
        colormax = 0.3
        colorstep = 0.02
        zType = 'unique'
        zmin = 0.1
        zmax = 1.0
        zstep = 0.1
        daymaxtype = 'unique'
        daymaxstep = 1.
        difflux = 0
        fulldbName = 'data_from_fake'
        fieldType = 'Fake'
        fcoadd = 1
        seasval = [1]

        # get the config file from these
        conf = getconfig(prodid, x1colorType,
                         x1Type, x1min, x1max, x1step,
                         colorType, colormin, colormax, colorstep,
                         zType, zmin, zmax, zstep,
                         daymaxtype, daymaxstep, difflux,
                         fulldbName, fieldType, fcoadd, seasval)

        print(conf)

        # SN_Simulation instance
        getRefDir('SALT2_Files')
        getRefDir('reference_files')

        absMag = conf['SN parameters']['absmag']
        x0normFile = 'reference_files/X0_norm_{}.npy'.format(absMag)

        x0_norm = np.load(x0normFile)

        area = 9.6  # survey area (deg2)

        simu = SN_Simulation(cosmo_par=conf['Cosmology'],
                             tel_par=conf['Instrument'],
                             sn_parameters=conf['SN parameters'],
                             save_status=conf['Output']['save'],
                             outdir=conf['Output']['directory'],
                             prodid=conf['ProductionID'],
                             simu_config=conf['Simulator'],
                             x0_norm=x0_norm,
                             display_lc=conf['Display_LC']['display'],
                             time_display=conf['Display_LC']['time'],
                             area=area,
                             mjdCol='observationStartMJD',
                             filterCol='filter',
                             nexpCol='numExposures',
                             exptimeCol='visitExposureTime',
                             x1colorDir=conf['SN parameters']['x1_color']['dirFile'],
                             salt2Dir=conf['SN parameters']['salt2Dir'],
                             nproc=conf['Multiprocessing']['nproc'])

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

        print(len(data))
        # now simulate LC on this data

        simu(data, conf['Observations']['fieldtype'], 100, -1, None)

        # save metadata

        simu.Finish()

        # check what we have inside the data

        simu_name = '{}/Simu_{}.hdf5'.format(
            conf['Output']['directory'], conf['ProductionID'])
        lc_name = '{}/LC_{}.hdf5'.format(
            conf['Output']['directory'], conf['ProductionID'])

        f = h5py.File(simu_name, 'r')
        print(f.keys())
        # reading the simu file
        for i, key in enumerate(f.keys()):
            simul = Table.read(simu_name, path=key)

        print(simul[['x1', 'color', 'z', 'daymax']])

        # now grab LC

        for simu in simul:
            lc = Table.read(lc_name, path='lc_{}'.format(simu['id_hdf5']))
            print(lc)


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main(verbosity=5)
