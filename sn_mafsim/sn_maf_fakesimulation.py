import numpy as np
from lsst.sims.maf.metrics import BaseMetric
from sn_simulation.sn_simclass import SN_Simulation
from sn_stackers.coadd_stacker import CoaddStacker
import healpy as hp


class SNSimulation:
    """
    LC simulations in the "MAF metric" framework
    on fake observations

    Parameters
    ---------------
    mjdCol: str, opt
     mjd col name in observations (default: 'observationStartMJD')
    RaCol: str, opt
      RA col name in observations (default: 'fieldRA')
    DecCol:str, opt
     Dec col name in observations (default: 'fieldDec')
    filterCol: str, opt
     filter col name in observations (default: filter') 
    m5Col: str, opt
     5-sigma depth col name in observations (default: 'fiveSigmaDepth')
    exptimeCol: str, opt
     exposure time  col name in observations (default: 'visitExposureTime)
    nightCol: str, opt
     night col name in observations (default: 'night')
    obsidCol: str, opt
     observation id col name in observations (default: 'observationId')
    nexpCol: str, opt
     number of exposures col name in observations (default: 'numExposures')
    visitimeCol: str, opt
     visit time col name in observations (default: 'visiTime')
    seeingEffCol: str, opt
     seeing eff col name in observations (default: 'seeingFwhmEff')
    seeingGeomCol: str, opt
     seeing geom  col name in observations (default: 'seeingFwhmGeom')
    config: dict
     configuration dict for simulation (SN parameters, cosmology, telescope, ...)
    ex: {'ProductionID': 'Fake_Observation_0.0_0.0', 'SN parameters': {'Id': 100, 'x1_color': {'type': 'fixed', 'min': [0.0, 0.0], 'max': [2.0, 0.2], 'rate': 'JLA'}, 'z': {'type': 'uniform', 'min': 0.01, 'max': 0.12, 'step': 0.01, 'rate': 'Perrett'}, 'daymax': {'type': 'unique', 'step': 1}, 'min_rf_phase': -20.0, 'max_rf_phase': 60.0, 'absmag': -19.0906, 'band': 'bessellB', 'magsys': 'vega', 'differential_flux': True}, 'Cosmology': {'Model': 'w0waCDM', 'Omega_m': 0.3, 'Omega_l': 0.7, 'H0': 72.0, 'w0': -1.0, 'wa': 0.0}, 'Instrument': {'name': 'LSST', 'throughput_dir': 'LSST_THROUGHPUTS_BASELINE', 'atmos_dir': 'THROUGHPUTS_DIR', 'airmass': 1.1, 'atmos': True, 'aerosol': False}, 'Observations': {'filename': 'None', 'fieldtype': 'None', 'coadd': 'None', 'season': 1}, 'Simulator': {'name': 'sn_simulator.sn_cosmo', 'model': 'salt2-extended', 'version': 1.0}, 'Host Parameters': 'None', 'Display_LC': {'display': True, 'time': 5}, 'Output': {'directory': 'Output_Simu', 'save': True}, 'Multiprocessing': {'nproc': 1}, 'Metric': 'sn_mafsim.sn_maf_fakesimulation', 'Param_file': 'input/Fake_cadence.yaml'}

    """

    def __init__(self, mjdCol='observationStartMJD', RaCol='fieldRA', DecCol='fieldDec',
                 filterCol='filter', m5Col='fiveSigmaDepth', exptimeCol='visitExposureTime',
                 nightCol='night', obsidCol='observationId', nexpCol='numExposures', vistimeCol='visitTime', seeingEffCol='seeingFwhmEff', seeingGeomCol='seeingFwhmGeom', config=None):

        self.mjdCol = mjdCol
        self.m5Col = m5Col
        self.filterCol = filterCol
        self.RaCol = RaCol
        self.DecCol = DecCol
        self.exptimeCol = exptimeCol
        self.seasonCol = 'season'
        self.nightCol = nightCol
        self.obsidCol = obsidCol
        self.nexpCol = nexpCol
        self.vistimeCol = vistimeCol
        self.seeingEffCol = seeingEffCol
        self.seeingGeomCol = seeingGeomCol

        self.config = config

        # load cosmology
        cosmo_par = config['Cosmology']
        # load telescope
        tel_par = config['Instrument']

        # this is for output

        save_status = config['Output']['save']
        outdir = config['Output']['directory']
        prodid = config['ProductionID']
        # sn parameters
        sn_parameters = config['SN parameters']

        simu_config = config['Simulator']
        display_lc = config['Display_LC']['display']
        display_time = config['Display_LC']['time']
        self.field_type = config['Observations']['fieldtype']
        self.season = config['Observations']['season']
        area = 9.6  # survey_area in sqdeg - 9.6 by default for DD

        # instantiate the simulator here

        self.simu = SN_Simulation(cosmo_par, tel_par, sn_parameters,
                                  save_status, outdir, prodid,
                                  simu_config, display_lc, display_time, area,
                                  mjdCol=self.mjdCol, RaCol=self.RaCol,
                                  DecCol=self.DecCol,
                                  filterCol=self.filterCol, exptimeCol=self.exptimeCol,
                                  m5Col=self.m5Col, seasonCol=self.seasonCol,
                                  seeingEffCol=self.seeingEffCol, seeingGeomCol=self.seeingGeomCol,
                                  nproc=config['Multiprocessing']['nproc'])

    def run(self, dataSlice):
        """ Run Simulation on dataSlice

        Parameters
        --------------
        dataSlice: array
          array of observations

        Returns
        ---------
        None

        """
        dataSlice.sort(order=self.mjdCol)
        # print('dataslice', np.unique(
        #   dataSlice[['fieldRA', 'fieldDec', 'season']]), dataSlice.dtype)
        time = dataSlice[self.mjdCol]-dataSlice[self.mjdCol].min()
        #print(dataSlice, time)
        self.simu(dataSlice, self.field_type, 100, self.season)
        return None
