import numpy as np
from lsst.sims.maf.metrics import BaseMetric
from sn_simulation.sn_simclass import SN_Simulation
from sn_stackers.coadd_stacker import CoaddStacker
import healpy as hp
import numpy.lib.recfunctions as rf

class SNMetric(BaseMetric):
    """LC simulations in the "MAF metric" framework

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
    coadd: bool, opt
     coaddition of obs (per band and per night) if set to True (default: True)
    config: dict
     configuration dict for simulation (SN parameters, cosmology, telescope, ...)
     ex: {'ProductionID': 'DD_baseline2018a_Cosmo', 'SN parameters': {'Id': 100, 'x1_color': {'type': 'fixed', 'min': [-2.0, 0.2], 'max': [0.2, 0.2], 'rate': 'JLA'}, 'z': {'type': 'uniform', 'min': 0.01, 'max': 0.9, 'step': 0.05, 'rate': 'Perrett'}, 'daymax': {'type': 'unique', 'step': 1}, 'min_rf_phase': -20.0, 'max_rf_phase': 60.0, 'absmag': -19.0906, 'band': 'bessellB', 'magsys': 'vega', 'differential_flux': False}, 'Cosmology': {'Model': 'w0waCDM', 'Omega_m': 0.3, 'Omega_l': 0.7, 'H0': 72.0, 'w0': -1.0, 'wa': 0.0}, 'Instrument': {'name': 'LSST', 'throughput_dir': 'LSST_THROUGHPUTS_BASELINE', 'atmos_dir': 'THROUGHPUTS_DIR', 'airmass': 1.2, 'atmos': True, 'aerosol': False}, 'Observations': {'filename': '/home/philippe/LSST/DB_Files/kraken_2026.db', 'fieldtype': 'DD', 'coadd': True, 'season': 1}, 'Simulator': {'name': 'sn_simulator.sn_cosmo', 'model': 'salt2-extended', 'version': 1.0, 'Reference File': 'LC_Test_today.hdf5'}, 'Host Parameters': 'None', 'Display_LC': {'display': True, 'time': 1}, 'Output': {'directory': 'Output_Simu', 'save': True}, 'Multiprocessing': {'nproc': 1}, 'Metric': 'sn_mafsim.sn_maf_simulation', 'Pixelisation': {'nside': 64}}
    x0_norm: array of float
     grid ox (x1,color,x0) values
    reference_lc: class
     reference lc for the fast simulation

    """

    def __init__(self, metricName='SNMetric',
                 mjdCol='observationStartMJD', RaCol='fieldRA', DecCol='fieldDec',
                 filterCol='filter', m5Col='fiveSigmaDepth', exptimeCol='visitExposureTime',
                 nightCol='night', obsidCol='observationId', nexpCol='numExposures', vistimeCol='visitTime', seeingEffCol='seeingFwhmEff', seeingGeomCol='seeingFwhmGeom', coadd=True,
                 uniqueBlocks=False, config=None, x0_norm=None,reference_lc=None,**kwargs):

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
        self.reference_lc = reference_lc

        cols = [self.RaCol, self.DecCol,self.nightCol, self.m5Col, self.filterCol, self.mjdCol, self.obsidCol,
                self.nexpCol, self.vistimeCol, self.exptimeCol, self.seasonCol, self.seeingEffCol, self.seeingGeomCol,self.nightCol]
        self.stacker = None

       
        if coadd:
            #cols += ['sn_coadd']
            self.stacker = CoaddStacker(mjdCol=self.mjdCol,RaCol=self.RaCol, DecCol=self.DecCol, m5Col=self.m5Col, nightCol=self.nightCol, filterCol=self.filterCol, numExposuresCol=self.nexpCol, visitTimeCol=self.vistimeCol,visitExposureTimeCol='visitExposureTime')
        super(SNMetric, self).__init__(
            col=cols, metricName=metricName, **kwargs)
        """
        super().__init__(
            col=cols, metricName=metricName, **kwargs)
        """
        #self.filterNames = np.array(['u', 'g', 'r', 'i', 'z', 'y'])
        self.filterNames = 'ugrizy'
        self.config = config
        self.nside = config['Pixelisation']['nside']

        # load cosmology
        cosmo_par = config['Cosmology']
        # load telescope
        tel_par = config['Instrument']

        # this is for output

        save_status = config['Output']['save']
        self.save_status = save_status
        outdir = config['Output']['directory']
        prodid = config['ProductionID']
        # sn parameters
        sn_parameters = config['SN parameters']

        simu_config = config['Simulator']
        display_lc = config['Display_LC']['display']
        display_time = config['Display_LC']['time']
        self.field_type = config['Observations']['fieldtype']
        self.season = config['Observations']['season']
        #area = 9.6  # survey_area in sqdeg - 9.6 by default for DD
        #if self.field_type == 'WFD':
        # in that case the survey area is the healpix area
        area = hp.nside2pixarea(self.nside, degrees=True)

        self.type = 'simulation'

        # instantiate simulator here

        self.simu = SN_Simulation(cosmo_par, tel_par, sn_parameters,
                                  save_status, outdir, prodid,
                                  simu_config, x0_norm,display_lc, display_time, area,
                                  mjdCol=self.mjdCol, RaCol=self.RaCol,
                                  DecCol=self.DecCol,
                                  filterCol=self.filterCol, exptimeCol=self.exptimeCol,
                                  m5Col=self.m5Col, seasonCol=self.seasonCol,
                                  seeingEffCol=self.seeingEffCol, seeingGeomCol=self.seeingGeomCol,
                                  x1colorDir=config['SN parameters']['x1_color']['dirFile'],
                                  salt2Dir=config['SN parameters']['salt2Dir'],
                                  nproc=config['Multiprocessing']['nproc'])

    def run(self, dataSlice, slicePoint=None):
        """ Run Simulation on dataSlice

        Parameters
        --------------
        dataSlice: array
          array of observations

        Returns
        ---------
        None

        """

        """
        goodFilters = np.in1d(dataSlice[self.filterCol], self.filterNames)
        dataSlice = dataSlice[goodFilters]
        if dataSlice.size == 0:
            return (self.badval, self.badval, self.badval)
        """
        dataSlice.sort(order=self.mjdCol)
        
        #print(dataSlice.dtype)
        #print(dataSlice[[self.mjdCol,self.filterCol,self.exptimeCol,self.nexpCol]])
        if self.stacker is not None:
            dataSlice = self.stacker._run(dataSlice)
            print('stacked')
        #print(dataSlice[[self.mjdCol,self.filterCol,self.exptimeCol,self.nexpCol]])

        #print(test)
        #print('stacked')
        # if the fields healpixID, pixRa, pixDec are not in dataSlice 
        # then estimate them and add them to dataSlice
        datacp = np.copy(dataSlice)
        
        if 'healpixID' not in datacp.dtype.names:
            Ra = np.mean(datacp[self.RaCol])
            Dec = np.mean(datacp[self.DecCol])
            
            table = hp.ang2vec([Ra], [Dec], lonlat=True)
            
            healpixs = hp.vec2pix(self.nside, table[:, 0], table[:, 1], table[:, 2], nest=True)
            coord = hp.pix2ang(self.nside, healpixs, nest=True, lonlat=True)
            
            healpixId, pixRa, pixDec = healpixs[0], coord[0][0],coord[1][0]
            
            datacp = rf.append_fields(datacp,'healpixId',[healpixId]*len(datacp))
            datacp = rf.append_fields(datacp,'pixRa',[pixRa]*len(datacp))
            datacp = rf.append_fields(datacp,'pixDec',[pixDec]*len(datacp))
        #print('alors',datacp[['healpixId','pixRa','pixDec']])
        
        # Run simulation
        
        self.simu(datacp, self.field_type, 100, self.season,self.reference_lc)

        return None
