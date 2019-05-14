import numpy as np
from lsst.sims.maf.metrics import BaseMetric
from sn_simulation.sn_simulation import SN_Simulation
from sn_stackers.coadd_stacker import CoaddStacker
import healpy as hp


class SNMetric(BaseMetric):
    """
    Measure how many time series meet a given time and filter distribution requirement.
    """

    def __init__(self, metricName='SNMetric',
                 mjdCol='observationStartMJD', RaCol='fieldRA', DecCol='fieldDec',
                 filterCol='filter', m5Col='fiveSigmaDepth', exptimeCol='visitExposureTime',
                 nightCol='night', obsidCol='observationId', nexpCol='numExposures', vistimeCol='visitTime', seeingEffCol='seeingFwhmEff', seeingGeomCol='seeingFwhmGeom', coadd=True,
                 uniqueBlocks=False, config=None, **kwargs):

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

        cols = [self.nightCol, self.m5Col, self.filterCol, self.mjdCol, self.obsidCol,
                self.nexpCol, self.vistimeCol, self.exptimeCol, self.seasonCol, self.seeingEffCol, self.seeingGeomCol]
        if coadd:
            cols += ['coadd']
        super(SNMetric, self).__init__(
            col=cols, metricName=metricName, **kwargs)

        self.filterNames = np.array(['u', 'g', 'r', 'i', 'z', 'y'])
        self.config = config

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
        area = 9.6  # survey_area in sqdeg - 9.6 by default for DD
        if self.field_type == 'WFD':
            # in that case the survey area is the healpix area
            area = hp.nside2pixarea(
                config['Pixelisation']['nside'], degrees=True)
        self.simu = SN_Simulation(cosmo_par, tel_par, sn_parameters,
                                  save_status, outdir, prodid,
                                  simu_config, display_lc, display_time, area,
                                  mjdCol=self.mjdCol, RaCol=self.RaCol,
                                  DecCol=self.DecCol,
                                  filterCol=self.filterCol, exptimeCol=self.exptimeCol,
                                  m5Col=self.m5Col, seasonCol=self.seasonCol,
                                  seeingEffCol=self.seeingEffCol, seeingGeomCol=self.seeingGeomCol,
                                  nproc=config['Multiprocessing']['nproc'])

    def run(self, dataSlice, slicePoint=None):
        # Cut down to only include filters in correct wave range.

        goodFilters = np.in1d(dataSlice['filter'], self.filterNames)
        dataSlice = dataSlice[goodFilters]
        if dataSlice.size == 0:
            return (self.badval, self.badval, self.badval)
        dataSlice.sort(order=self.mjdCol)
        print('dataslice', np.unique(
            dataSlice[['fieldRA', 'fieldDec', 'season']]), dataSlice.dtype)
        time = dataSlice[self.mjdCol]-dataSlice[self.mjdCol].min()

        self.simu(dataSlice, self.field_type, 100, self.season)

        return None
