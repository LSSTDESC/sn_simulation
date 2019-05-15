from sn_tools.sn_telescope import Telescope
import numpy as np
import astropy.units as u
from astropy.table import Table
from collections import OrderedDict as odict


class SN_Object:
    def __init__(self, name, sn_parameters, gen_parameters, cosmology,
                 Telescope, snid, area,
                 mjdCol='mjd', RaCol='pixRa', DecCol='pixDec',
                 filterCol='band', exptimeCol='exptime', nexpCol='numExposures',
                 m5Col='fiveSigmaDepth', seasonCol='season',
                 seeingEffCol='seeingFwhmEff', seeingGeomCol='seeingFwhmGeom'):
        """ class SN object
        handles sn name, parameters,
        cosmology, snid, telescope...
        necessary parameters for simulation
        SN classes inherit from SN_Object

        Parameters
        --------------
        name: str
          production name?
        sn_parameters: dict
         supernovae parameters
        gen_parameters: dict
         simulation parameters
        cosmology: dict
         cosmological parameters used for simulation
        telescope: dict
         telescope parameters
        snid: int
         supernova identifier
        area: float
          survey area (usually in deg2)
         mjdCol: str, opt
           mjd col name in observations (default: 'mjd')
        RaCol: str, opt
          RA col name in observations (default: 'pixRa')
        DecCol:str, opt
          Dec col name in observations (default: 'pixDec')
        filterCol: str, opt
          filter col name in observations (default: band')
        exptimeCol: str, opt
         exposure time  col name in observations (default: 'exptime')
        nexpCol: str, opt
         number of exposures col name in observations (default: 'numExposures')
        m5Col: str, opt
          5-sigma depth col name in observations (default: 'fiveSigmaDepth')
        seasonCol: str, opt
         season col name in observations (default: 'season')
        seeingEffCol: str, opt
         seeing eff col name in observations (default: 'seeingFwhmEff')
        seeingGeomCol: str, opt
        seeing geom  col name in observations (default: 'seeingFwhmGeom')

    """
        self._name = name
        self._sn_parameters = sn_parameters
        self._gen_parameters = gen_parameters
        self._cosmology = cosmology
        self._telescope = Telescope
        self._SNID = snid

        self.mjdCol = mjdCol
        self.RaCol = RaCol
        self.DecCol = DecCol
        self.filterCol = filterCol
        self.exptimeCol = exptimeCol
        self.nexpCol = nexpCol
        self.m5Col = m5Col
        self.seasonCol = seasonCol
        self.seeingEffCol = seeingEffCol
        self.seeingGeomCol = seeingGeomCol
        self.area = area

    @property
    def name(self):
        return self._name

    @property
    def sn_parameters(self):
        """SN parameters
        """
        return self._sn_parameters

    @property
    def gen_parameters(self):
        """ Simulation parameters
        """
        return self._gen_parameters

    @property
    def cosmology(self):
        """ Cosmology
        """
        return self._cosmology

    @property
    def telescope(self):
        """ Telescope
        """
        return self._telescope

    @property
    def SNID(self):
        """ SN identifier
        """
        return self._SNID

    def cutoff(self, obs, T0, z, min_rf_phase, max_rf_phase):
        """ select observations depending on phases

        Parameters
        -------------
        obs: array
          array of observations
        T0: float
          daymax of the supernova
        z: float
          redshift
        min_rf_phase: float
          min phase rest-frame
        max_rf_phase: float
         max phase rest-frame

        Returns
        ----------
        array of obs passing the selection
        """

        blue_cutoff = 300.
        red_cutoff = 800.

        mean_restframe_wavelength = np.asarray(
            [self.telescope.mean_wavelength[obser[self.filterCol][-1]] /
             (1. + z) for obser in obs])

        p = (obs[self.mjdCol]-T0)/(1.+z)

        idx = (p >= min_rf_phase) & (p <= max_rf_phase)
        idx &= (mean_restframe_wavelength > blue_cutoff)
        idx &= (mean_restframe_wavelength < red_cutoff)
        return obs[idx]

    def plotLC(self, table, time_display):
        """ Light curve plot using sncosmo methods

        Parameters
        ---------------
        table: astropy table
         table with LS informations (flux, ...)
       time_display: float
         duration of the window display
        """

        import pylab as plt
        import sncosmo
        prefix = 'LSST::'
        """
        _photdata_aliases = odict([
            ('time', set(['time', 'date', 'jd', 'mjd', 'mjdobs', 'mjd_obs'])),
            ('band', set(['band', 'bandpass', 'filter', 'flt'])),
            ('flux', set(['flux', 'f'])),
            ('fluxerr', set(
                ['fluxerr', 'fe', 'fluxerror', 'flux_error', 'flux_err'])),
            ('zp', set(['zp', 'zpt', 'zeropoint', 'zero_point'])),
            ('zpsys', set(['zpsys', 'zpmagsys', 'magsys']))
        ])
        """
        for band in 'grizy':
            name_filter = prefix+band
            if self.telescope.airmass > 0:
                bandpass = sncosmo.Bandpass(
                    self.telescope.atmosphere[band].wavelen,
                    self.telescope.atmosphere[band].sb,
                    name=name_filter,
                    wave_unit=u.nm)
            else:
                bandpass = sncosmo.Bandpass(
                    self.telescope.system[band].wavelen,
                    self.telescope.system[band].sb,
                    name=name_filter,
                    wave_unit=u.nm)
            # print('registering',name_filter)
            sncosmo.registry.register(bandpass, force=True)

        z = table.meta['z']
        x1 = table.meta['x1']
        color = table.meta['color']
        daymax = table.meta['daymax']

        model = sncosmo.Model('salt2')
        model.set(z=z,
                  c=color,
                  t0=daymax,
                  # x0=self.X0,
                  x1=x1)
        """
        print('tests',isinstance(table, np.ndarray),isinstance(table,Table),isinstance(table,dict))
        array_tab = np.asarray(table)
        print(array_tab.dtype)
        colnames = array_tab.dtype.names
        # Create mapping from lowercased column names to originals
        lower_to_orig = dict([(colname.lower(), colname) for colname in colnames])
        
        # Set of lowercase column names
        lower_colnames = set(lower_to_orig.keys())
        orig_colnames_to_use = []
        for aliases in _photdata_aliases.values():
            i = lower_colnames & aliases
            if len(i) != 1:
                raise ValueError('Data must include exactly one column from {0} '
                                 '(case independent)'.format(', '.join(aliases)))
            orig_colnames_to_use.append(lower_to_orig[i.pop()])

        
        new_data = table[orig_colnames_to_use].copy()
        print('bbbb',orig_colnames_to_use,_photdata_aliases.keys(),new_data.dtype.names)
        new_data.dtype.names = _photdata_aliases.keys()
        """
        sncosmo.plot_lc(data=table, model=model)

        plt.draw()
        plt.pause(time_display)
        plt.close()
