import numpy as np
from scipy import interpolate
from astropy.table import Table, Column, vstack, hstack
import glob
import h5py
import pylab as plt
from scipy.spatial import distance
import time
import multiprocessing
from optparse import OptionParser
import os
from scipy.interpolate import griddata,interp2d,RegularGridInterpolator
import numpy.lib.recfunctions as rf
import scipy.linalg.lapack as lapack
from sn_simulation.sn_object import SN_Object
import time
import pandas as pd

class SN(SN_Object):
    """ SN class - inherits from SN_Object
          Input parameters (as given in the input yaml file):
          - SN parameters (x1, color, daymax, z, ...)
          - simulation parameters

         Output:
         - astropy table with the simulated light curve:
               - columns : band, flux, fluxerr, snr_m5,flux_e,zp,zpsys,time
               - metadata : SNID,Ra,Dec,DayMax,X1,Color,z

    """

    def __init__(self, param, simu_param, reference_lc):
        super().__init__(param.name, param.sn_parameters, param.gen_parameters,
                         param.cosmology, param.telescope, param.SNID, param.area,param.x0_grid,
                         mjdCol=param.mjdCol, RaCol=param.RaCol, DecCol=param.DecCol,
                         filterCol=param.filterCol, exptimeCol=param.exptimeCol,
                         m5Col=param.m5Col, seasonCol=param.seasonCol)

        self.x1 = self.sn_parameters['x1']
        self.color = self.sn_parameters['color']
        zvals = [self.sn_parameters['z']]
        
        #Loading reference file
        self.reference_lc = reference_lc
       
        # This cutoffs are used to select observations:
        # phase = (mjd - DayMax)/(1.+z)
        # selection: min_rf_phase < phase < max_rf_phase
        # and        blue_cutoff < mean_rest_frame < red_cutoff
        # where mean_rest_frame = telescope.mean_wavelength/(1.+z)
        self.blue_cutoff = 300.
        self.red_cutoff = 800.

        # SN parameters for Fisher matrix estimation
        self.param_Fisher = ['x0', 'x1', 'color']
       
    def __call__(self, obs, index_hdf5, gen_par=None, display=False, time_display=0.):
        """ Simulation of the light curve
        We use multiprocessing (one band per process) to increase speed

        Parameters
        ---------
        obs: array
         array of observations
        index_hdf5: int
         index of the LC in the hdf5 file (to allow fast access)
        gen_par: array
         simulation parameters
        display: bool,opt
         to display LC as they are generated (default: False)
        time_display: float, opt
         time persistency of the displayed window (defalut: 0 sec)
        


        Returns
        ---------
        astropy table with:
        columns: band, flux, fluxerr, snr_m5,flux_e,zp,zpsys,time
        metadata : SNID,Ra,Dec,DayMax,X1,Color,z
        """

        #assert (len(np.unique(obs[self.RaCol])) == 1)
        #assert (len(np.unique(obs[self.DecCol])) == 1)
        #ra = np.asscalar(np.unique(obs[self.RaCol]))
        #dec = np.asscalar(np.unique(obs[self.DecCol]))
        ra = np.mean(obs[self.RaCol])
        dec = np.mean(obs[self.DecCol])

        area = self.area
        self.index_hdf5 = index_hdf5

        if len(obs) == 0:
            return None

        result_queue = multiprocessing.Queue()
        bands = 'grizy'

        tab_tot = pd.DataFrame()

        # multiprocessing here: one process (processBand) per band
        jproc = -1
        time_ref = time.time()
        for j, band in enumerate(bands):
            idx = obs[self.filterCol] == band
            if len(obs[idx]) > 0:
                jproc+=1
                p = multiprocessing.Process(name='Subprocess-'+str(
                    j), target=self.processBand, args=(obs[idx], band, gen_par, jproc, result_queue))
                p.start()

        resultdict = {}
        for j in range(jproc+1):
            resultdict.update(result_queue.get())

        for p in multiprocessing.active_children():
            p.join()

        for j in range(jproc+1):
            tab_tot = tab_tot.append(resultdict[j], ignore_index=True)
           
        
        # Collect the results
        #tab_tot = [resultdict[j] for j in range(jproc+1) if resultdict[j] is not None]

        # There is a trick here
        # Usually one would just use vstack to get one astropy Table
        # But vstack seems to trigger problems with special h5py dype
        # namely h5py.special_dtype(vlen=str)))
        # so the idea is to convert the Tables to list,
        # to add the lists,
        # and then recreate an astropy table.

        """
        r = []
        if len(tab_tot) == 0:
            return ra, dec, None
        for val in tab_tot:
            valb = np.asarray(val)
            r += np.ndarray.tolist(valb)
            dtype = val.dtype
            names = val.colnames
        
        dtype = [dtype.fields[name][0] for name in names]
        tab_tot = Table(rows=r, names=names, dtype=dtype)
        """

        print('full end',time.time()-time_ref)
        return tab_tot

    def processBand(self, sel_obs, band, gen_par, j=-1, output_q=None):
        """ LC simulation of a set of obs corresponding to a band
        The idea is to use python broadcasting so as to estimate 
        all the requested values (flux, flux error, Fisher components, ...)
        in a single path (i.e no loop!)

        Parameters
        -----------
        sel_obs: array
         array of observations
        band: str
         band of observations
        gen_par: array
         simulation parameters
        j: int, opt
         index for multiprocessing (default: -1)
        output_q: multiprocessing.Queue(),opt
         queue for multiprocessing (default: None)
        

        Returns
        -------
        astropy table with fields corresponding to LC components


        """

        
        # method used for interpolation
        method = 'linear'
        interpType = 'griddata'
        interpType = 'regular'
        
        # if there are no observations in this filter: return None
        if len(sel_obs) == 0:
            if output_q is not None:
                output_q.put({j: None})
            else:
                return None

        # Get the fluxes (from griddata reference)

        # xi = MJD-T0
        xi = sel_obs[self.mjdCol]-gen_par['daymax'][:, np.newaxis]

        # yi = redshift simulated values
        yi = np.round(gen_par['z'],4) # requested to avoid interpolation problems near boundaries

        # p = phases of LC points = xi/(1.+z)
        p = xi/(1.+yi[:, np.newaxis])  
        yi_arr = np.ones_like(p)*yi[:, np.newaxis]

        time_ref = time.time()
        if interpType == 'griddata':
            # Get reference values: phase, z, flux, fluxerr
            x = self.reference_lc.lc_ref[band]['phase']
            y = self.reference_lc.lc_ref[band]['z']
            z = self.reference_lc.lc_ref[band]['flux']
            zb = self.reference_lc.lc_ref[band]['fluxerr']
        
    
            # flux interpolation
            fluxes_obs = griddata((x, y), z, (p, yi_arr),
                                  method=method, fill_value=0.)

            # flux error interpolation
            fluxes_obs_err = griddata(
                (x, y), zb, (p, yi_arr), method=method, fill_value=0.)
        
            # Fisher components estimation

            dFlux = {}

            # loop on Fisher parameters
            for val in self.param_Fisher:
                #get the reference components
                z_c = self.reference_lc.lc_ref[band]['d'+val]
                # get Fisher components from interpolation
                dFlux[val] = griddata((x, y), z_c, (p, yi_arr),
                                      method=method, fill_value=0.)

       

        if interpType == 'regular':
            
            """
            # remove LC points outside the restframe phase range
            min_rf_phase = gen_par['min_rf_phase'][:, np.newaxis]
            max_rf_phase = gen_par['max_rf_phase'][:, np.newaxis]
            flag = (p >= min_rf_phase) & (p <= max_rf_phase)
            
            time_ref = time.time()
            p_mask = np.ma.array(p, mask=~flag)
            yi_mask = np.ma.array(yi_arr, mask=~flag)
           
            pts = (p_mask[~p.mask],yi_mask[~p.mask])
            """
            pts = (p,yi_arr)
            fluxes_obs = self.reference_lc.flux[band](pts)
            fluxes_obs_err = self.reference_lc.fluxerr[band](pts)

            # Fisher components estimation

            dFlux = {}

            # loop on Fisher parameters
            for val in self.param_Fisher:
                dFlux[val] = self.reference_lc.param[band][val](pts)
            # get the reference components
            #z_c = self.reference_lc.lc_ref[band]['d'+val]
            # get Fisher components from interpolation
            #dFlux[val] = griddata((x, y), z_c, (p, yi_arr),
            #                      method=method, fill_value=0.)

        
        # replace crazy fluxes by dummy values
        fluxes_obs[fluxes_obs<=0.] = 1.e-8
        fluxes_obs_err[fluxes_obs_err<=0.] = 1.e-10
            

        # Fisher matrix components estimation
        # loop on SN parameters (x0,x1,color)
        # estimate: dF/dxi*dF/dxj/sigma_flux**2
        Derivative_for_Fisher = {}
        for ia, vala in enumerate(self.param_Fisher):
            for jb, valb in enumerate(self.param_Fisher):
                if jb >= ia:
                    Derivative_for_Fisher[vala+valb] = dFlux[vala] * dFlux[valb]

       
        # remove LC points outside the restframe phase range
        min_rf_phase = gen_par['min_rf_phase'][:, np.newaxis]
        max_rf_phase = gen_par['max_rf_phase'][:, np.newaxis]
        flag = (p >= min_rf_phase) & (p <= max_rf_phase)
        

        # remove LC points outside the (blue-red) range
        mean_restframe_wavelength = np.array(
            [self.telescope.mean_wavelength[band]]*len(sel_obs))
        mean_restframe_wavelength = np.tile(
            mean_restframe_wavelength, (len(gen_par), 1))/(1.+gen_par['z'][:, np.newaxis])
        flag &= (mean_restframe_wavelength > self.blue_cutoff) & (
            mean_restframe_wavelength < self.red_cutoff)

        flag_idx = np.argwhere(flag)

        # Correct fluxes_err (m5 in generation probably different from m5 obs)
        
        #gamma_obs = self.telescope.gamma(
        #    sel_obs[self.m5Col], [band]*len(sel_obs), sel_obs[self.exptimeCol])

        gamma_obs = self.reference_lc.gamma[band]((sel_obs[self.m5Col],sel_obs[self.exptimeCol]))


        mag_obs = -2.5*np.log10(fluxes_obs/3631.)
        
        m5 = np.asarray([self.reference_lc.m5_ref[band]]*len(sel_obs))
        gammaref = np.asarray([self.reference_lc.gamma_ref[band]]*len(sel_obs))
        m5_tile =  np.tile(m5, (len(p), 1))
        srand_ref = self.srand(
            np.tile(gammaref, (len(p), 1)), mag_obs, m5_tile)
        srand_obs = self.srand(np.tile(gamma_obs, (len(p), 1)), mag_obs, np.tile(
            sel_obs[self.m5Col], (len(p), 1)))
        correct_m5 = srand_ref/srand_obs
        fluxes_obs_err = fluxes_obs_err/correct_m5

        # now apply the flag to select LC points
        fluxes = np.ma.array(fluxes_obs, mask=~flag)
        fluxes_err = np.ma.array(fluxes_obs_err, mask=~flag)
        phases = np.ma.array(p, mask=~flag)
        snr_m5 = np.ma.array(fluxes_obs/fluxes_obs_err, mask=~flag)
        
        nvals = len(phases)
        
        obs_time = np.ma.array(
            np.tile(sel_obs[self.mjdCol], (nvals, 1)), mask=~flag)
        seasons = np.ma.array(
            np.tile(sel_obs[self.seasonCol], (nvals, 1)), mask=~flag)
        exp_time = np.ma.array(
            np.tile(sel_obs[self.exptimeCol], (nvals, 1)), mask=~flag)
        m5_obs = np.ma.array(
            np.tile(sel_obs[self.m5Col], (nvals, 1)), mask=~flag)
        healpixIds = np.ma.array(
            np.tile(sel_obs['healpixID'], (nvals, 1)), mask=~flag)
        
        pixRas = np.ma.array(
            np.tile(sel_obs['pixRa'], (nvals, 1)), mask=~flag)

        pixDecs = np.ma.array(
            np.tile(sel_obs['pixDec'], (nvals, 1)), mask=~flag)
        

        z_vals = gen_par['z'][flag_idx[:, 0]]
        daymax_vals = gen_par['daymax'][flag_idx[:, 0]]
        mag_obs = np.ma.array(mag_obs, mask=~flag)
        Fisher_Mat = {}
        for key, vals in Derivative_for_Fisher.items():
            Fisher_Mat[key] = np.ma.array(vals, mask=~flag)

        """
        # Store in an astropy Table
        tab = Table()
        tab.add_column(Column(fluxes[~fluxes.mask], name='flux'))
        tab.add_column(Column(fluxes_err[~fluxes_err.mask], name='fluxerr'))
        #tab.add_column(Column(fluxes_obs, name='flux'))
        #tab.add_column(Column(fluxes_obs_err, name='fluxerr'))
        tab.add_column(Column(phases[~phases.mask], name='phase'))
        tab.add_column(Column(snr_m5[~snr_m5.mask], name='snr_m5'))
        tab.add_column(Column(mag_obs[~mag_obs.mask], name='mag'))
        tab.add_column(
            Column((2.5/np.log(10.))/snr_m5[~snr_m5.mask], name='magerr'))
        tab.add_column(Column(obs_time[~obs_time.mask], name='time'))

        tab.add_column(
            Column(['LSST::'+band]*len(tab), name='band',
                   dtype=h5py.special_dtype(vlen=str)))

        tab.add_column(Column([2.5*np.log10(3631)]*len(tab),
                              name='zp'))

        tab.add_column(
            Column(['ab']*len(tab), name='zpsys',
                   dtype=h5py.special_dtype(vlen=str)))

        tab.add_column(Column(seasons[~seasons.mask], name='season'))

        tab.add_column(Column(healpixIds[~healpixIds.mask], name='healpixId'))
        tab.add_column(Column(pixRas[~pixRas.mask], name='pixRa'))
        tab.add_column(Column(pixDecs[~pixDecs.mask], name='pixDec'))

        tab.add_column(Column(z_vals, name='z'))
        tab.add_column(Column(daymax_vals, name='daymax'))

        tab.add_column(Column(self.reference_lc.mag_to_flux_e_sec[band](
            tab['mag'])), name='flux_e_sec')
        for key, vals in Fisher_Mat.items():
            tab.add_column(Column(vals[~vals.mask], name='F_'+key))
        """

        #Store in a panda dataframe
        lc = pd.DataFrame()
        lc['flux'] = fluxes[~fluxes.mask]
        lc['fluxerr'] = fluxes_err[~fluxes_err.mask]
        lc['phase'] = phases[~phases.mask]
        lc['snr_m5'] = snr_m5[~snr_m5.mask]
        lc['m5'] = m5_obs[~m5_obs.mask]
        lc['mag'] = mag_obs[~mag_obs.mask]
        lc['magerr'] = (2.5/np.log(10.))/snr_m5[~snr_m5.mask]
        lc['time'] = obs_time[~obs_time.mask]
        lc['exposuretime'] = exp_time[~exp_time.mask]
        lc['band'] = ['LSST::'+band]*len(lc)
        lc['zp'] = [2.5*np.log10(3631)]*len(lc)
        lc['zpsys'] = ['ab']*len(lc)
        lc['season'] = seasons[~seasons.mask]
        lc['healpixID'] = healpixIds[~healpixIds.mask]
        lc['pixRa'] = pixRas[~pixRas.mask]
        lc['pixDec'] =pixDecs[~pixDecs.mask]
        lc['z'] = z_vals
        lc['daymax'] = daymax_vals
        lc['flux_e_sec'] = self.reference_lc.mag_to_flux_e_sec[band](
            lc['mag'])
        lc['flux_5'] = self.reference_lc.mag_to_flux_e_sec[band](
            lc['m5'])
        for key, vals in Fisher_Mat.items():
            lc['F_{}'.format(key)] = vals[~vals.mask]

                     
        
        if output_q is not None:
            output_q.put({j: lc})
        else:
            return lc

    def srand(self, gamma, mag, m5):
        x = 10**(0.4*(mag-m5))
        return np.sqrt((0.04-gamma)*x+gamma*x**2)

"""
    def Plot_Sigma_c(self, tab):

        import pylab as plt
        #season = 1

        for season in np.unique(tab['season']):
            idx = tab['season'] == season
            sel = tab[idx]
            print('number of LC', len(np.unique(sel['daymax'])))
            for daymax in np.unique(sel['daymax']):
                idxb = np.abs(sel['daymax']-daymax) < 1.e-5
                selb = sel[idxb]
                plt.plot(selb['z'], selb['sigma_color'], 'k.')
        plt.show()

    def dpotrf(self, M):
        return lapack.dpotrf(M, lower=True, overwrite_a=True)[0]

    def dtrtri(self, M):
        return lapack.dtrtri(M, lower=True)[0]

    def matmul(self, a, b, out):
        return np.matmul(a, b, out=out)
"""
"""
    def invert_matrix(self, M, dtype="float32"):
        #Invert a positive definite matrix using cholesky decomposition.
        #WARNING : This DOES NOT check if the matrix is positive definite and can lead to wrong results if a non positive definite matrix is given.

        #Arguments:
        #- `M`: Matrix to invert
        
        M = self.dpotrf(M)  # L
        invL = np.ascontiguousarray(self.dtrtri(M), dtype=dtype)  # invL
        if np.__version__ == '1.13.3':
            self.matmul(invL.T, invL, out=invL)  # invC
        else:
            invL = np.matmul(invL.T, invL)
        return M, invL
"""
"""
    def Add_Gamma(self, obs):

        gamma = self.telescope.gamma(obs[self.m5Col], [
                                     seas[self.filterCol][-1] for seas in obs], obs[self.exptimeCol])
        obs = rf.append_fields(obs, 'gamma', gamma)

        return obs

    def Proc_Extend(self, param, obs_c, j=-1, output_q=None):

        obs_tot = None

        time_ref = time.time()

        obs_c = rf.append_fields(obs_c, 'daymax', [param['daymax']]*len(obs_c))
        obs_c = rf.append_fields(obs_c, 'z', [param['z']]*len(obs_c))
        obs_c = rf.append_fields(obs_c, 'x1', [param['x1']]*len(obs_c))
        obs_c = rf.append_fields(obs_c, 'color', [param['color']]*len(obs_c))

        if output_q is not None:
            output_q.put({j: lc})
        else:
            return obs_c
"""
"""
    def Multiproc(self, obs):

        tab_tot = Table()
       

        njobs = -1
        result_queue = multiprocessing.Queue()
       
        for band in np.unique(obs[self.filterCol]):
            idx = obs[self.filterCol] == band
            obs_b = obs[idx]

            if len(obs_b) > 0:
                njobs += 1
             #print('starting multi')
                p = multiprocessing.Process(
                    name='Subprocess-'+str(njobs), target=self.Process, args=(obs_b, band, njobs, result_queue))
                p.start()
             #print('starting multi done')
        resultdict = {}
        for j in range(njobs+1):
            resultdict.update(result_queue.get())

        for p in multiprocessing.active_children():
            p.join()

        # print('stacking')

        for j in range(0, njobs+1):
            if resultdict[j] is not None:
                tab_tot = vstack([tab_tot, resultdict[j]])

        return tab_tot
"""
"""
    def Process(self, obs, band, j=-1, output_q=None):

        deriv = {}

        diff = np.copy(obs[self.mjdCol]-obs['daymax'])
        x = self.lc_ref[band]['time']
        y = self.lc_ref[band]['z']
        z = self.lc_ref[band]['flux']
        zb = self.lc_ref[band]['fluxerr']
        xi = np.copy(obs[self.mjdCol]-obs['daymax'])
        yi = obs['z']

        method = 'nearest'
        fluxes_obs = griddata((x, y), z, (xi, yi), method=method)
        fluxes_err = griddata((x, y), zb, (xi, yi), method=method)


        mag_obs = -2.5*np.log10(fluxes_obs/3631.)

        m5 = np.asarray([self.m5_ref[band]]*len(obs))
        gammaref = np.asarray([self.gamma_ref[band]]*len(obs))

        correct_m5 = self.srand(gammaref, mag_obs, m5) / \
            self.srand(obs['gamma'], mag_obs, obs[self.m5Col])
        
        tab = Table()

        tab.add_column(Column(fluxes_obs, name='flux'))
        tab.add_column(Column(fluxes_err/correct_m5, name='fluxerr'))
        snr_m5_opsim = fluxes_obs/(fluxes_err/correct_m5)
        tab.add_column(Column(snr_m5_opsim, name='snr_m5'))
        tab.add_column(Column(mag_obs, name='mag'))
        tab.add_column(Column((2.5/np.log(10.))/snr_m5_opsim, name='magerr'))
        tab.add_column(Column(obs[self.mjdCol], name='time'))
        tab.add_column(
            Column(['LSST::'+obs[self.filterCol][i][-1]
                    for i in range(len(obs[self.filterCol]))], name='band',
                   dtype=h5py.special_dtype(vlen=str)))
        tab.add_column(Column([2.5*np.log10(3631)]*len(obs),
                              name='zp'))
        tab.add_column(
            Column(['ab']*len(obs), name='zpsys',
                   dtype=h5py.special_dtype(vlen=str)))

        idx = tab['flux'] >= 0.
        tab = tab[idx]

        if output_q is not None:
            output_q.put({j: tab})
        else:
            return tab

    def srand(self, gamma, mag, m5):
        x = 10**(0.4*(mag-m5))
        return np.sqrt((0.04-gamma)*x+gamma*x**2)
"""
"""
class SNFast_old:

    def __init__(self, fieldname, fieldid, sim_name, sim_type, X1, Color, zvals):

        file_obs = '/sps/lsst/users/gris/Files_from_OpSim/OpSimLogs_' + \
            str(sim_name)+'/'+fieldname+'/Observations_' + \
            fieldname+'_'+str(fieldid)+'.txt'
        print('loading obs')
        time_ref = time.time()
        self.telescope = Telescope(airmass=1.2)
        observation = Observations(int(fieldid), filename=file_obs)
        print('end of loading obs', time.time()-time_ref)
        # print(len(observation.seasons))

        self.seasons = {}
        for i in range(len(observation.seasons)):
            self.seasons[i] = self.Add_Gamma(observation.seasons[i])[
                ['band', 'mjd', 'm5sigmadepth', 'gamma']]
            print(i, self.season_limit(self.seasons[i]))
        self.dir_out = '/sps/lsst/users/gris/Light_Curves_' + \
            sim_type+'/'+str(sim_name)
        if not os.path.isdir(self.dir_out):
            os.makedirs(self.dir_out)

        self.fieldname = fieldname
        self.fieldid = fieldid
        self.sim_name = sim_name
        self.sim_type = sim_type

        self.X1 = X1
        self.Color = Color
        print('loading ref', zvals)
        self.lc_ref_dict = Load_Reference(X1, Color, zvals).tab

    def __call__(self, season, param):

        outname = self.fieldname+'_'
        outname += str(self.fieldid)+'_'
        outname += 'X1_'+str(self.X1)+'_Color_'+str(self.Color)+'_'
        outname += 'season_'+str(season)+'_'
        self.name_out = self.dir_out+'/'+'LC_'+outname+'.hdf5'
        self.name_ana_out = 'Summary_'+outname+'.npy'

        time_ref = time.time()
        # print('calling',np.unique(param['z']))

        self.Extend_Observations(param, self.lc_ref_dict)

        print('Time', time.time()-time_ref, len(param))

    def Add_Gamma(self, season):

        obs_c = season.copy()
        gamma = self.telescope.gamma(obs_c['m5sigmadepth'], [
                                     seas['band'][-1] for seas in obs_c], obs_c['exptime'])
        obs_c = rf.append_fields(obs_c, 'gamma', gamma)

        return obs_c

    def season_limit(self, myseason):
        prefix = ''
        if '::' in myseason['band'][0]:
            prefix = myseason['band'][0].split('::')[0]
            prefix += '::'
        iddx = np.asarray([seas['band'] != prefix+'u' for seas in myseason])

        mysel = myseason[iddx]
        mysel = np.sort(mysel, order='mjd')

        min_season = np.min(mysel['mjd'])
        max_season = np.max(mysel['mjd'])

        return (min_season, max_season)

    def Proc_Extend(self, params, lc_ref_dict, j=-1, output_q=None):

        obs_tot = None
        X1 = np.unique(params['X1'])[0]
        Color = np.unique(params['Color'])[0]

        #print('X1 and Color',len(params),X1,Color)
        time_ref = time.time()
        for param in params:
            season = self.seasons[param['season']]
            obs_c = np.copy(season)

            obs_c = rf.append_fields(
                obs_c, 'daymax', [param['daymax']]*len(obs_c))
            mean_restframe_wavelength = np.asarray(
                [self.telescope.throughputs.mean_wavelength[obser['band'][-1]] / (1. + param['z']) for obser in obs_c])
            p = (obs_c['mjd']-obs_c['daymax'])/(1.+param['z'])

            # idx=(np.min(p)<=-5)&(np.max(p)>=10)

            idx = (p >= min_rf_phase) & (p <= max_rf_phase) & (
                mean_restframe_wavelength > blue_cutoff) & (mean_restframe_wavelength < red_cutoff)
            obs_c = obs_c[idx]

            if len(obs_c) > 1:
                obs_c = rf.append_fields(obs_c, 'z', [param['z']]*len(obs_c))
                obs_c = rf.append_fields(obs_c, 'x1', [param['x1']]*len(obs_c))
                obs_c = rf.append_fields(
                    obs_c, 'color', [param['color']]*len(obs_c))
                obs_c = rf.append_fields(obs_c, 'SNID', [100+j]*len(obs_c))
                if obs_tot is None:
                    obs_tot = obs_c
                else:
                    obs_tot = np.concatenate((obs_tot, obs_c))

        # print('alors',len(obs_tot),np.unique(obs_tot['DayMax']),time.time()-time_ref)
        lc = None
        if obs_tot is not None:
            time_proc = time.time()
            # print('processing')
            lc = Process_X1_Color(X1, Color, obs_tot,
                                  lc_ref_dict, self.telescope).lc_fast
            # print('processed',len(lc),np.min(lc['DayMax']),np.max(lc['DayMax']),time.time()-time_proc)
            if len(lc) == 0:
                lc = None
        if output_q is not None:
            output_q.put({j: lc})
        else:
            return obs_c

    def Extend_Observations(self, param, lc_ref_dict):

        distrib = param
        nlc = len(distrib)
        # n_multi=7
        n_multi = min(7, nlc)
        n_multi = min(2, nlc)
        print('extended ', n_multi, nlc)
        nvals = nlc/n_multi
        batch = range(0, nlc, nvals)
        if batch[-1] != nlc:
            batch = np.append(batch, nlc)

        result_queue = multiprocessing.Queue()

        print('hello', batch)
        for i in range(len(batch)-1):

            ida = int(batch[i])
            idb = int(batch[i+1])
            print('processing here', ida, idb, param[ida:idb])
            p = multiprocessing.Process(name='Subprocess-'+str(
                i), target=self.Proc_Extend, args=(param[ida:idb], lc_ref_dict, i, result_queue))
            p.start()
        # print('processing',ida,idb,nlc,len(param[ida:idb]))

        resultdict = {}
        for j in range(len(batch)-1):
            resultdict.update(result_queue.get())

        for p in multiprocessing.active_children():
            p.join()

        lc = Table()
        for j in range(len(batch)-1):
            #print('hello lc',len(resultdict[j]))
            if resultdict[j] is not None:
                lc = vstack([lc, resultdict[j]])
        # print('calling')
        if not os.path.isfile(self.name_out):
            itot = 0
            lc.write(self.name_out, path='lc_'+str(itot), compression=True)
            time_ana = time.time()
            #print('LC Analysis')
            Ana_LC(lc=lc, outname=self.name_ana_out, sim_name=self.sim_name,
                   dir_lc=self.dir_out, sim_type=self.sim_type)
            #print('after analysi',time.time()-time_ana)
        else:
            print('This file', self.name_out, 'already exists -> out')
"""
"""
class Process_X1_Color:
    def __init__(self, X1, Color, obs_extended, lc_ref_tot, telescope):

        bands = np.unique(lc_ref_tot['band'])

        self.lc_ref = {}

        self.gamma_ref = {}
        self.m5_ref = {}

        self.key_deriv = ['dX1', 'dColor', 'dX0']

        for band in bands:

            idx = lc_ref_tot['band'] == band
            self.lc_ref[band] = lc_ref_tot[idx]

            self.gamma_ref[band] = self.lc_ref[band]['gamma'][0]
            self.m5_ref[band] = np.unique(lc_ref_tot[idx]['m5'])[0]

        if len(obs_extended) > 0:
            self.Multiproc(obs_extended, telescope, bands)
          

    def Process_Single(self, obs, telescope, bands):
        tab_tot = Table()
        prefix = ''
        if '::' in obs['band'][0]:
            prefix = obs['band'][0].split('::')[0]
            prefix += '::'
        for band in bands:
            idx = obs['band'] == prefix+band
            obs_b = obs[idx]
            if len(obs_b) > 0:
                # lc=self.Process(obs_b,band,telescope)
                tab_tot = vstack(
                    [tab_tot, self.Process(obs_b, band, telescope)])
                # tab_tot=vstack([tab_tot,lc])

        self.lc_fast = tab_tot

    def Multiproc(self, obs, telescope, bands):

        tab_tot = Table()
        

        njobs = -1
        result_queue = multiprocessing.Queue()
        prefix = ''
        if '::' in obs['band'][0]:
            prefix = obs['band'][0].split('::')[0]
            prefix += '::'
        for band in bands:
            idx = obs['band'] == prefix+band
            obs_b = obs[idx]

            if len(obs_b) > 0:
                njobs += 1
             #print('starting multi')
                p = multiprocessing.Process(name='Subprocess-'+str(
                    njobs), target=self.Process, args=(obs_b, band, telescope, njobs, result_queue))
                p.start()
             #print('starting multi done')
        resultdict = {}
        for j in range(njobs+1):
            resultdict.update(result_queue.get())

        for p in multiprocessing.active_children():
            p.join()

        # print('stacking')

        for j in range(0, njobs+1):
            if resultdict[j] is not None:
                tab_tot = vstack([tab_tot, resultdict[j]])

            # print(tab_tot.dtype)
            # print(len(np.unique(tab_tot['DayMax'])))

      
        self.lc_fast = tab_tot
        #print('there we go',len(tab_tot),len( self.lc_fast))

    def Process(self, obs, band, telescope, j=-1, output_q=None):

        deriv = {}

        diff = np.copy(obs['mjd']-obs['daymax'])
        x = self.lc_ref[band]['time']
        y = self.lc_ref[band]['z']
        z = self.lc_ref[band]['flux']
        zb = self.lc_ref[band]['fluxerr']
        xi = np.copy(obs['mjd']-obs['daymax'])
        yi = obs['z']

        method = 'nearest'
        fluxes_obs = griddata((x, y), z, (xi, yi), method=method)
        fluxes_err = griddata((x, y), zb, (xi, yi), method=method)

      
        mag_obs = -2.5*np.log10(fluxes_obs/3631.)

        m5 = np.asarray([self.m5_ref[band]]*len(obs))
        gammaref = np.asarray([self.gamma_ref[band]]*len(obs))

        correct_m5 = self.srand(
            gammaref, mag_obs, m5)/self.srand(obs['gamma'], mag_obs, obs['m5sigmadepth'])
        for key in self.key_deriv:
            xa = self.lc_ref[band]['time']
            ya = self.lc_ref[band]['z']
            za = self.lc_ref[band][key]

            deriv[key] = griddata((xa, ya), za, (xi, yi), method=method)

        tab = Table()

        tab.add_column(Column(fluxes_obs, name='flux'))

        tab.add_column(Column(mag_obs, name='mag'))
        tab.add_column(Column(fluxes_err/correct_m5, name='fluxerr'))
        tab.add_column(Column(obs['mjd'], name='time'))
        tab.add_column(Column(['LSST::'+band]*len(diff), name='band'))
        tab.add_column(Column(obs['m5sigmadepth'], name='m5sigmadepth'))
        tab.add_column(Column(obs['daymax'], name='daymax'))
        # tab.add_column(Column(obs['season'],name='season'))
        tab.add_column(Column([2.5*np.log10(3631)]*len(tab), name='zp'))
        tab.add_column(Column(['ab']*len(tab), name='zpsys'))
        tab.add_column(Column(obs['z'], name='z'))
        tab.add_column(Column(obs['X1'], name='X1'))
        tab.add_column(Column(obs['Color'], name='Color'))

        # tab.add_column(Column(obs['SNID'],name='SNID'))
        for key in self.key_deriv:
            tab.add_column(Column(deriv[key], name=key))

        if output_q is not None:
            output_q.put({j: tab})
        else:
            return tab

    def srand(self, gamma, mag, m5):
        x = 10**(0.4*(mag-m5))
        return np.sqrt((0.04-gamma)*x+gamma*x**2)
"""
"""
class Load_Reference:

    def __init__(self, filename, zvals):

        self.fi = filename
        self.tab = self.Read_Ref(zvals)

    def Read_Ref(self, zvals, j=-1, output_q=None):

        tab_tot = Table()
       
        f = h5py.File(self.fi, 'r')
        keys = f.keys()
        zvals = np.arange(0.01, 0.9, 0.01)
        zvals_arr = np.array(zvals)

        for kk in keys:

            tab_b = Table.read(self.fi, path=kk)
            # tab_tot=vstack([tab_tot,tab_b])

            if tab_b is not None:
                diff = tab_b['z']-zvals_arr[:, np.newaxis]
                #flag = np.abs(diff)<1.e-3
                flag_idx = np.where(np.abs(diff) < 1.e-3)
                if len(flag_idx[1]) > 0:
                    tab_tot = vstack([tab_tot, tab_b[flag_idx[1]]])

           
        if output_q is not None:
            output_q.put({j: tab_tot})
        else:
            return tab_tot

    def Read_Multiproc(self, tab):

        # distrib=np.unique(tab['z'])
        nlc = len(tab)
        print('ici pal', nlc)
        # n_multi=8
        if nlc >= 8:
            n_multi = min(nlc, 8)
            nvals = nlc/n_multi
            batch = range(0, nlc, nvals)
            batch = np.append(batch, nlc)
        else:
            batch = range(0, nlc)

        # lc_ref_tot={}
        #print('there pal',batch)
        result_queue = multiprocessing.Queue()
        for i in range(len(batch)-1):

            ida = int(batch[i])
            idb = int(batch[i+1])

            p = multiprocessing.Process(
                name='Subprocess_main-'+str(i), target=self.Read_Ref, args=(tab[ida:idb], i, result_queue))
            p.start()

        resultdict = {}
        for j in range(len(batch)-1):
            resultdict.update(result_queue.get())

        for p in multiprocessing.active_children():
            p.join()

        tab_res = Table()
        for j in range(len(batch)-1):
            if resultdict[j] is not None:
                tab_res = vstack([tab_res, resultdict[j]])

        return tab_res
"""
