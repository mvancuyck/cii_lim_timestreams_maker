import h5py, os, sys, argparse, time
sys.path.append('../simulations')
import sim_tools_flatsky, sim_tools_tod
from src.load_params import load_params
import numpy as np
#from pylab import *
from progress.bar import Bar
import pandas as pd
from IPython import embed
import astropy.units as u 
from scipy import interpolate
from astropy.wcs import WCS
from astropy.io import fits
from progress.bar import Bar

import numpy as np

def make_gaussian_realisations_uncorrelated(el, cl_dict, sim_shape, pixel_res_radians_or_inv_samp_freq):
    """
    Generate uncorrelated Gaussian realisations for n PSDs.

    Parameters
    ----------
    el : array
        Multipoles or frequency indices (used for 2D map conversion if needed)
    cl_dict : dict
        Dictionary of auto-spectra. Keys: 0, 1, 2,... for each timestream.
        Values: 1D or 2D PSDs.
    sim_shape : tuple
        Shape of each simulation (2D maps: (ny, nx) or 1D timestreams: (nt,)).
    pixel_res_radians_or_inv_samp_freq : float
        Pixel resolution in radians (maps) or inverse sampling frequency (tods)

    Returns
    -------
    sim_arr : ndarray
        Array of simulated timestreams/maps, shape (n_sims, ...) 
        where ... is sim_shape.
    """

    n_sims = int(len(cl_dict)**(1/2))

    # Determine FFT and normalization depending on sim shape
    sim_dim = np.ndim(sim_shape)
    if sim_dim == 2:  # maps
        fftvar = np.fft.fft2
        ifftvar = np.fft.ifft2
        ny, nx = sim_shape
        dx = dy = pixel_res_radians_or_inv_samp_freq
        norm = np.sqrt(1./(dx*dy))
    else:  # 1D timestream
        fftvar = np.fft.fft
        ifftvar = np.fft.ifft
        nx = sim_shape[0]
        dx = pixel_res_radians_or_inv_samp_freq
        norm = np.sqrt(1./dx**2)

    sim_arr = []

    bar = Bar(f'Generate PSDs', max=n_sims)
    for i in range(n_sims):
        pspec = cl_dict[i,i]
        # Clean PSD
        pspec = np.nan_to_num(pspec, nan=0., posinf=0., neginf=0.)

        # Generate Gaussian white noise
        if sim_dim == 2:
            g = np.random.standard_normal((ny, nx))
        else:
            g = np.random.standard_normal(nx)

        # FFT
        g_fft = fftvar(g)

        # Multiply by sqrt(PSD) (PSD can be 1D or 2D)
        if np.ndim(pspec) == 1 and sim_dim == 2:
            # convert 1D PSD to 2D
            pspec_2d = cl_to_cl2d(el, pspec, sim_shape, pixel_res_radians_or_inv_samp_freq)
        else:
            pspec_2d = pspec

        g_fft_scaled = g_fft * np.sqrt(pspec_2d)

        # Back to real space
        g_sim = ifftvar(g_fft_scaled).real * norm
        g_sim -= np.mean(g_sim)

        sim_arr.append(g_sim)
        bar.next()
    bar.finish
    sim_arr = np.asarray(sim_arr)
    return sim_arr

def gaussian_random_tod(l, clt, nx, res, l_cutoff=None):

    """
    Generate a gaussian random noise timestream from a power spectrum.  

    Parameters
    ----------
    l:  array
        wavenumber array
    clt: array
        power spectrum amplitude array
    nx: int
        the number of elements of the timestream
    res: float
        the (time) resolution
    l_cutoff: float
        maximum wavenumber used in the generation of the timestream. 

    Returns
    -------
    real_space_tod: array
        the generated timestream. 
    """ 
    
    lmap_rad_x = nx*res
    #Generate gaussian amplitudes
    norm = 1/res

    np.random.seed()

    noise = np.random.normal(loc=0, scale=1, size=nx)
    dmn1  = np.fft.fft( noise )

    if(l_cutoff is not None): lmax = np.minimum(l_cutoff, l.max()) 
    else: lmax = l.max()

    cl_map = np.zeros(l.shape) 
        
    w = np.where((l>0) & (l<=lmax))

    f = interpolate.interp1d( l, clt,  kind='linear')
    cl_map[w] = f(l[w])
    w1 = np.where( cl_map <= 0)
    if(w1[0].shape[0] != 0): cl_map[w1] = 0
    #Fill amn_t
    amn_t = dmn1 * norm * np.sqrt( cl_map )
    
    #Output map
    real_space_tod = np.real(np.fft.ifft( amn_t ))
    
    return real_space_tod

def make_correlated_timestreams(total_detectors, T, sample_freq, tod_len, tod_shape, fmin, fmax, nsims, tod_noise_level, fknee, alphaknee, rho_one_over_f, plot=False):

    '''
    For pixels seeing the same beam, but at different frequency bands, 
    this function generates noise timestreams with the 1/f correlated,
    due to all the pixel seeing the same atmosphere. 

    Parameters
    ----------
    total_detectors: int
        the number of detectors to generate noise for. 
    T: array
        the time timestreamsim_tools_tod.get_correlated_powspec
    sample_freq: float
        the qcquisition rate of T and of the sky TODs. 
    tod_len: int
        lenght of the noise TODs
    tod_shape: list
        list of the TODs dimensions
    fmin: float
        minimum frequency from which to generate the noise TOD. NOT IMPLEMENTED 
    fmax: float
        maximum frequency from which to generate the noise TOD. NOT IMPLEMENTED 
    nsims: int
        number of simulation to generate. 
    tod_noise_level: float

    fknee: float

    alphaknee: float

    rho_one_over_f: float

    plot: bool
        If to plot a summary plot for the group of pixels or not.

    Returns
    -------
    noise_tods_list: list
        list containing the n=total_detectors noise timestreams.

    '''
    

    #------------------------------------------------------------------
    #define some detectors
    detector_array = np.arange( total_detectors )
    detector_combs_autos   = [[detector, detector] for detector in detector_array]
    detector_combs_crosses = [[detector1, detector2] for detector1 in detector_array for detector2 in detector_array if (detector1!=detector2 and detector1<detector2)]
    detector_combs = detector_combs_autos + detector_combs_crosses
    #------------------------------------------------------------------

    #------------------------------------------------------------------
    #Generate the noise model for auto- and cross-power between detectors

    freq_fft = np.fft.fftfreq(tod_len, 1/sample_freq) #TOD frequencies.
    inds = np.where(freq_fft>0) 
    
    freq, noise_powspec, noise_powspec_one_over_f, noise_powspec_white = sim_tools_tod.detector_noise_model(tod_noise_level, fknee, alphaknee, tod_len, sample_freq)
    cross_noise_powspec = sim_tools_tod.get_correlated_powspec(rho_one_over_f, noise_powspec_one_over_f, noise_powspec_one_over_f)
    noise_powspec_dic = {}

    bar = Bar(f'Generate PSDs', max=total_detectors**2)
    for i in range(total_detectors):
        for j in range(total_detectors):
            if i == j:
                noise_powspec_dic[i, j] = noise_powspec
            else:
                noise_powspec_dic[i, j] = cross_noise_powspec
            bar.next()
    bar.finish


    #------------------------------------------------------------------
    #If the detectors don't all have a noise timestream yet, generate correlated power spectra
    tod_sims_dic = {}
    pspec_dic_sims = {}

    for sim_no in range(nsims):
        #Generate un-correlated simulated timestreams
        
        #tod_sim_arr = sim_tools_flatsky.make_gaussian_realisations(freq_fft, noise_powspec_dic, tod_shape, 1./sample_freq)
        tod_sim_arr = make_gaussian_realisations_uncorrelated(freq_fft, noise_powspec_dic, tod_shape, 1./sample_freq)
        embed()
        #get the correlated power spectra now.
        curr_sim_pspec_dic = {}
        #bar = Bar(f'Generate correlation sim n{sim_no}', max=len(tod_sim_arr)**2)
        for (cntr1, tod1) in enumerate( tod_sim_arr ):
            for (cntr2, tod2) in enumerate( tod_sim_arr ):
                if cntr2<cntr1: continue       
                curr_spec = ( np.fft.fft(tod1) * (1/sample_freq) * np.conj( np.fft.fft(tod2) * (1/sample_freq) ) / tod_len  ).real
                curr_sim_pspec_dic[(cntr1, cntr2)] = [freq_fft, curr_spec]
                #bar.next()
        #bar.finish
        pspec_dic_sims[sim_no] = curr_sim_pspec_dic
    
    #------------------------------------------------------------------

    #------------------------------------------------------------------
    #From the power spectra, generate random gaussian TODs and save them. 
    curr_spec_list = []
    for d1d2 in detector_combs_autos:
        d1, d2 = d1d2
        curr_theory = noise_powspec_dic[(d1, d2)]
        
        #sims
        curr_spec_arr = []
        for sim_no in pspec_dic_sims:
            curr_freq, curr_spec = pspec_dic_sims[sim_no][(d1, d2)]
            curr_spec_arr.append( curr_spec )
        curr_spec_list.append( np.mean( curr_spec_arr, axis = 0 ) )

    noise_tods_list = []

    #opf.write(f'get noise tod'); opf.flush()
    for d1d2 in detector_combs:
        d1, d2 = d1d2
        curr_theory = noise_powspec_dic[(d1, d2)]
        if(d1==d2):
            noise_tod = gaussian_random_tod(freq_fft, curr_spec_list[d1], res = (1/sample_freq), nx = tod_len)
            noise_tods_list.append(noise_tod)
    #------------------------------------------------------------------

    return noise_tods_list

if __name__ == "__main__":

    '''
    Generate noise TOD. 
    The noise TOD are frequency, detector, and time independant.
    The noise TODs are gaussian. 
    The noise TODs for pixels seeing the same beam (at different frequency bands) are correlated. 
    '''
    #------------------------------------------------------------------------------------------
    #load the .par file parameters

    parser = argparse.ArgumentParser(description="strategy parameters",
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    #options
    parser.add_argument('params', help=".par file with params", default = None)
    parser.add_argument('--non_iteractive', help = "deactivate matplotlib", action="store_true")

    args = parser.parse_args()

    P = load_params(args.params)

    #------------------------------------------------------------------------------------------
    
    #Load the scan duration and generate the time coordinates with the desired acquisition rate. 
    T_duration = P['T_duration']*3600
    dt = 1/P['acquisition_frequency']#Make the timestep non rational to avoid some stripes in the hitmap. 
    T = np.arange(0,T_duration,dt) 

    #TOD specs
    tod_len = len(T)
    tod_shape = [tod_len]
    fmin = P['fmin']
    fmax = P['fmax']
    tod_noise_level = P['tod_noise_level'] 
    fknee = P['fknee'] 
    alphaknee = P['alphaknee'] 
    rho_one_over_f = P['rho_one_over_f'] 
    nsims = P['nsim_noise']

    '''
    det_names_dict = pd.read_csv(P['detectors_name_file'], sep='\t')
    #Each pixel with the same offset sees the same beam, but in different frequency band. 
    same_offset_groups = det_names_dict.groupby(['XEL', 'EL'])['Name'].apply(list).reset_index()
    '''
    
    #For each group of pixels seeing the same beam: 
    #for group in range(len(same_offset_groups)):

        
    #total_detectors = len(same_offset_groups.iloc[group]['Name'])
            
    tod_list = make_correlated_timestreams(1000, T, P['acquisition_frequency'], tod_len, tod_shape, fmin, fmax, nsims, tod_noise_level, fknee, alphaknee, rho_one_over_f)
    embed()
    '''
    H = h5py.File(tod_file, "a")    
    for j, (tod, name) in  enumerate(zip(tod_list, same_offset_groups.iloc[group]['Name'])):
        namegrp = f'kid_{name}_roach'
        if namegrp not in H: grp = H.create_group(namegrp)
        else:                grp = H[namegrp]
        if('corr_noise_data' in grp): del grp['corr_noise_data'] 
        grp.create_dataset('corr_noise_data', data=tod, compression='gzip', compression_opts=9)
    H.close()
    '''
    