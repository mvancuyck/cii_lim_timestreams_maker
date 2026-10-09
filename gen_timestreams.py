import argparse
from src.load_params import *
from src.hdf5_fcts import * 
from src.scan_fcts import *
from src.astrometry_fcts import *
from astropy.io import fits
import astropy.units as u 
from astropy.coordinates import SkyCoord
from progress.bar import Bar
import pandas as pd
import h5py 
import matplotlib.patches as mpatches
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
from astropy.wcs import WCS
import datetime
from IPython import embed
from scipy.signal import resample_poly

def group_detectors(det_names_dict, array):

    lw = det_names_dict[det_names_dict['Array'] == array]

    # Group by (EL, XEL), sort offsets to keep a stable order
    grouped = (lw.groupby(['EL', 'XEL'])['Name'].apply(list).sort_index())  # ensure deterministic order)

    # These are the unique (EL, XEL) pairs in order
    offsets = list(grouped.index)  # list of (EL, XEL) tuples

    # Lists of detectors for each offset pair
    lists_per_offset = list(grouped.values)

    # Number of groups = min size among offset groups
    N_groups = min(len(lst) for lst in lists_per_offset)

    # Build the groups
    final_groups = []
    for i in range(N_groups):
        group = [lst[i] for lst in lists_per_offset]
        final_groups.append(group)

    return final_groups, offsets

def add_polynome_to_timestream(timestream, time, order=1, percent_scale=30, random_coeffs=True):
    """
    Add a polynomial trend (order 1 to 4) to a timestream.

    Parameters
    ----------
    timestream : array
        Input data array (only used to set scale).
    time : array
        Time array (same length as timestream).
    order : int, default=1
        Order of polynomial (1 = linear, up to 4).
    percent_scale : float, default=30
        Scale of polynomial relative to data range, in percent.
    random_coeffs : bool, default=True
        If True, coefficients are randomized within scale. 
        If False, only the highest order term is used.
    Returns
    ----------
    poly: array
        the resulting timestream
    """
    assert 1 <= order <= 4, "Order must be between 1 and 4."

    # Normalize time to [0,1] for stability
    t = (time - time[0]) / (time[-1] - time[0])

    data_range = np.max(timestream) - np.min(timestream)
    scale = (percent_scale / 100) * data_range

    # Generate coefficients
    if random_coeffs:
        coeffs = np.random.uniform(-scale, scale, order + 1)
    else:
        coeffs = np.zeros(order + 1)
        coeffs[-1] = scale  # only highest-order term

    poly = np.polyval(coeffs, t)

    return poly

def add_peaks_to_timestream(timestream,n_peaks,prominence_sigma=3,width=10,rng=None):
    """
    Add Gaussian peaks to a timestream.

    Parameters
    ----------
    timestream : array-like
        Input data.
    n_peaks : int
        Number of peaks to add.
    prominence_sigma : float
        Peak height in units of the input timestream's standard deviation.
    width : float
        Peak FWHM, in samples.
    rng : numpy.random.Generator, optional
        Random number generator. Pass one for reproducible results.

    Returns
    -------
    result : ndarray
        Input timestream with peaks added.
    """
    data = np.asarray(timestream, dtype=float)

    if data.ndim != 1: raise ValueError("timestream must be one-dimensional.")
    if n_peaks < 0: raise ValueError("n_peaks must be non-negative.")
    if width <= 0: raise ValueError("width must be positive.")

    rng = np.random.default_rng(rng)
    result = data.copy()

    amplitude = prominence_sigma * np.std(data)
    sample_indices = np.arange(data.size)

    # Convert Gaussian FWHM to its standard deviation.
    peak_sigma = width / (2 * np.sqrt(2 * np.log(2)))

    centers = rng.uniform(0, max(data.size - 1, 0), size=n_peaks)

    for center in centers: result += amplitude * np.exp(-0.5 * ((sample_indices - center) / peak_sigma) ** 2)

    return result

def gen_tod(wcs, Map, hdr, ybins, xbins, pointing_paths, T=None):

    """
    Generate the sky amplitude TODs for a set of detectors from the same (electromagnetic) frequency channel, using a simulated sky map.

    Parameters
    ----------
    wcs: astropy.wcs.wcs.WCS
        The wcs used to generate the TODs
    Map: 2d array
        the 2d angular map of the sky in a frequency channel
    ybins: array
        edges of the pixels
    xbins: array
        edges of the pixels
    pointing_paths: list of 2d array
        coordinates of the sky scan path of each pixel, for pixels seeing the same frequency band
    Returns
    -------
    hist: 2d array
        the reconstructed sky map given the pointing paths 
    norm: 2d array
        the hitmap
    samples: list
        list of the amplitude timestreams of each detectors
    positions_x: list
        list of RA coordinates timestreams of each detectors
    positions_y: list
        list of DEC coordinates timestreams of each detectors 
    """ 
    
    positions_x = np.zeros((len(pointing_paths), len(pointing_paths[0][:,0])))
    positions_y = np.zeros((len(pointing_paths), len(pointing_paths[0][:,0])))
    samples = np.zeros((len(pointing_paths), len(pointing_paths[0][:,0])))
    
    #We sample the map for each detector, following its path on the sky. 
    for detector, path in enumerate(pointing_paths):
        
        #Convert the path on the sky from WCS to pixel coordinates.
        y_pixel_coords, x_pixel_coords = wcs.world_to_pixel_values(pointing_paths[detector][:,0], pointing_paths[detector][:,1])    
        # Round the positions and convert to integer indices
        x_pixel_coords_rounded = np.round(x_pixel_coords).astype(int)
        y_pixel_coords_rounded = np.round(y_pixel_coords).astype(int)
        # Create a mask for positions within bounds of the map
        valid_mask = (
            (x_pixel_coords_rounded >= 0) & (x_pixel_coords_rounded < hdr['NAXIS1'] - 1) &  # x within bounds
            (y_pixel_coords_rounded >= 0) & (y_pixel_coords_rounded < hdr['NAXIS2'] - 1) )  # y within bounds
        # Initialize the output array with zeros
        values = np.zeros_like(x_pixel_coords_rounded, dtype=float)
        # Assign values from the map for valid positions
        values[valid_mask] = Map[x_pixel_coords_rounded[valid_mask], y_pixel_coords_rounded[valid_mask]]
        samples[detector,:] = np.asarray(values.astype(float))
        positions_x[detector,:] = x_pixel_coords
        positions_y[detector,:] = y_pixel_coords

        norm, edges = np.histogramdd(sample=(x_pixel_coords.ravel(), y_pixel_coords.ravel()), bins=(xbins,ybins),  )
        hist, edges = np.histogramdd(sample=(x_pixel_coords.ravel(), y_pixel_coords.ravel()), bins=(xbins,ybins), weights=samples[detector].ravel())
        #plt.figure()
        #plt.imshow(hist/norm, origin = 'lower')
        #plt.title(f'detector={detector}')
    #plt.show()

    #Compute the number of times each sky pixel is hit by the detectors.
    norm, edges = np.histogramdd(sample=(positions_x.ravel(), positions_y.ravel()), bins=(xbins,ybins),  )
    hist, edges = np.histogramdd(sample=(positions_x.ravel(), positions_y.ravel()), bins=(xbins,ybins), weights=samples.ravel())

    #Create the observed map: 
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", RuntimeWarning)  # Catch all runtime warnings
        hist /= norm  # Perform the division

    #Add a polynomial slope to data if the timestamps are provided, for Namap testing purposes only
    #Add peaks to the timestreams if the timestamps are provided, for Namap testing purposes only
    if(T is not None):
        for i,s in enumerate(samples):
            samples[i] += add_polynome_to_timestream(s, T, order=3, percent_scale=30, random_coeffs=True)
            samples[i]  = add_peaks_to_timestream(s,5)

    return hist, norm, samples, positions_x, positions_y

def main_tod(P):

    #-----------------------------
    #Initiate the parameters

    #The coordinates of the field
    name=P['name_field']
    c=SkyCoord.from_name(name)
    ra = 0 
    rafield = c.ra.value
    dec = c.dec.value

    #load the observer position
    lat = P['latitude']

    #Load the resolution. 
    #if not in params, load it from the map used to generate the TOD. 
    res = P['res']
    if(res is None):
        hdr = fits.getheader(P['path']+P['file'])
        res = (hdr['CDELT1'] * u.Unit(hdr['CUNIT1'])).to(u.deg).value
    
    acquisition_frequency = P['acquisition_frequency']  #sample per frame defined here as the acquisition rate in Hz. 
    dt = 1/acquisition_frequency/3600*np.pi/3.14 #Make the timestep non rational to avoid some stripes in the hitmap. 
    spf = np.round(acquisition_frequency).astype(int)

    tod_file=P['output_path']+P['output_name']

    if('.hdf5' in tod_file):
        H = h5py.File(tod_file, "a")
        T = H['data_time']['data'][()]
        LST = H['data_lst']['data'][()]
        RA_path = H['data_RA_path']['data'][()]
        DEC_path = H['data_DEC_path']['data'][()]
        H.close()
    else: 
        d = gd.dirfile(tod_file, gd.RDONLY)
        num = d.nframes
        first_frame = 0
        T = d.getdata('data_time', gd.FLOAT64, num_frames = num, first_frame=first_frame)
        T = np.asarray(T, dtype=np.float64)
        LST = d.getdata('data_lst', gd.FLOAT64, num_frames = num, first_frame=first_frame)
        LST = np.asarray(LST, dtype=np.float64)
        RA_path = d.getdata('data_RA_path', gd.FLOAT64, num_frames = num, first_frame=first_frame)
        RA_path = np.asarray(RA_path, dtype=np.float64)
        DEC_path = d.getdata('data_DEC_path', gd.FLOAT64, num_frames = num, first_frame=first_frame)
        DEC_path = np.asarray(DEC_path, dtype=np.float64)

    scan_path = np.asarray((RA_path, DEC_path)).T
    #-----------------------------

    #-------------------------------
    #Load the sky simulation from which to generate the TODs from
    simu_sky_path = P['path']+P['file'] #os.getcwd()
    hdr  = fits.getheader(simu_sky_path)
    pix_size = ((hdr['CDELT1']*u.Unit(hdr['CUNIT1']))**2).to(u.sr).value
    hdr['CRVAL1'] = ra 
    hdr['CRVAL2'] = dec
    hdr['CRPIX1'] = hdr['NAXIS1']//2
    hdr['CRPIX2'] = hdr['NAXIS2']//2
    wcs = WCS(hdr, naxis=2) 
    #Create the list of frequency channels of the simulated cube. 
    freqs =( np.arange(hdr['CRVAL3'], hdr['CRVAL3']+hdr['NAXIS3']*hdr['CDELT3'], hdr['CDELT3'])*u.Unit(hdr['CUNIT3']) ).to(u.GHz).value
    #Create the binning of the map in pixel coordinates. 
    xbins = np.arange(-0.5, hdr['NAXIS1']+0.5, 1)
    ybins = np.arange(-0.5, hdr['NAXIS2']+0.5, 1)
    #load the angular spectral cube. 
    
    if(simu_sky_path is not None):
        cube = fits.getdata(simu_sky_path)
        #Remove the mean in each map, to wich we are not sensitive. 
        cubemean = np.mean(cube, axis=(1,2)) 
        cube -= cubemean[:, None, None]
        cube *= pix_size #conversion Jy/sr to Jy/beam
        if('MJy' in hdr['BUNIT'] ): cube *= 1e6 #conversion MJy/beam to Jy/beam
    #-----------------------------

    #-----------------------------
    det_names_dict = pd.read_csv(P['detectors_name_file'], sep='\t')

    for array_name, freqs_array in zip( ('LW', 'SW'), 
                                   (freqs[:P['nb_channels_per_array']], 
                                    freqs[ P['nb_channels_per_array']:P['nb_channels_per_array']*2 ])):
        
        cube_simu=  []
        cube_obs =  []
        cube_hits = []

        groups, offset = group_detectors(det_names_dict, array_name)

        xel = np.asarray(offset)[:,1]
        el = np.asarray(offset)[:,0]

        pixel_offsets = pixels_rotations(el, xel, P['theta'])
        #Generate the pointing on the sky of each pixel. 
        pointing_paths_to_save = [genPointingPath(T, scan_path, LST, lat, dec, offsets) for offsets in pixel_offsets]
        #-------------------------------

        #------------------------------------------------------------------
        bar = Bar(f'Generate the TODs of the {array_name} array', max=len(freqs_array))
        #for each frequency,
        for fi, freq in enumerate(freqs_array):

            #----------------------------------------
            #select the detectors
            names = groups[fi]
            index = np.argmin(np.abs(freqs - freq))
            #----------------------------------------

            #----------------------------------------
            #Select the electromagnetic frequency channel out of which the TODs will be sampled. 
            Map = cube[index,:,:]
            cube_simu.append(Map/pix_size)
            #----------------------------------------

            #----------------------------------------
            hist, norm, samples, positions_x, positions_y = gen_tod(wcs, Map, hdr, ybins, xbins, pointing_paths_to_save, T=None)

            #---
            samples += np.nanmin(samples)+1e-5
            IQ = samples / np.sqrt(2)
            #---

            cube_obs.append(hist/pix_size)
            cube_hits.append(norm)
            #----------------------------------------

            #----------------------------------------
            fig, axs = plt.subplots(1,3, figsize=(12,4), dpi = 200,subplot_kw={'projection': wcs}, sharex=True, sharey=True )
            imgdec = axs[0].imshow(hist, interpolation='nearest', origin='lower', vmin=Map.min(), vmax=Map.max(), cmap='cividis' )
            img = axs[1].imshow(Map, interpolation='nearest', origin='lower', vmin=Map.min(), vmax=Map.max(), cmap='cividis' )
            count = axs[2].imshow(norm, interpolation='nearest', origin='lower', cmap='binary' )
            for ax in (axs[0], axs[1], axs[2]):
                lon = ax.coords[0]
                LAT = ax.coords[1]
                LAT.set_major_formatter('d.d')
                lon.set_major_formatter('d.d')
                lon.set_axislabel('RA')
                LAT.set_axislabel('Dec')
                if(ax is not axs[0]): ax.tick_params(axis='y', labelleft=False)
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.savefig('plots/'+f'freq{freq:.0f}GHz_channel_{P["scan"]}_summary_plot.png')
            plt.close()
            #----------------------------------------

            #----------------------------------------
            if('.hdf5' in tod_file): 
                save_tod_in_hdf5(tod_file, names, samples, el, xel, P['detectors_name_file'], freq, spf, acquisition_frequency, pointing_paths_to_save, save=P['format'], compression=P['compression'])
                save_tod_in_hdf5(tod_file, names, IQ, el, xel, P['detectors_name_file'], freq, spf, acquisition_frequency, pointing_paths_to_save, save=P['format'], compression=P['compression'], suffix = '_I')
                save_tod_in_hdf5(tod_file, names, IQ, el, xel, P['detectors_name_file'], freq, spf, acquisition_frequency, pointing_paths_to_save, save=P['format'], compression=P['compression'], suffix = '_Q')

            else: 
                save_tods_dirfile(tod_file, names, samples, P['detectors_name_file'], freq, spf, acquisition_frequency, el, xel)
                save_tods_dirfile(tod_file, names, IQ, P['detectors_name_file'], freq, spf, acquisition_frequency, el, xel, suffix='_I')
                save_tods_dirfile(tod_file, names, IQ, P['detectors_name_file'], freq, spf, acquisition_frequency, el, xel, suffix='_Q')


            bar.next()
        #------------------------------------------------------------------


        #-------------------------------------------------------------------------------------
        #Example for taking the PSD of downsampled timestreams
        if(False):
            H = h5py.File(tod_file, "a")
            flags = H['data_turnaround_flags']['data'][()]
            H.close()
            timestamps = T[flags==1]
            det_data = np.asarray(samples)[:, flags==1]
            dt = np.diff(timestamps)
            breaks = np.where(np.abs(np.diff(dt)) > 1)[0] + 1 #np.where(dt != dt[0])[0] + 1
            timestamp_chunks = np.split(timestamps, breaks)
            det_chunks = np.split(det_data, breaks, axis=1)
            acq_freq = P['acquisition_frequency'] #Hz
            for i, chunk in enumerate(det_chunks):
                for det_id, det_tod in enumerate(chunk):
                    if(len(det_tod) <= 1): continue
                    freq_fft = np.fft.fftfreq(len(det_tod), 1/acq_freq) #TOD frequencies.
                    inds = np.where(freq_fft>0) 
                    tod_psd = ( np.fft.fft(det_tod) * (1/acq_freq) * np.conj( np.fft.fft(det_tod) * (1/acq_freq) ) / len(det_tod)  ).real
                    plt.step(freq_fft[inds], tod_psd[inds], where='mid', alpha=0.1, c='k')

                    tod_resampled = resample_poly( det_tod, up=100, down=int(round(acq_freq)), window=('kaiser', 8.6))
                    freq_fft = np.fft.fftfreq(len(tod_resampled), 1/100) #TOD frequencies.
                    inds = np.where(freq_fft>0) 
                    tod_psd = ( np.fft.fft(tod_resampled) * (1/100) * np.conj( np.fft.fft(tod_resampled) * (1/100) ) / len(tod_resampled)  ).real
                    plt.step(freq_fft[inds], tod_psd[inds] /( len(det_tod) / len(tod_resampled) ), where='mid', alpha=0.1, c='r')

            plt.axvline(acq_freq, ymin=1e-20, ymax=1e10, color='grey', ls=':' , label=f'{acq_freq:.1f}Hz sampling frequency')
            plt.axvline(acq_freq/2, ymin=1e-20, ymax=1e10, color='grey', label=f'{acq_freq:.1f}Hz Nyquist frequency')
            plt.axvline(100/2, ymin=1e-20, ymax=1e10, color='k', label=f'{100:.1f}Hz Nyquist frequency')
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('f [Hz]')
            plt.ylabel('PSD [$\\rm Jy^2/beam.s^2$]')
            plt.legend()
            plt.show()
        #-------------------------------------------------------------------------------------
   

        # ------------------ Save FITS --------------------
        hdr_out = hdr
        hdr_out["DATE"]  = str(datetime.datetime.now())
        hdr_out["COMMENT"] = f'dt={dt:.3f}s'

        hdr_norm = hdr_out.copy()
        hdr_norm["BUNIT"] = "counts"

        hdus = [
            fits.PrimaryHDU(cube_simu, header=hdr_out),
            fits.ImageHDU(cube_obs, header=hdr_out,   name="SCANNED"),
            fits.ImageHDU(cube_hits, header=hdr_norm, name="HITMAP")
        ]
        
        hdul = fits.HDUList(hdus)
        savepath = f'{P["output_path"]}/scanned_map_{P["output_name"][:-5]}_{array_name}.fits'
        hdul.writeto(savepath, overwrite=True)
        bar.finish
        print('')
        print("saved ", savepath)
        print('')
        #----------------------------------------
        

if __name__ == "__main__":

    '''
    PAR_files/params_strategy.par is a file containing all the modifiable parameters. 
    To generate your TODs: 

    Step 1/3: Generate your observation scan path with python hitmap_1detector.py.py PAR_files/params_strategy.par
    Step 2/3: Generate your detector array with python gen_detectors_arrays.py PAR_files/params_strategy.par
    Step 3/3: Sample the TODs for your detector array following your observation scan path from a simulation with python gen_timestreams.py PAR_files/params_strategy.par
    '''
    #------------------------------------------------------------------------------------------
    #load the .par file parameters
    parser = argparse.ArgumentParser(description="strategy parameters",
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    #options
    parser.add_argument('params', help=".par file with params", default = None)
    parser.add_argument('--non_iteractive', help = "deactivate matplotlib", action="store_true")

    args = parser.parse_args()

    if(args.non_iteractive): 
        import matplotlib
        matplotlib.use("Agg")

    P = load_params(args.params)
    #------------------------------------------------------------------------------------------

    main_tod(P)