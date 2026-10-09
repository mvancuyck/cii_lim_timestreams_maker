import numpy as np
from src.astrometry_fcts import *
from IPython import embed
import pandas as pd
import scipy.constants as cst 
from specutils import Spectrum1D
from specutils.manipulation import LinearInterpolatedResampler
from scipy import interpolate

def hitsPerSqdeg(total_hits, area):
    """
    hits per sqare degree 
    Parameters
    ----------
    total_hits: 2d array
        hit map   
    area: float
        area in deg2
    res: float
        resolution in same units as area

    Returns
    -------
    hitsPerSqdeg: 2d array
        map of hits per square degree 
    """ 
    return np.sum(total_hits)/area

def timeFractionAbove(hmap, threshold):
    """
    fraction of time above a level of hits
    Parameters
    ----------
    hmap: 2d array
        hit map   
    treshold: int
        treshold of hits
    Returns
    -------
    timeFractionAbove: float
        fraction of the hitmaps above a level of hits
    """ 
    hits = hmap.flatten()
    return np.sum(hits[hits>threshold])/np.sum(hits)

def genLocalPath_cst_el_scan_crisscross(az_size = 1, alt_size = 1, alt_step=0.02, acc = 0.05, scan_v=0.05, dt= 0.01):

    """
    Function that generates the local scanning pattern.
    This function generate a constant elevation scan, that steps in elevation after each turn-around. 
    Parameters
    ----------
    az_size: float
        azimuth angular size, in degrees   
    alt_size: float
        altitude angular size, in degrees
    alt_step: float
        step in altitude angle, in degrees
    acc: float
        acceleration in second angle 
    scan_v: float
        angular speed of the scan, in deg/sec
    dt: float
        time step in second angle 
    Returns
    -------
    az: array
        azimuth scan path coordinates, in degrees
    alt: array
        altitude scan path coordinates, in degrees
    flag: array
        constant scan speed part.
    scan_eff: array
        scan efficiency: the ratio between the constant scan speed part and not constant scan speed part
    t: array
        time during the scan, in second angle
    """ 

    #----
    #Compute Number of Vertical Steps 
    vertical_steps = int(alt_size//alt_step)
    #Compute Time for Scan and Turns
    scan_time = az_size/scan_v #Time required to cover the full azimuth range at scan_v
    turn_time = 2*scan_v/acc #Time required to perform a turn (deceleration, reversal, acceleration)
    #Generate Azimuth Acceleration Pattern (az_acc):
    #The motion consists of acceleration, constant velocity, and deceleration, forming a symmetric back-and-forth oscillation in azimuth.
    az_acc = np.concatenate((np.ones(int(turn_time/dt))*acc,np.zeros(int(scan_time/dt))))
    az_acc = np.concatenate((az_acc,-1*az_acc))
    #The sequence is repeated for each altitude step (vertical_steps times).
    az_acc = np.tile(az_acc,vertical_steps)

    #Compute the Altitude Acceleration
    acc_alt_value =  alt_step/(turn_time/2)**2   

    #Generate Altitude Acceleration Pattern (acc_alt)
    #The altitude changes slightly during turns, using a small acceleration.
    oscillation = np.tile(
        np.concatenate([
            np.ones(int(turn_time / dt / 2)) * acc_alt_value,
            np.ones(int(turn_time / dt / 2)) * -acc_alt_value
        ]), 1
    )
    # Ensure no extra oscillation at the ends of azimuth scan
    acc_alt = np.concatenate((oscillation, np.zeros(int(scan_time / dt))))
    acc_alt = np.concatenate((acc_alt, acc_alt))  # No altitude change on the leftward scan
    acc_alt = np.tile(acc_alt, vertical_steps)

    #Compute Azimuth (az) and Altitude (alt) Coordinates:
    #Computed by integrating acceleration to get velocity, then integrating velocity to get position.
    az_v = np.cumsum(az_acc)*dt-scan_v
    az = np.cumsum(az_v)*dt
    alt_v = np.cumsum(acc_alt)*dt
    alt  = np.cumsum(alt_v)*dt

    flag = np.where(az_acc==0,1,0) #constant scan speed part
    return az,alt,flag  

def genLocalPath_cst_el_scan(az_size = 1, alt_size = 1, alt_step=0.02, acc = 0.05, scan_v=0.05, dt= 0.01):

    """
    Function that generates the local scanning pattern.
    This function generate a constant elevation scan, that steps in elevation at every other turn-around.
    Parameters
    ----------
    az_size: float
        azimuth angular size, in degrees   
    alt_size: float
        altitude angular size, in degrees
    alt_step: float
        step in altitude angle, in degrees
    acc: float
        acceleration in second angle 
    scan_v: float
        angular speed of the scan, in deg/sec
    dt: float
        time step in second angle 
    Returns
    -------
    az: array
        azimuth scan path coordinates, in degrees
    alt: array
        altitude scan path coordinates, in degrees
    flag: array
        constant scan speed part.
    scan_eff: array
        scan efficiency: the ratio between the constant scan speed part and not constant scan speed part
    t: array
        time during the scan, in second angle
    """ 

    #----
    #Compute Number of Vertical Steps 
    vertical_steps = int(alt_size//alt_step) 
    print(vertical_steps)

    #Compute Time for Scan and Turns
    scan_time = az_size/scan_v #Time required to cover the full azimuth range at scan_v
    turn_time = 2*scan_v/acc #Time required to perform a turn (deceleration, reversal, acceleration).

    #Generate Azimuth Acceleration Pattern (az_acc):
    #The motion consists of acceleration, constant velocity, and deceleration, forming a symmetric back-and-forth oscillation in azimuth.
    az_acc = np.concatenate((np.ones(int(turn_time/dt))*acc,np.zeros(int(scan_time/dt))))
    az_acc = np.concatenate((az_acc,-1*az_acc))
    #The sequence is repeated for each altitude step (vertical_steps times).
    az_acc = np.tile(az_acc,vertical_steps)

    #Compute the Altitude Acceleration
    acc_alt_value =  alt_step/(turn_time/2)**2   

    #Generate Altitude Acceleration Pattern (acc_alt)
    #The altitude changes slightly during turns, using a small acceleration.
    oscillation = np.tile(
        np.concatenate([
            np.ones(int(turn_time / dt / 2)) * acc_alt_value,
            np.ones(int(turn_time / dt / 2)) * -acc_alt_value
        ]), 1
    )

    # Ensure no extra oscillation at the ends of azimuth scan
    acc_alt = np.concatenate((oscillation, np.zeros(int(scan_time / dt))))
    #acc_alt = np.concatenate((acc_alt, -1 * acc_alt))  # Repeat for downward scan
    acc_alt = np.concatenate((acc_alt, np.zeros_like(acc_alt)))  # No altitude change on the leftward scan
    acc_alt = np.tile(acc_alt, vertical_steps)
    acc_alt[:int(turn_time/dt)] = 0 #np.concatenate((flat_alt_block, acc_alt))

    #Compute Azimuth (az) and Altitude (alt) Coordinates:
    #Computed by integrating acceleration to get velocity, then integrating velocity to get position.
    az_v = np.cumsum(az_acc)*dt-scan_v
    az = np.cumsum(az_v)*dt
    alt_v = np.cumsum(acc_alt)*dt
    alt  = np.cumsum(alt_v)*dt

    flag = np.where(az_acc==0,1,0) #constant scan speed part

    #t = np.arange(len(alt)*dt,dt)
    #v = np.vstack((az_v,alt_v)).T

    return az,alt,flag #,v

def genLocalPath_gittering(az_size = 1, vertical_steps=2, alt_step=0.02, acc = 0.05, scan_v=0.05, dt= 0.01, N = 10):

    #Compute Time for Scan and Turns
    scan_time = az_size/scan_v #Time required to cover the full azimuth range at scan_v
    turn_time = 2*scan_v/acc #Time required to perform a turn (deceleration, reversal, acceleration).

    #Generate Azimuth Acceleration Pattern (az_acc):
    #The motion consists of acceleration, constant velocity, and deceleration, forming a symmetric back-and-forth oscillation in azimuth.
    az_acc = np.concatenate((np.ones(int(turn_time/dt))*acc,np.zeros(int(scan_time/dt))))
    az_acc = np.concatenate((az_acc,-1*az_acc))
    #The sequence is repeated for each altitude step (vertical_steps times).

    #Compute the Altitude Acceleration
    acc_alt_value =  alt_step/(turn_time/2)**2   

    #Generate Altitude Acceleration Pattern (acc_alt)
    #The altitude changes slightly during turns, using a small acceleration.
    oscillation = np.tile(
        np.concatenate([
            np.ones(int(turn_time / dt / 2)) * acc_alt_value,
            np.ones(int(turn_time / dt / 2)) * -acc_alt_value
        ]), 1
    )

    # Ensure no extra oscillation at the ends of azimuth scan
    acc_alt = np.concatenate((oscillation, np.zeros(int(scan_time / dt))))
    #acc_alt = np.concatenate((acc_alt, -1 * acc_alt))  # Repeat for downward scan
    acc_alt = np.concatenate((acc_alt, np.zeros_like(acc_alt)))  # No altitude change on the leftward scan

    #------------------------------------------------------------
    #------------------------------------------------------------
    # Add N extra left-right scans without altitude step
    #------------------------------------------------------------


    # Define one full azimuth oscillation (right + left)
    extra_az_acc = np.concatenate((
        np.ones(int(turn_time/dt))*acc,
        np.zeros(int(scan_time/dt))
    ))
    extra_az_acc = np.concatenate((extra_az_acc, -1*extra_az_acc))

    # Repeat this pattern N times
    extra_az_acc = np.tile(extra_az_acc, N)

    # Extend az_acc with these last flat scans
    az_acc = np.concatenate((az_acc, extra_az_acc))

    # For the extra scans, altitude stays constant (flat)
    extra_acc_alt = np.zeros_like(extra_az_acc)

    # Extend altitude acceleration array
    acc_alt = np.concatenate((acc_alt, extra_acc_alt))

    az_acc = np.tile(az_acc,vertical_steps)
    acc_alt = np.tile(acc_alt, vertical_steps)
    acc_alt[:int(turn_time/dt)] = 0 #np.concatenate((flat_alt_block, acc_alt))

    #------------------------------------------------------------

    #Compute Azimuth (az) and Altitude (alt) Coordinates:
    #Computed by integrating acceleration to get velocity, then integrating velocity to get position.
    az_v = np.cumsum(az_acc)*dt-scan_v
    az = np.cumsum(az_v)*dt
    alt_v = np.cumsum(acc_alt)*dt
    alt  = np.cumsum(alt_v)*dt

    flag = np.where(az_acc==0,1,0) #constant scan speed part

    return az,alt,flag #,v

def genLocalPath(az_size = 1, alt_size = 1, alt_step=0.02, acc = 0.05, scan_v=0.05, dt= 0.01):
    """
    Function that generates the local scanning pattern.
    This function a generate closed loop, that steps in elevation at every turn around.
    Then, it come back to the starting point by stepping down in elevation ar    Parameters
    ----------
    az_size: float
        azimuth angular size, in degrees   
    alt_size: float
        altitude angular size, in degrees
    alt_step: float
        step in altitude angle, in degrees
    acc: float
        acceleration in second angle 
    scan_v: float
        angular speed of the scan, in deg/sec
    dt: float
        time step in second angle 
    Returns
    -------
    az: array
        azimuth scan path coordinates, in degrees
    alt: array
        altitude scan path coordinates, in degrees
    flag: array
        constant scan speed part.
    scan_eff: array
        scan efficiency: the ratio between the constant scan speed part and not constant scan speed part
    t: array
        time during the scan, in second angle
    """ 
    #----
    #Compute Number of Vertical Steps 
    vertical_steps = int(alt_size//alt_step)

    #Compute Time for Scan and Turns
    scan_time = az_size/scan_v #Time required to cover the full azimuth range at scan_v
    turn_time = 2*scan_v/acc #Time required to perform a turn (deceleration, reversal, acceleration).

    #Generate Azimuth Acceleration Pattern (az_acc):
    #The motion consists of acceleration, constant velocity, and deceleration, forming a symmetric back-and-forth oscillation in azimuth.
    az_acc = np.concatenate((np.ones(int(turn_time/dt))*acc,np.zeros(int(scan_time/dt))))
    az_acc = np.concatenate((az_acc,-1*az_acc))
    #The sequence is repeated for each altitude step (vertical_steps times).
    az_acc = np.tile(az_acc,vertical_steps)

    #Compute the Altitude Acceleration
    acc_alt_value =  alt_step/(turn_time/2)**2   

    #Generate Altitude Acceleration Pattern (acc_alt)
    #The altitude changes slightly during turns, using a small acceleration.
    acc_alt = np.concatenate((np.ones(int(turn_time/dt/2))*acc_alt_value,-1*np.ones(int(turn_time/dt/2))*acc_alt_value,np.zeros(int(scan_time/dt))))
    acc_alt = np.concatenate((np.tile(acc_alt,vertical_steps),np.tile(-1*acc_alt,vertical_steps)))

    flag = np.where(az_acc==0,1,0) #constant scan speed part

    #Compute Azimuth (az) and Altitude (alt) Coordinates:
    #Computed by integrating acceleration to get velocity, then integrating velocity to get position.
    az_v = np.cumsum(az_acc)*dt-scan_v
    az = np.cumsum(az_v)*dt

    alt_v = np.cumsum(acc_alt)*dt
    alt  = np.cumsum(alt_v)*dt

    return az,alt,flag  

def genScanPath(T, dt, alt, az, flag ):

    N_total = len(T)
    # Number of points in one pattern
    N_pattern = len(az)
    N_repeat = N_total // N_pattern
    N_remainder = N_total % N_pattern

    # Repeat az-alt pattern
    az_full = np.tile(az, N_repeat)
    alt_full = np.tile(alt, N_repeat)
    flag_full = np.tile(flag, N_repeat)

    # Add partial repetition if needed
    if N_remainder > 0:
        az_full = np.concatenate((az_full, az[:N_remainder]))
        alt_full = np.concatenate((alt_full, alt[:N_remainder]))
        flag_full = np.concatenate((flag_full, flag[:N_remainder]))

    print(f"One pattern: {N_pattern*dt*60:.2f} min, repeats: {N_repeat}, leftover: {N_remainder*dt*60:.2f} min")

    coor = np.vstack((az_full,alt_full)).T

    coor[:,0] -= np.mean(az)
    coor[:,1] -= np.mean(alt)
    N_total = len(az_full)
    # Time array centered on 0

    return coor, flag_full

def genScanPath_original(T, alt, az, flag, plot=False):
    """    
    Function that generates the pointing coordinates vs time.

    Parameters
    ----------
    T: array
        time stream
    az: array
        azimuth scan path coordinates, in degrees
    alt: array
        altitude scan path coordinates, in degrees
    flag: array
        constant scan speed part. 
    Returns
    -------
    coor: 2d array
        coordinates in degrees
    flag: array
        constant scan speed part. 
    """ 

    coor = np.zeros((len(T),2))

    idx = np.int_(np.fmod(T,len(alt)/100)*100)
    
    coor[:,0] = az[idx]-np.mean(az)
    coor[:,1] = alt[idx]-np.mean(alt)
    flag      = flag[idx]
    
    return coor,flag

def pixelOffset(pixel_num, pixel_pitch, pixel_array_separation):
    """
    Function that  gernerates the pixel offset vs pointing center
    Parameters
    ----------
    pixel_num: int
        number of spatial pixels
    pixel_pitch: float
        spatial distance between adjacent pixels in degrees
    Returns
    -------
    yoffsets: array
        the pixel offset vs pointing center, in degrees
    """ 
    yoffsets = (np.arange(0,pixel_num)-pixel_num/2)*pixel_pitch
#     offsets = np.vstack((np.zeros(pixel_num),yoffsets)).T
    xoffsets = np.ones(len(yoffsets)) * pixel_array_separation
    
    return yoffsets, xoffsets

def pixels_rotations(pixel_offset, pixel_shift, theta):
    """
    Function that gernerates the pointing time stream for each pixel
    Parameters
    ----------
    pointing_path: 2d array
        coordinates timestream of the pointing
    pixel_offset: float
        spatial distance between adjacent pixels in degrees
    theta: float
        angle in degree
    Returns
    -------
    pixel_path: nd array
        the coordinates timestream of the pointing of each pixel, in degrees
    """ 
    rotated_pixel = []
    for pixel, xpixel in zip(pixel_offset, pixel_shift):  
        pixel_w_time = np.array([xpixel * np.cos(theta) - pixel * np.sin(theta), 
                                 xpixel * np.sin(theta) + pixel * np.cos(theta)])  # Apply rotation
        #pixel_w_time = np.append( pixel*np.sin(theta), pixel*np.cos(theta))
        rotated_pixel.append(pixel_w_time) 
    return np.asarray(rotated_pixel)

def genPointingPath(T, scan_path, HA, lat, dec, offsets = np.zeros(2), azel=False):

    """
    Function that takes local paths and generates the pointing on sky vs time.
    Parameters
    ----------
    T: array
        coordinates timestream of the pointing
    pixel_offset: float
        spatial distance between adjacent pixels in degrees
    Returns
    -------
    pixel_path: nd array
        the coordinates timestream of the pointing of each pixel, in degrees
    """     

    #-------------------------------------------------------------------
    alt = elevationAngle(dec,lat,HA)+np.radians(scan_path[:,1]) #Check
    azi = azimuthAngle(dec,lat,HA)+np.radians(scan_path[:,0])   

    x_el = azi * np.cos(alt) - np.radians(offsets[0])
    azi = x_el / np.cos(alt)
    
    alt += np.radians(offsets[1])

    dec_point = declinationAngle(np.degrees(azi), np.degrees(alt), lat)
    ha_point  = hourAngle(       np.degrees(azi), np.degrees(alt), lat)

    ra = (HA*np.pi/12-ha_point)
    ra_unwrapped = np.unwrap(ra) #( ra + np.pi) % (2 * np.pi) - np.pi

    path = np.vstack((np.degrees(ra_unwrapped),np.degrees(dec_point))).T
    azel_path = np.vstack((np.degrees(azi),np.degrees(alt))).T
    #-------------------------------------------------------------------
    if(azel): return path, azel_path
    else: return path

def binMap(pointing_paths, res=0.02,  dec=0, ra=0, shape=None):
    
    """
    Binning the pointing into 2d array
    Parameters
    ----------
    pointing_paths: 2d array
        coordinates timestream of the pointing
    res: float
        spatial resolution of the map
    Returns
    -------
    xedges: array
        the x edges of the binned hitmap
    yedges: array
        the y edges of the binned hitmap
    hit_map: 2d array
        2d histogram of hit on the sky

    """ 
    x_res = res
    y_res = x_res

    if(len(np.asarray(pointing_paths).shape)>2 ):
        pointings = np.concatenate([pixel for pixel in pointing_paths])
    else: pointings = pointing_paths

    if(shape is None):

        x_range =  np.max((np.abs(pointings[:,0].max() - pointings[:,0].min()), np.abs(pointings[:,1].max() - pointings[:,1].min())))
        y_range =  x_range

        xedges = ra+np.arange(-x_range/2, x_range/2+x_res, x_res)
        yedges = dec+np.arange(-y_range/2, y_range/2+y_res, y_res)

    else:
        xedges = ra+(np.arange(-shape[1]/2, shape[1]/2 + 1)) * x_res
        yedges = dec+(np.arange(-shape[0]/2, shape[0]/2 + 1)) * y_res

    hit_map = binning(xedges,yedges, pointings)

    return xedges,yedges,hit_map

def binning(xedges,yedges,pointings):
    """
    Binning the pointing into 2d array
    Parameters
    ----------
    xedges: array
        the x edges of the binned hitmap
    yedges: array
        the y edges of the binned hitmap
    pointing: 2d array
        coordinates timestream of the pointing
    Returns
    -------
    H: 2d array
        2d histogram of hit on the sky
    """ 

    H, xedges, yedges = np.histogram2d(pointings[:,0], pointings[:,1], bins=(xedges, yedges))
    return H.T

def noise_map(hitmap, nei, dt):

    """
    Generate a white noise map in Jy.sr**-1
    Parameters
    ----------
    hitmap: 2d array
        the hitmap from the scan strategy
    nei: float
        the net equivalent intensiy in Jy.sr**-1.s**-1/2
    Returns
    -------
    noise_map: 2d array
        the white map in Jy.sr**-1
    """ 

    t_pix = hitmap * dt

    ny, nx = hitmap.shape
    n_chan = len(nei)

    # Initialize cube
    noise_cube = np.zeros((n_chan, ny, nx, ))

    # Generate noise slice by slice
    '''
    for i, I in enumerate(nei):
        sigma = np.zeros_like(hitmap, dtype=float)
        sigma[hitmap > 0] = I / np.sqrt(t_pix[hitmap > 0])
        sigma[hitmap == 0] = 0
        noise_cube[i,:, :] = np.random.normal(loc=0.0, scale=sigma)
        noise_cube_list.append(noise_cube)
        if(False):
            BS = 8; plt.rc('font', size=BS); plt.rc('axes', titlesize=BS); plt.rc('axes', labelsize=BS)
            fig, axs = plt.subplots(figsize=(4,3), dpi=160,)# sharey=True, sharex=True)
            #---
            img = axs.imshow(noise_cube[i,:,:], origin = 'lower', vmin = 0.1*noise_cube[i,:,:].min(), vmax = 0.1*noise_cube[i,:,:].max(), cmap = 'binary')
            fig.colorbar(img, ax=axs, label='Noise [MJy/sr]')
    '''

    for i, I in enumerate(nei):

        sigma = np.zeros_like(hitmap, dtype=float)
        sigma[hitmap > 0] = I / np.sqrt(t_pix[hitmap > 0])
        sigma[hitmap == 0] = np.nan  # optional: mark empty pixels
        noise = np.random.normal(loc=0.0, scale=sigma)

        fig, axs = plt.subplots(1,3, figsize=(12,4), dpi=150)
        im = axs[0].imshow(noise, origin='lower', cmap='binary')
        #noise[hitmap > 0]  *= 1 / np.sqrt(t_pix[hitmap > 0])
        cbar = fig.colorbar(im, ax=axs[0], orientation='vertical',)
        cbar.set_label('noise [MJy/sr/s^1/2]')  # Adjust the label if needed
        noise  *= 1 / np.sqrt(t_pix)
        noise[hitmap <= 0] = 0

        im = axs[1].imshow(noise, origin='lower', cmap = 'binary', )
        cbar = fig.colorbar(im, ax=axs[1], orientation='vertical',)
        cbar.set_label('noise [MJy/sr]')  # Adjust the label if needed
        im = axs[2].imshow(hitmap, origin='lower')
        cbar = fig.colorbar(im, ax=axs[2], orientation='vertical',)
        cbar.set_label('hit counts')  # Adjust the label if needed
        fig.tight_layout()
        plt.close()
        noise_cube[i,:, :] = noise

    return noise_cube

def gaussian_random_field(k, pk, ny, nx, res, k_cutoff=None, pk_map = None, force = True):


    """
    Create a map of a Gaussian random field, given its angular power spectrum.
    
    Parameters
    ----------
    k: 1d array
        wavenumber array
    pk: 1d array
        power spectrum array
    ny: int
        the size of the y side in pixel of the map to generate
    ny: int
        the size of the x side in pixel of the map to generate
    res: float
        the resolution of the map in rad
    k_cutoff: float
        the maximum multipol to taken into account in the map generation
    cl_map:
        power amplitude map of size (ny,nx). If provided, it is not recomputed.
    force: bool
        set the negative values in the power spectrum map to zero. 
    Returns
    -------
    real_space_map: 2d array
        the generated map in real space.
    pk_map: 2d array
        the angular power spectrum map
    """

    kmap_rad_y = ny*res
    kmap_rad_x = nx*res
    #Generate gaussian amplitudes
    norm = 1/res

    np.random.seed()

    noise = np.random.normal(loc=0, scale=1, size=(ny,nx))
    dmn1  = np.fft.fft2( noise )

    #Interpolate input power spectrum
    if(pk_map is None):

        k_map = give_map_spatial_freq(res, ny, nx)

        if(k_cutoff is not None): kmax = np.minimum(k_cutoff, k.max())
        else: kmax = np.minimum(k.max(), k_map.max())
        
        pk_map = np.zeros(k_map.shape)
        w = np.where((k_map>k.min()) & (k_map<=kmax))
        if(not w[0].any()): print("wrong k range")
        else:
            #Power law spectrum
            print("interpolate")
            f = interpolate.interp1d( k, pk,  kind='linear')
            pk_map[w] = f(k_map[w])
            w1 = np.where( pk_map <= 0)
            if(w1[0].shape[0] != 0 and force): pk_map[w1] = 0
            pk_map = pk_map

    #Fill amn_t
    amn_t = dmn1 * norm * np.sqrt( pk_map )
    
    #Output map
    real_space_map = np.real(np.fft.ifft2( amn_t ))
    
    return real_space_map, pk_map

def give_map_spatial_freq(res, ny, nx):
    """
    Create a map of a Gaussian random field, given its angular power spectrum.
    
    Parameters
    ----------
    res: float
        resolution in radians
    ny: int
        the size of the y side in pixel 
    ny: int
        the size of the x side in pixel 
    Returns
    -------
    map_k: 2d array
        the angular power spectrum map
    """
    lmap_y = ny*res #rad
    lmap_x = nx*res #rad
    map_ky = np.float64(np.zeros((ny, nx)))
    map_kx = np.float64(np.zeros((ny, nx)))
    map_k  = np.float64(np.zeros((ny, nx)))
    for m in range(0,nx):
        if(m <= nx/2): m1 = m
        else: m1 = m - nx
        for n in range(0,ny):
            if(n <= ny/2): n1 = n
            else: n1 = n - ny
            kx = np.float64(m1/lmap_x) 
            ky = np.float64(n1/lmap_y)
            
            map_kx[n,m] = kx
            map_ky[n,m] = ky
            map_k[n,m] = np.float64(np.sqrt( kx**2 + ky**2))

    return map_k

def load_TIM_noise(observed_frequencies = None, HF_noise = 'TIM_SW_loading.tsv', LF_noise = 'TIM_LW_loading.tsv', ):

    """
    Load the Net Equivalent Intensity (NEI) in MJy/sr/s**1/2 use to generate the white noise from the hitmap. 

    Parameters
    ----------
    HF_noise: str
        the file for the high frequency array NEI
    
    LF_noise: str
        the file for the low frequency array NEI
    Returns
    -------
    freqs: 1d array
        the frequency addresses in GHz
    noise: 1d array
        the NEI values in MJy/sr/s**1/2 associated with the frequencies.    
    """ 

    #------------------------------------------------------------------------------------------
    #TIM params
    noise_model_HF = pd.read_csv(HF_noise, sep='\t')
    noise_model_LF = pd.read_csv(LF_noise, sep='\t')

    lambda_HF = noise_model_HF["# Wavelength[um]"]*1e3 #nm
    nu_HF = cst.c/(lambda_HF*1e-9)/1e9 #GHz
    nHF = noise_model_HF["NEI[Jy/sr s^1/2]"]
    lambda_LF = noise_model_LF["# Wavelength[um]"]*1e3 #nm
    nu_LF = cst.c/(lambda_LF*1e-9)/1e9 #GHz
    nLF = noise_model_LF["NEI[Jy/sr s^1/2]"]
    model_frequencies = np.concatenate((nu_LF[::-1], nu_HF[::-1]))
    noise = (np.concatenate((nLF[::-1], nHF[::-1]))*u.Jy/u.sr).to(u.MJy/u.sr).value

    if(observed_frequencies is None): return model_frequencies, noise

    else: 

        # Make Spectrum1D object
        spec = Spectrum1D(spectral_axis=model_frequencies*u.GHz, flux=noise*u.MJy/u.sr)
        # -----------------------------
        # Resample spectrum
        # -----------------------------
        resampler = LinearInterpolatedResampler()
        resampled_spec = resampler(spec, observed_frequencies)

        return resampled_spec.spectral_axis,resampled_spec.flux
    #------------------------------------------------------------------------------------------







