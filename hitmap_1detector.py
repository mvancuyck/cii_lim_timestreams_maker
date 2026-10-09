import argparse
from src.load_params import load_params, format_duration
from src.scan_fcts import *
from src.astrometry_fcts import *
from src.hdf5_fcts import *
from astropy.io import fits 
import os
from astropy.wcs import WCS
from gen_timestreams import gen_tod
from scipy.interpolate import PchipInterpolator

def resampling(X, spf_start, spf_end):

    '''
    Interpolates an array with a sample per frame to a different sample per frame 
    Parameters
    ----------
    X: array
        The arrray to be interpolated. 
    spf_start: int
        the sample per frame of X
    spf_end: int
        the final sample per frame wanted. 
    Returns
    -------
    x: array
        the array with the new sample per frame. 
    '''

    # Compute ratio in DT
    ratio = spf_start/spf_end

    # Create interpolator
    interper = PchipInterpolator(np.arange(0, len(X)), X)

    # New sample points
    new_points = np.arange(0, len(X), ratio)

    # Interpolated values
    x = interper(new_points)

    return x

def main_1det(P):
    """
    """
    #The coordinates of the field
    name=P['name_field']
    c=SkyCoord.from_name(name)
    ra = 0 #center the field to RA=0
    rafield = c.ra.value
    dec = c.dec.value
    #The contour of the field
    if(P['contours'] is not None):
        contours = P['contours']
        x_cen, y_cen = np.mean(contours[:, 1]), np.mean(contours[:, 0])

    #load the observer position
    lat = P['latitude']

    #Plot parameter
    f_range = P['f_range']

    #Load the resolution. 
    #if not in params, load it from the map used to generate the TOD. 
    res = P['res']
    hdr = fits.getheader(P['path']+P['file'])
    res = (hdr['CDELT1'] * u.Unit(hdr['CUNIT1'])).to(u.deg).value
    print(f'res={res*3600:.2f}arcsecs')

    #Angle of the rotation to apply to the detector array. 
    theta = np.radians(P['theta'])

    #Load the scan duration and generate the time coordinates with the desired acquisition rate. 
    T_duration = P['T_duration'] 
    acquisition_frequency = P['acquisition_frequency']  #sample per frame defined here as the acquisition rate in Hz. 
    dt = 1/acquisition_frequency/3600*np.pi/3.14 #Make the timestep non rational to avoid some stripes in the hitmap. 
    spf = np.round(acquisition_frequency).astype(int)

    #---
    #local sideral time
    LST = np.arange(-T_duration/2,T_duration/2,dt) #hours
    T = LST*3600 #s
    pps = np.floor(T).astype(int) #pulse per seconds for Namap
    subsecond_ps = T-pps #subsecond for Namap
    #---

    tod_file=P['output_path']+P['output_name']
    #------------------------------------------------------------------------------------------

    #------------------------------------------------------------------------------------------    
    #Generate the scan path for the center of the arrays.
    if(P['scan']=='loop'):   az, alt, flag = genLocalPath(az_size=P['az_size'], alt_size=P['alt_size'], alt_step=P['alt_step'], acc=P['acc'], scan_v=P['scan_v'], dt=np.round(dt*3600,3))
    if(P['scan']=='raster'): az, alt, flag = genLocalPath_cst_el_scan(az_size=P['az_size'], alt_size=P['alt_size'], alt_step=P['alt_step'], acc=P['acc'], scan_v=P['scan_v'], dt=np.round(dt*3600,3))
    if(P['scan']=='crisscross'): az, alt, flag = genLocalPath_cst_el_scan_crisscross(az_size=P['az_size'], alt_size=P['alt_size'], alt_step=P['alt_step'], acc=P['acc'], scan_v=P['scan_v'], dt=np.round(dt*3600,3))
    if(P['scan']=='gittering'): az, alt, flag = genLocalPath_gittering(az_size=P['az_size'], vertical_steps=P['vertical_steps'], alt_step=P['alt_step'], acc=P['acc'], scan_v=P['scan_v'], dt=np.round(dt*3600,3), N=P['N_scans'])
    
    scan_path, scan_flag = genScanPath(T,dt, alt, az, flag)

    if(P['cut_turnarounds']):
        scan_path = scan_path[scan_flag==1] #Use the scan flag to keep only the constant scan speed part of the pointing. 
        T = T[scan_flag==1]
        LST = LST[scan_flag==1]

    #Generate the pointing on the sky for the center of the arrays
    scan_path_sky, azel = genPointingPath(T, scan_path, LST, lat, dec, azel=True) 
    #----------------------------------------

    #----------------------------------------
    #Plot a scan route
    BS = 10; plt.rc('font', size=BS); plt.rc('axes', titlesize=BS); plt.rc('axes', labelsize=BS)
    fig, axs = plt.subplots(2,2,figsize=(7,7), dpi=160,)# sharey=True, sharex=True)

    axradec, ax, axr, axc = axs[0,0], axs[1,1], axs[0,1], axs[1,0]
    #---
    axc.scatter(scan_path_sky[:,0], scan_path_sky[:,1], s=0.1,c='r')
    axc.set_xlabel('RA [deg]')
    axc.set_ylabel('Dec [deg]')
    axc.set_aspect('auto')
    #---
    axradec.plot(az-az.max()/2,alt-alt.max()/2,'k', )
    axradec.set_aspect('auto')
    axradec.set_xlabel('RA [deg]')
    axradec.set_ylabel('Dec [deg]')
    #---
    hitmap, xedges, yedges = np.histogram2d(scan_path_sky[:,0], scan_path_sky[:,1], bins=int(2/res))
    im = ax.imshow(hitmap.T, origin='lower', cmap='binary',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    cbar = fig.colorbar(im, ax=ax, label='1 detector counts')
    ax.set_xlabel('RA [deg]')
    ax.set_ylabel('Dec [deg]')
    ax.set_aspect('auto')
    if(contours is not None):
        ax.plot(contours[:, 1]-rafield, contours[:, 0], c='g' )
        axradec.plot(contours[:, 1]-rafield, contours[:, 0]-dec, c='g' )
        axc.plot(contours[:, 1]-rafield, contours[:, 0], c='g' )
    #---
    az_unwrapped = (np.radians(azel[:,0]) + np.pi) % (2 * np.pi) - np.pi
    axr.plot(np.degrees(az_unwrapped),azel[:,1],'b')
    axr.set_xlabel('Az [deg]')
    axr.set_ylabel('El [deg]')
    #-----
    if(True): 
        coords = SkyCoord(alt=np.degrees(az_unwrapped)*u.deg, az=azel[:,1]*u.deg, frame='altaz')
        separations = coords[:-1].separation(coords[1:])
        total_length = np.sum(separations)
        title = f'{total_length:.1f} scanned.'
    else: title = ''
    fig.suptitle(title)
    patchs = []
    fig.tight_layout()
    plt.savefig(os.getcwd()+'/plots/'+f"scan_route_1det_{P['scan']}_{format_duration(P['T_duration'])}.png")
    plt.close()
    #----------------------------------------

    #latitude  timestream
    lat = np.ones(len(LST)) * lat
    #----------------------------------------

    #----------------------------------------
    #Save the coordinates timestreams

    if('.hdf5' in tod_file):
        save_scan_path(tod_file, np.array((LST, lat)).T, spf,acquisition_frequency, ('data_lst', 'data_lat'), save=P['format'], compression=P['compression'])
        save_scan_path(tod_file, azel, spf,acquisition_frequency,('data_AZ', 'data_EL'), save=P['format'], compression=P['compression'])
        save_scan_path(tod_file, scan_path_sky, spf, acquisition_frequency,('data_RA', 'data_DEC'), save=P['format'], compression=P['compression'])
        save_scan_path(tod_file, scan_path,     spf, acquisition_frequency,('data_RA_path', 'data_DEC_path'), save=P['format'], compression=P['compression'])
        save_timestamps(tod_file, T, spf, acquisition_frequency,'data_time', save=P['format'], compression=P['compression'])
        save_timestamps(tod_file, pps, spf, acquisition_frequency,'data_pps',save=P['format'], compression=P['compression'])
        save_timestamps(tod_file, subsecond_ps, spf, acquisition_frequency,'data_subsecond_ps',save=P['format'], compression=P['compression'])
        save_timestamps(tod_file, scan_flag, spf, acquisition_frequency,'data_turnaround_flags', save=P['format'], compression=P['compression'])
    else: 
        save_scan_path_dirfile(tod_file, np.array((LST, lat)).T, spf,acquisition_frequency, ('data_lst', 'data_lat'))
        save_scan_path_dirfile(tod_file, azel, spf,acquisition_frequency,('data_AZ', 'data_EL'))
        save_scan_path_dirfile(tod_file, scan_path_sky, spf, acquisition_frequency,('data_RA', 'data_DEC'))
        save_scan_path_dirfile(tod_file, scan_path,     spf, acquisition_frequency,('data_RA_path', 'data_DEC_path'))
        save_timestamps_dirfile(tod_file, T, spf, acquisition_frequency,'data_time')
        save_timestamps_dirfile(tod_file, pps, spf, acquisition_frequency,'data_pps')
        save_timestamps_dirfile(tod_file, subsecond_ps, spf, acquisition_frequency,'data_subsecond_ps')
        save_timestamps_dirfile(tod_file, scan_flag, spf, acquisition_frequency,'data_turnaround_flags')

    #-------------------------------------------

    #----------------------------------------
    #Create the coordinates sampled differently, to test the synchronization with data in Namap.
    acquisition_frequency_prime = P['acquisition_frequency_coords']  #sample per frame defined here as the acquisition rate in Hz. 
    spf_prime = np.round(P['acquisition_frequency_coords'] ).astype(int)

    LST_prime = resampling(LST, acquisition_frequency, acquisition_frequency_prime)
    T_prime = resampling(T, acquisition_frequency, acquisition_frequency_prime)
    pps_prime = np.floor(T_prime).astype(int)
    subsecond_ps_prime = T_prime-pps_prime
    lat_prime = P['latitude'] * np.ones(len(LST_prime))

    az_prime = resampling(azel[:,0], acquisition_frequency, acquisition_frequency_prime)
    el_prime = resampling(azel[:,1], acquisition_frequency, acquisition_frequency_prime)
    azelprime = np.vstack((az_prime,el_prime)).T
    ra_prime = resampling(scan_path_sky[:,0], acquisition_frequency, acquisition_frequency_prime)
    dec_prime = resampling(scan_path_sky[:,1], acquisition_frequency, acquisition_frequency_prime)
    scan_path_sky_prime = np.vstack((ra_prime,dec_prime)).T
    
    scan_flag_prime = resampling(scan_flag, acquisition_frequency, acquisition_frequency_prime)
    if('.hdf5' in tod_file):
        save_scan_path(tod_file, np.array((LST_prime, lat_prime)).T, spf_prime, acquisition_frequency_prime,('lst', 'lat'), save=P['format'],compression=P['compression'])
        save_scan_path(tod_file, azelprime, spf_prime,acquisition_frequency_prime,('AZ', 'EL'), save=P['format'],compression=P['compression'])
        save_scan_path(tod_file, scan_path_sky_prime, spf_prime,acquisition_frequency_prime, ('RA', 'DEC'), save=P['format'],compression=P['compression'])
        save_timestamps(tod_file, T_prime, spf_prime, acquisition_frequency_prime,'coords_time', save=P['format'],compression=P['compression'])
        save_timestamps(tod_file, pps_prime, spf_prime, acquisition_frequency_prime,'coords_pps', save=P['format'],compression=P['compression'])
        save_timestamps(tod_file, subsecond_ps_prime, spf_prime, acquisition_frequency_prime,'coords_subsecond_ps', save=P['format'],compression=P['compression'])
        save_timestamps(tod_file, scan_flag_prime, spf_prime, acquisition_frequency_prime, 'turnaround_flags', save=P['format'],compression=P['compression'])
    else:
        save_scan_path_dirfile(tod_file, np.array((LST_prime, lat_prime)).T, spf_prime, acquisition_frequency_prime,('lst', 'lat'))
        save_scan_path_dirfile(tod_file, azelprime, spf_prime,acquisition_frequency_prime,('AZ', 'EL'))
        save_scan_path_dirfile(tod_file, scan_path_sky_prime, spf_prime,acquisition_frequency_prime, ('RA', 'DEC'))
        save_timestamps_dirfile(tod_file, T_prime, spf_prime, acquisition_frequency_prime,'coords_time')
        save_timestamps_dirfile(tod_file, pps_prime, spf_prime, acquisition_frequency_prime,'coords_pps')
        save_timestamps_dirfile(tod_file, subsecond_ps_prime, spf_prime, acquisition_frequency_prime,'coords_subsecond_ps')
        save_timestamps_dirfile(tod_file, scan_flag_prime, spf_prime, acquisition_frequency_prime, 'turnaround_flags')


if __name__ == "__main__":
    '''
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

    main_1det(P)