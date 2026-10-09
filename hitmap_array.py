import argparse
from src.load_params import load_params, format_duration
from src.scan_fcts import *
from src.astrometry_fcts import *
from src.hdf5_fcts import *
from astropy.io import fits 
import os
import matplotlib.patches as mpatches
from matplotlib.pyplot import cm
from astropy.wcs import WCS
import pickle
import datetime
import time
import astropy.table as tb
from progress.bar import Bar
from gen_timestreams import group_detectors

def main_arrays(P):

    #-----------------------------
    #Initiate the parameters

    #The coordinates of the field
    name=P['name_field']
    c=SkyCoord.from_name(name)
    ra = 0 
    rafield = c.ra.value
    dec = c.dec.value
    #The contour of the field
    contours = P['contours']
    x_cen, y_cen = np.mean(contours[:, 1]), np.mean(contours[:, 0])

    #load the observer position
    lat = P['latitude']

    #Load the resolution. 
    #if not in params, load it from the map used to generate the TOD. 
    res = P['res']
    if(res is None):
        hdr = fits.getheader(P['path']+P['file'])
        res = (hdr['CDELT1'] * u.Unit(hdr['CUNIT1'])).to(u.deg).value

    aquisition_frequency = P['acquisition_frequency']  #sample per frame defined here as the acquisition rate in Hz. 
    dt = 1/aquisition_frequency/3600*np.pi/3.14 #Make the timestep non rational to avoid some stripes in the hitmap. 
    spf = np.round(aquisition_frequency).astype(int)

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
        d.close()

    scan_path = np.asarray((RA_path, DEC_path)).T
    #-----------------------------

    #-----------------------------

    det_names_dict = pd.read_csv(P['detectors_name_file'], sep='\t')
    groups, offset_SW = group_detectors(det_names_dict, 'SW')
    groups, offset_LW = group_detectors(det_names_dict, 'SW')
    pixel_offset_xEL = np.concatenate((np.asarray(offset_SW)[:,1], np.asarray(offset_LW)[:,1]))
    pixel_offset_EL = np.concatenate((np.asarray(offset_SW)[:,0], np.asarray(offset_LW)[:,0]))

    #-------------------------------
    pixel_offsets = pixels_rotations(pixel_offset_EL, pixel_offset_xEL, P['theta'])
    #Generate the pointing on the sky of each pixel. 
    pointing_paths = [genPointingPath(T, scan_path, LST, lat, dec, offsets) for offsets in pixel_offsets]
    #Generate the hitmap, using all the detectors. 
    xedges,yedges,hit_map = binMap(pointing_paths,res=res,dec=dec,ra=ra) 
    #-------------------------------

    #----------------------------------------
    #Plot a scan route
    BS = 8; plt.rc('font', size=BS); plt.rc('axes', titlesize=BS); plt.rc('axes', labelsize=BS)
    fig, axs = plt.subplots(1,3,figsize=(9,3), dpi=160,)# sharey=True, sharex=True)
    #---
    axradec, axp, axpix = axs[0], axs[1], axs[2]
    axradec.plot(RA_path,DEC_path,'b')
    axradec.set_xlabel('RA [deg]')
    axradec.set_ylabel('Dec [deg]')
    #axradec.set_aspect(aspect=1)
    #---
    P['f_range'] = 1.2
    img = axp.imshow((hit_map), extent=[x_cen-P['f_range'], x_cen+P['f_range'],y_cen-P['f_range'], y_cen+P['f_range'],], 
                    interpolation='nearest', origin='lower', vmin=0, vmax=np.max(hit_map), cmap='binary' )
    if(contours is not None): axp.plot(contours[:, 1], contours[:, 0], c= 'g')
    fig.colorbar(img, ax=axp, label='Counts')
    #CS = axp.contour(np.roll(np.sqrt(hit_map*np.roll(hit_map, 4, axis=1)),-2,axis=1), levels=[0.5*np.max(hit_map)], lw=0.5, origin='lower', linewidths=0.6,
    #extent=[x_cen-f_range, x_cen+f_range,y_cen-f_range, y_cen+f_range,], colors='magenta')
    #axp.clabel(CS, inline=1, fontsize=8, )
    axp.set_xlabel('RA [deg]')
    axp.set_ylabel('Dec [deg]')
    #---
    patchs = []
    axpix.set_title('Pointing Path per pixel')
    idx = np.arange(len(pointing_paths))[::15]
    n = 11
    for i,c in zip(idx, cm.rainbow(np.linspace(0.,1,len(idx)))):
        axpix.scatter(pointing_paths[i][::n,0], pointing_paths[i][::n,1], s=0.1,c=c)
        patch = mpatches.Patch(color=c, label= 'pixel %d'%i); patchs.append(patch); 
    #axpix.set_aspect(aspect=1)
    axpix.legend(handles=patchs,frameon=False, bbox_to_anchor=(1,1))
    axpix.set_xlabel('RA [deg]')
    axpix.set_ylabel('Dec [deg]')
    fig.tight_layout()
    plt.savefig(os.getcwd()+'/plots/'+f"scan_route_{P['scan']}_{format_duration(P['T_duration'])}_for_array.png")
    plt.show()
    #----------------------------------------

    #-------------------------------
    wcs = WCS(naxis=2)
    #Define pixel-to-world transformation 
    wcs.wcs.crpix = [hit_map.shape[1] // 2, hit_map.shape[0] // 2]      # Reference pixel
    wcs.wcs.cdelt = [res, res] # Pixel scale in degrees/pixel (RA, Dec)
    wcs.wcs.crval = [ra, dec]          # Reference coordinates (RA, Dec)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]  # Projection type
    d = {'wcs':wcs}; pickle.dump(d, open(P['wcs_dict'], 'wb'))

    f = fits.PrimaryHDU(hit_map, header=wcs.to_header())
    hdu = fits.HDUList([f])
    hdr = hdu[0].header
    hdr.set("map")
    hdr.set("Datas")
    hdr["BITPIX"] = ("64", "array data type")
    hdr["BUNIT"] = 'counts'
    hdr["DATE"] = (str(datetime.datetime.now()), "date of creation")
    hdu.writeto( f'datasets/hit_map_array_{P["az_size"]:.1f}_{P["alt_step"]:.1f}deg2_{format_duration(P["T_duration"])}.fits', overwrite=True)
    print('save ',f'datasets/hit_map_array_{P["az_size"]:.1f}_{P["alt_step"]:.1f}deg2_{format_duration(P["T_duration"])}.fits')
    hdu.close()
    #-------------------------------

    #-------------------------------
    if(False): 
        start = time.time()
        #Load the names of the detectors.
        det_names_dict = pd.read_csv(P['detectors_name_file'], sep='\t')
        same_offset_groups = det_names_dict.groupby(['XEL', 'EL'])['Name'].apply(list).reset_index()
            
        bar = Bar(f'Save the detector poiting paths', max=len(same_offset_groups))
        for i, (pointing_path, offset, shift) in enumerate(zip(pointing_paths, pixel_offset, pixel_shift )):
            names = same_offset_groups.iloc[i]['Name']
            for name in names: 
                save_scan_path(tod_file, pointing_path, spf, (f'RA_{name}', f'DEC_{name}'))
            bar.next()
        bar.finish
        print('')
        print(f"Saved the detector pointing paths in {np.round((time.time() - start),2)}"+'s')
    #-------------------------------

if __name__ == "__main__":

    '''
    Description
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
    print(f'running arrays detector {args.params}')
    #------------------------------------------------------------------------------------------

    main_arrays(P)