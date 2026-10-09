import h5py
import pandas as pd
import os
import numpy as np
from IPython import embed
import pygetdata as gd

def to8bit_intprecision(array): 

    float_array = np.array(array, dtype=np.float32)
    # Assuming the original float range is 0.0-1.0

    min = float_array.min()
    if(min<0): float_array -= min

    max = np.abs(float_array).max()
    float_array /= max

    # 1. Scale the values to the 0-255 range
    scaled_array = float_array * 255

    # 2. Clip the values to ensure they are within the 8-bit range (0-255)
    clipped_array = np.clip(scaled_array, 0, 255)

    # 3. Convert to unsigned 8-bit integer type
    downsampled_array = clipped_array.astype(np.uint8)

    return downsampled_array, np.float16(min), np.float16(max)

def save_scan_path(tod_file, scan_path, spf, acquisition_frequency, keys,save="64bytes", compression='gzip'):
    """
    Save the scan path in the .hdf5 format. 

    Parameters
    ----------
    tod_file: string 
        name of the output hdf5 file  
    scan_path_sky: 2d array
        (ra, dec) coordinates timestreams of the center pixel
    spf: int
        the number of samples per frame
    keys: list
        list of names under which the two coordinates are saved
    
    Returns
    -------
    """ 
    H = h5py.File(tod_file, "a")
    for i, (name, coord) in enumerate(zip(keys, (scan_path[:,0],scan_path[:,1]))):

        if name in H: del H[name]   # WARNING: deletes entire detector group
        grp = H.create_group(name)

        if(save=="32bytes"): coord = (coord).astype(np.float32)
        elif(save=="16bytes"): coord = (coord).astype(np.float16)

        if(compression is not None): grp.create_dataset('data', data=coord, compression='gzip', compression_opts=9)
        else: grp.create_dataset('data', data=coord, compression=compression)
        grp.create_dataset('spf', data=spf)
        grp.create_dataset('acquisition frequency', data=acquisition_frequency)
    H.close() 

def save_timestamps(tod_file, T, spf, acquisition_frequency, key, compression='gzip',save="64bytes"):
    '''
    Save the time tod in the .hdf5 format. 

    Parameters
    ----------
    tod_file: string 
        name of the output hdf5 file  
    T: array
        time timestreams
    spf: int
        the number of samples per frame
    key: string
        name under which the timestamps are saved
    Returns
    -------
    '''
    H = h5py.File(tod_file, "a")

    if key in H: del H[key]   # WARNING: deletes entire detector group
    grp = H.create_group(key)

    if(save=="32bytes"): T = (T).astype(np.float32)
    elif(save=="16bytes"): T = (T).astype(np.float16)
    if(compression is not None): grp.create_dataset('data', data=T, compression=compression, compression_opts=9)
    else: grp.create_dataset('data', data=T, compression=compression)

    grp.create_dataset('spf', data=spf)
    grp.create_dataset('acquisition frequency', data=acquisition_frequency)

    H.close()

def save_tod_in_hdf5(tod_file, det_names, samples, pixel_offset, pixel_shift, dect_file, F, spf, acquisition_frequency, pointing_paths_to_save, suffix='', compression='gzip',save="64bytes"):
    """
    Save the tod for one array of TIM detectors in the .hdf5 format. 

    Parameters
    ----------
    tod_file: string 
        name of the output hdf5 file   
    det_names: list
        list of names for the detectors, same lenght as pixel_offset
    samples: list
        list of amplitude timestreams.  
    pixel_offset: array
        vertical position of each pixel on the array with respect to the center 
    pixel_shift array
        horizontal position of each pixel on the array with respect to the center 
    dect_file: string
        the name of the .csv where the info on each pixel is stored. This function add the frequency band info to the .csv file. 
    F: float
        the frequency band seen by the detectors [GHz]
    spf: int
        the number of samples per frame
    acquisition_frequency: float
        the acquisition frequency of the detectors [Hz]

    Returns
    -------
    """ 
    
    H = h5py.File(tod_file, "a")

    for detector, (offset, shift, name) in enumerate(zip(pixel_offset, pixel_shift, det_names)):
            
        namegrp = f'kid_{name+suffix}_roach'
        if namegrp in H: del H[namegrp]   # WARNING: deletes entire detector group
        grp = H.create_group(namegrp)

        grp.create_dataset('spf', data=np.float16(spf))
        grp.create_dataset('pixel_offset_y', data=np.float16(offset))
        grp.create_dataset('pixel_offset_x', data=np.float16(shift))
        grp.create_dataset('acquisition frequency', data=np.float16(acquisition_frequency))

        scan_path = pointing_paths_to_save[detector]
        for i, (name_coord, coord) in enumerate(zip(('RA_roach','DEC_roach'), (scan_path[:,0],scan_path[:,1]))):
            if(compression is not None): 
                    grp.create_dataset(name_coord, data=coord, compression=compression, compression_opts=9)
            else: grp.create_dataset(name_coord, data=coord, compression=compression)
        
        if(save=="32bytes"):
            sample = (samples[detector,:]).astype(np.float32)
            if(compression is not None): 
                grp.create_dataset('data', data=sample, compression=compression, compression_opts=9)
                grp.create_dataset('data_Q', data=sample, compression=compression, compression_opts=9)
            else: 
                grp.create_dataset('data', data=sample, compression=compression)
                grp.create_dataset('data_Q', data=sample, compression=compression)
            

        elif(save=="16bytes"):

            sample = (samples[detector,:]).astype(np.float16)
            if(compression is not None): 
                grp.create_dataset('data', data=sample, compression=compression, compression_opts=9)
                grp.create_dataset('data_Q', data=sample, compression=compression,compression_opts=9)
            else: 
                grp.create_dataset('data', data=sample, compression=compression)
                grp.create_dataset('data_Q', data=sample, compression=compression)

        else: 
            sample = samples[detector,:]
            if(compression is not None): 
                grp.create_dataset('data', data=sample, compression=compression, compression_opts=9)
                grp.create_dataset('data_Q', data=sample, compression=compression,compression_opts=9)
            else: 
                grp.create_dataset('data', data=sample, compression=compression)
                grp.create_dataset('data_Q', data=sample, compression=compression)

    H.close()

    if( not os.path.isfile(dect_file) ):

        with open(dect_file, 'w') as f:
            f.write("Name\tEL\tXEL\tFrequency\n")  # Column headers
            for detector, (offset, shift, name) in enumerate(zip(pixel_offset, pixel_shift, det_names)):

                f.write(f"{name}\t{offset}\t{shift}\t\t\t\t\n")  # Tab-separated values
    #---------------------------
    if(True):

        # Load detectors file   
        det_names_dict = pd.read_csv(dect_file, sep="\t")

        # Ensure Frequency column exists and is numeric
        det_names_dict["Frequency"] = pd.to_numeric(det_names_dict["Frequency"], errors="coerce")

        # Build mask
        mask = det_names_dict["Name"].isin(det_names)

        # Assign F to all matching detectors
        det_names_dict.loc[mask, "Frequency"] = float(F)

        # Write back TSV (same columns, same order)
        det_names_dict.to_csv(dect_file, sep="\t", index=False)

    return 0

def save_tods_dirfile(tod_file, det_names, samples, dect_file, EM_freq, spf, acquisition_frequency, pixel_offset,pixel_shift, suffix=''):

    '''
    Save the timestreams in a dirfile
    Parameters
    ----------
    Returns
    -------
    '''

    df = gd.dirfile(tod_file, gd.RDWR | gd.CREAT )
    for d, kid in zip(samples, det_names):

        field_name = f"kid_{kid+suffix}_roach"
        entry = gd.entry(gd.RAW_ENTRY, field_name, 0, parameters={"type": gd.FLOAT64, "spf": int(spf)})
        try:
            df.add(entry)
        except gd.DuplicateError:
            df.delete(field_name)   # remove existing field
            df.add(entry)           # recreate it
        df.putdata(field_name, d.tolist())
    df.close()
    
    if(True):

        if( not os.path.isfile(dect_file) ):
            with open(dect_file, 'w') as f:
                f.write("Name\tEL\tXEL\tFrequency\n")  # Column headers
                for detector, (offset, shift, name) in enumerate(zip(pixel_offset, pixel_shift, det_names)):
                    f.write(f"{name}\t{offset}\t{shift}\t\t\t\t\n")  # Tab-separated values
        # Load detectors file   
        det_names_dict = pd.read_csv(dect_file, sep="\t")
        # Ensure Frequency column exists and is numeric
        det_names_dict["Frequency"] = pd.to_numeric(det_names_dict["Frequency"], errors="coerce")
        # Build mask
        mask = det_names_dict["Name"].isin(det_names)
        # Assign F to all matching detectors
        det_names_dict.loc[mask, "Frequency"] = float(EM_freq)
        # Write back TSV (same columns, same order)
        det_names_dict.to_csv(dect_file, sep="\t", index=False)

    return 0

def save_scan_path_dirfile(tod_file, scan_path, spf, acquisition_frequency, keys, compression='gzip'):
    """
    Save the scan path in the .hdf5 format. 

    Parameters
    ----------
    tod_file: string 
        name of the output hdf5 file  
    scan_path_sky: 2d array
        (ra, dec) coordinates timestreams of the center pixel
    spf: int
        the number of samples per frame
    keys: list
        list of names under which the two coordinates are saved
    
    Returns
    -------
    """ 

    df = gd.dirfile(tod_file, gd.RDWR | gd.CREAT )

    for i, (field_name, coord) in enumerate(zip(keys, (scan_path[:,0],scan_path[:,1]))):

        entry = gd.entry(gd.RAW_ENTRY, field_name, 0, parameters={"type": gd.FLOAT64, "spf": int(spf)})

        try:
            df.add(entry)
        except gd.DuplicateError:
            df.delete(field_name)   # remove existing field
            df.add(entry)           # recreate it

        df.putdata(field_name, coord.tolist())
    df.close()

def save_timestamps_dirfile(tod_file, T, spf, acquisition_frequency, key):
    '''
    Save the time tod in the .hdf5 format. 

    Parameters
    ----------
    tod_file: string 
        name of the output hdf5 file  
    T: array
        time timestreams
    spf: int
        the number of samples per frame
    key: string
        name under which the timestamps are saved
    Returns
    -------
    '''


    df = gd.dirfile(tod_file, gd.RDWR | gd.CREAT )

    entry = gd.entry(gd.RAW_ENTRY, key, 0, parameters={"type": gd.FLOAT64, "spf": int(spf)})

    try:
        df.add(entry)
    except gd.DuplicateError:
        df.delete(key)   # remove existing field
        df.add(entry)           # recreate it

    T = np.asarray(T, dtype=np.float64)
    df.putdata(key, T.tolist())
    df.close()

        
    return 0