import random
from src.scan_fcts import pixelOffset
from src.load_params import load_params
import argparse
import numpy as np
import os
from IPython import embed
random.seed(42)   # Fixes the seed for Python's random module
np.random.seed(42)  # Fixes the seed for NumPy


def generate_strings(total_count, groups):
    """
    Generate unique detector names in the format A_XX_NNNN.
    XX for the sub-array, and is set by groups
    NNNN for the detector, between 0000 and 9999
    
    Parameters
    ----------
    total_count:int
        Total number of detector names to generate.  
    group: list
        the index refering to each sub-array.   
    Returns
    -------
    list: list
        List of unique detector names.
    """

    n_groups = len(groups)

    # Distribute as evenly as possible
    base = total_count // n_groups
    remainder = total_count % n_groups

    # Example for total_count=3, groups=['01','02','03','04']:
    # base = 0, remainder = 3
    # distribution = [1,1,1,0]

    counts = [base + (1 if i < remainder else 0) for i in range(n_groups)]

    all_strings = []

    for group, n_per_group in zip(groups, counts):
        strings = set()
        while len(strings) < n_per_group:
            rand_digits = f"{random.randint(0, 9999):04d}"
            strings.add(f"A_{group}_{rand_digits}")
        
        all_strings.extend(sorted(strings))

    return all_strings

def gen_detectors_main(P):

    #---------------------------    
    #Generate the offset of the pixels with respect to the center of the two arrays, in degrees. 
    pixel_offset_EL_SW, pixel_offset_xEL_SW = pixelOffset(P['nb_pixel_SW'], P['offset_SW'], -P['arrays_separation']/2)
    pixel_offset_EL_LW, pixel_offset_xEL_LW = pixelOffset(P['nb_pixel_LW'], P['offset_LW'], P['arrays_separation']/2) 
    #pixel_offset_xEL_LW[::2] +=  P['arrays_separation']*2
    in_array_LW = ['LW'] * len(pixel_offset_EL_LW)
    in_array_SW = ['SW'] * len(pixel_offset_EL_SW)

    pixel_offset_EL_SW_tot = np.tile(pixel_offset_EL_SW, P['nb_channels_per_array'])
    pixel_offset_EL_LW_tot = np.tile(pixel_offset_EL_LW, P['nb_channels_per_array'])
    pixel_offset_EL = np.concatenate((pixel_offset_EL_SW_tot, pixel_offset_EL_LW_tot))

    pixel_offset_xEL_SW_tot = np.tile(pixel_offset_xEL_SW, P['nb_channels_per_array'])
    pixel_offset_xEL_LW_tot = np.tile(pixel_offset_xEL_LW, P['nb_channels_per_array'])
    pixel_offset_xEL = np.concatenate((pixel_offset_xEL_SW_tot, pixel_offset_xEL_LW_tot))

    in_array_LW_tot = np.tile(in_array_LW, P['nb_channels_per_array']) 
    in_array_SW_tot = np.tile(in_array_SW, P['nb_channels_per_array']) 
    in_array_tot = np.concatenate((in_array_SW_tot, in_array_LW_tot))
    #---------------------------    

    #---------------------------
    # Generate detector names
    det_names_SW = generate_strings(P['nb_pixel_SW'] * P['nb_channels_per_array'], ['01', '02', '03', '04'])
    det_names_LW = generate_strings(P['nb_pixel_LW'] * P['nb_channels_per_array'], ['05', '06', '07', '08'])
    det_names = np.concatenate((det_names_SW,det_names_LW))
    #---------------------------

    #---------------------------    
    #The response of the detectors
    resp = np.ones(len(det_names))
    #The noise of the detectors
    noise = np.zeros(len(det_names))
    #Time delay between pointing solution and detector data
    time_offset = np.zeros(len(det_names)) 
    #Offsets
    EL = pixel_offset_EL
    XEL = pixel_offset_xEL
    #EM Frequency band
    freqs = np.zeros(len(det_names))
    #---------------------------

    # Ensure pixel_offset_SW has the same length as det_names
    if len(resp) < len(det_names) or len(noise) < len(det_names) or  len(time_offset) < len(det_names):
        raise ValueError("Not enough pixel offsets for detector names.")

    #---------------------------
    # Save as a .tsv file
    # Define the output file path
    file = P['detectors_name_file']    
    with open(file, 'w') as f:
        f.write("Name\tResp.\tWhiteNoise\ttime offset\tEL\tXEL\tArray\tFrequency\n")  # Column headers
        for name,  r,n,t, e, xe,F, arr in zip(det_names, resp, noise,time_offset, EL, XEL, freqs, in_array_tot):
            f.write(f"{name}\t{r:1f}\t{n:1f}\t{t:1f}\t{e:1f}\t{xe:1f}\t{arr}\t{F:1f}\n")  # Tab-separated values
    #---------------------------

    #---------------------------
    # Show first 5 rows
    import pandas as pd
    df = pd.read_csv(file, sep='\t') #os.getcwd()+'/'+
    print(df.head())
    #---------------------------


if __name__ == "__main__":
    '''
    Run python gen_det_names.py params_strategy.par
    Generate a .cvs file containing the info of the detectors. 
    '''

    random.seed(42)   # Fixes the seed for Python's random module
    np.random.seed(42)  # Fixes the seed for NumPy

    #---------------------------
    #Iinitialization
    parser = argparse.ArgumentParser(description='Gen a random file of names for TIM kids', \
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    #options
    parser.add_argument('params', help=".par file with params", default = None)
    args = parser.parse_args()

    P = load_params(args.params)
    #---------------------------

    gen_detectors_main(P)