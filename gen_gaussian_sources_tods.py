import astropy.units as u
import scipy.constants as cst
from astropy.stats import gaussian_fwhm_to_sigma
import astropy.convolution as conv
import datetime
import matplotlib.pyplot as plt
from hitmap_1detector import main_1det
from hitmap_array import main_arrays
from gen_timestreams import main_tod, gen_tod
from src.load_params import load_params, format_duration
from src.hdf5_fcts import save_tod_in_hdf5
from IPython import embed
import numpy as np 
import os
import pickle
from src.scan_fcts import *
from gen_detectors_arrays import gen_detectors_main
import h5py 
import shutil
from astropy.io import fits
from astropy.wcs import WCS

if __name__ == "__main__":

    P = load_params('PAR_files/params_strategy.par')    

    P['detectors_name_file'] = 'TIM_kid_table_reduced.tsv'
    P['nb_pixel_SW'] = 20 #Number of pixel per frequency band in the SW array.
    P['nb_pixel_LW'] = 20 #Number of pixel per frequency band in the LW array.

    if( not os.path.isfile(P['detectors_name_file']) ): gen_detectors_main(P)

    for factor_on_Sigma in (2,1,5): 

        P['T_duration'] = 1

        for separation_in_arcsecs in (  150.8, 90.5, 301.5,30.2, 45.2, 60.3, 30.2, 45.2, 75.4, 60.3, 211.1,): 

            P['output_path'] = f'datasets/'
            P['path'] = f'datasets/'
            P['output_name'] = f'TOD_on_2_sources_separated_by_{separation_in_arcsecs}arcsecs_with_{factor_on_Sigma}xbigger_sigma_PSF.hdf5' 
            P['file'] = f"cube_2sources_separated_by_{separation_in_arcsecs}arcsecs_with_{factor_on_Sigma}xbigger_sigma_PSF.fits" 
            P['scan']='gittering'
            P['az_size'] = 0.3
            P['acquisition_frequency'] = 100
            P['acquisition_frequency_coords'] = 100
            P['scan']='raster' #'loop', 'raster', 'crisscross', 'gittering'
            P['alt_step']= 40/3600*1/3
            P['vertical_steps'] = 6
            P['N_scans'] = 10

            if( not os.path.isfile(P['output_path']+P['output_name']) ):
                main_1det(P)
                #main_arrays(P)
                main_tod(P)


