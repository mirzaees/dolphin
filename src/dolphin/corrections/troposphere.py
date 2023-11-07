#!/usr/bin/env python3
from __future__ import annotations

import sys
import argparse
import journal


import pyaps3 as pa
from dolphin._log import get_log, log_runtime
from dolphin.corrections.cutils import *
logger = get_log(__name__)


EXAMPLE="""
troposphere.py -f ../interferograms/stitched/20221119_20221213.int --slc-files ../gslcs/*.h5 --geo-files static_layers/stat*.h5 -d ../dem.tif
"""

##################
DATA_INFO = """Global Atmospheric Models:
  re-analysis_dataset      coverage  temp_resolution  spatial_resolution       latency       assimilation
  --------------------------------------------------------------------------------------------------------
  ERA5(T)  (ECMWF)          global       hourly        0.25 deg (~31 km)   3 months (5 days)    4D-Var
  ERA-Int  (ECMWF)          global       6-hourly      0.75 deg (~79 km)        2 months        4D-Var
  MERRA(2) (NASA Goddard)   global       6-hourly     0.5*0.625 (~50 km)       2-3 weeks        3D-Var
  NARR     (NOAA, working from Jan 1979 to Oct 2014)

Notes for data access:
  For MERRA2, you need an Earthdata account, and pre-authorize the "NASA GESDISC DATA ARCHIVE" application
      following https://disc.gsfc.nasa.gov/earthdata-login.
  For ERA5 from CDS, you need to agree to the Terms of Use of every datasets that you intend to download.
"""


WEATHER_DIR_DEMO = """--weather-dir ~/data/aux
atmosphere/
    /ERA5
        ERA5_N20_N40_E120_E140_20060624_14.grb
        ERA5_N20_N40_E120_E140_20060924_14.grb
        ...
    /MERRA
        merra-20110126-06.nc4
        merra-20110313-06.nc4
        ...
"""



#################
def _get_cli_args():
    parser = argparse.ArgumentParser(description='Create troposphere layer.',
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog=EXAMPLE)
  
    parser.add_argument('-f', '--file', dest='inp_interferogram',
                        help='Input wrapped Interferogram, i.e. 20120806_20120812.int',
                        required=True,)
    parser.add_argument("--slc-files", nargs=argparse.ONE_OR_MORE,
                        help="List the paths of corresponding SLC files.", required=True,)
    parser.add_argument("--geo-files", nargs=argparse.ONE_OR_MORE,
                        help="List the paths of corresponding geometry files.", required=True,)

    # delay calculation
    delay = parser.add_argument_group('delay calculation')
    delay.add_argument('-m', '--model', dest='tropo_model', default='ERA5', 
                       help=f'source of the atmospheric model (default: %(default)s). Choices are {ALL_MODELS}')
    delay.add_argument('--delay', dest='delay_type', default='comb', choices={'comb', 'dry', 'wet'},
                       help='Delay type to calculate, comb contains both wet and dry delays (default: %(default)s).')
    delay.add_argument('--hour', type=str, help='time of data in HH, e.g. 12, 06')
    
    delay.add_argument('-s', '--sdir', '--scratch-dir', dest='scratch_dir', default='./',
                       help='Scratch directory as the working directory for outputs' +
                            'e.g.: '+WEATHER_DIR_DEMO)
    delay.add_argument('-w', '--dir', '--weather-dir', dest='weather_dir', default='${WEATHER_DIR}',
                       help='parent directory of downloaded weather data file (default: %(default)s).\n' +
                            'e.g.: '+WEATHER_DIR_DEMO)
    delay.add_argument('-p','--package', dest='tropo_package', type=str, default='pyaps',
                       help='Tropospheric phase delay package, i.e. pyaps, raider')
    delay.add_argument('-d','--dem', dest='dem_file', type=str,
                       help='DEM file to calculate 2D tropospheric layer')
    parser.add_argument("-r", "--range", dest="lks_x", type=int,
                        default=1, help=("number of looks in range direction (default: %(default)s)."),)
    parser.add_argument("-a", "--azimuth", dest="lks_y", type=int, default=1, help=("number of looks in azimuth direction. (default: %(default)s)."),)
    #delay.add_argument('--custom-height', dest='custom_height', type=float,
    #                   help='specify a custom height value for delay calculation.')

    return parser

###########

def compute_troposphere_delay(cfg: dict):
    '''
    Compute the troposphere delay datacube

    Parameters
     ----------
     cfg: dict
        runconfig dictionary
     ifg_nc: str
        unwrapped interferogram in netcdf format

    Returns
     -------
     troposphere_delay_datacube: dict
        troposphere delay datacube dictionary
    '''

    error_channel = journal.error('troposphere.compute_troposphere_delay')

    tropo_delay_products = []
    # comb is short for the summation of wet and dry components
    for delay_type in ['wet', 'hydrostatic', 'comb']:
        if cfg['delay_type']:
            if (delay_type == 'hydrostatic') and \
                    (cfg['tropo_package'] == 'raider'):
                delay_type = 'hydro'
            if (delay_type == 'hydrostatic') and \
                    (cfg['tropo_package'] == 'pyaps'):
                delay_type = 'dry'

            tropo_delay_products.append(delay_type)

    grid = read_grid_param(cfg)  # Update incidence angle later to be a cube

    weather_model_params = read_weather_model_params(cfg)
    tropo_package = cfg['tropo_package'].lower()
    
    # Compute delay datacube in zenith direction:
    # pyaps package
    if tropo_package == 'pyaps':
        troposphere_delay_datacube = compute_pyaps(tropo_delay_products, grid, weather_model_params)

    # raider package
    else:
        troposphere_delay_datacube = comput_raider(tropo_delay_products, grid, weather_model_params)  
    
            
    return troposphere_delay_datacube, grid




@log_runtime
def main(iargs=None):
    """Create one interferogram from two SLCs."""
    parser = _get_cli_args()
    args = parser.parse_args(args=iargs)

     # check + default: -w / --weather-dir option (expand special symbols)
    args.weather_dir = os.path.expanduser(args.weather_dir)
    args.scratch_dir = os.path.expandvars(args.scratch_dir)
    if args.weather_dir == '${WEATHER_DIR}':
        # fallback to current dir if env var WEATHER_DIR is not defined
        args.weather_dir = './'
    args.weather_dir = os.path.abspath(args.weather_dir)
    args.correction_dir = os.path.join(args.scratch_dir, 'corrections')
    
    check_package(args)

    # get corresponding grib files info
    get_grib_info(args)
    
    ## 1. download
    print('\n'+'-'*80)
    print('Download global atmospheric model files...')   # So far only pyaps
    if args.tropo_package == 'pyaps':
        args.grib_files = dload_grib_files(
            args.grib_files,
            tropo_model=args.tropo_model,
            snwe=args.snwe)
    else:
        args.grib_files = dload_grib_files_raider(args)
        #args.grib_files = ['/u/aurora-r0/smirzaee/scratch/D06190000061900001.nc', 
        #                   '/u/aurora-r0/smirzaee/scratch/D07010000070100001.nc']

    
    # prepare geometry data
    print('\n'+'-'*80)
    print('Prepare geometry files...')
    args.geo_files = prepare_geometry(args)

    inputs = vars(args)

    tropo_delay_datacube, grid = compute_troposphere_delay(inputs)

    tropo_delay_2d = compute_2d_delay(tropo_delay_datacube, grid, inputs['geo_files'])
    
    write_tropo(tropo_delay_2d, inputs['inp_interferogram'], inputs['correction_dir'])
    

#############





if __name__ == "__main__":
    main(sys.argv[1:])