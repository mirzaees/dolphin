from __future__ import annotations

import os
#from pathlib import Path
from typing import Iterable, Optional, Sequence, Union
#from shapely import wkt
import datetime
from scipy.interpolate import RegularGridInterpolator
import h5py 
import re
from argparse import Namespace
import numpy as np
from osgeo import osr, gdal

import matplotlib
matplotlib.use('TKagg')
import matplotlib.pyplot as plt

import pyaps3 as pa
from dolphin._log import get_log
from dolphin._types import Bbox, Filename
from dolphin import io, stitching
from dolphin.utils import group_by_date
from dolphin.opera_utils import get_union_polygon

#dal.UseExceptions()

logger = get_log(__name__)

EARTH_RADIUS = 6371.0088e3 
HEIGHT = 693000.0

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"

WEATHER_MODEL_HOURS = {
    'ERA5'   : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    'ERAINT' : [0, 6, 12, 18],
    'MERRA'  : [0, 6, 12, 18],
}

HEIGHT_LEVELS = np.concatenate(([-100], np.arange(0, 9000, 500)))

WEATHER_MODELS = {'pyaps':['ERA5', 'ERAI', 'MERRA2', 'NARR', 'ERA', 'MERRA1'],
                  'raider':['ERA5', 'ERA5T', 'HRRR', 'GMAO', 'HRES']}

ALL_MODELS = []
for key, value in WEATHER_MODELS.items():
    for model in value:
        if not model in ALL_MODELS:
            ALL_MODELS.append(model)


def check_package(inps):
    if not inps.tropo_model in WEATHER_MODELS[inps.tropo_package]:
        raise ValueError(f"{inps.tropo_package} does not support {inps.tropo_model} or you do not have the licence")
    return

def prepare_geometry(cfg): 

    cfg.geom_dir = os.path.join(os.path.abspath(cfg.scratch_dir), 'geometry')
    os.makedirs(cfg.geom_dir, exist_ok=True)

    stitched_geo_list = {}

    strides = {"x": int(cfg.lks_x), "y": int(cfg.lks_y)}
    stitched_geom_files = []
    
    # local_incidence_angle needed by anyone?
    datasets = ["los_east", "los_north", "layover_shadow_mask", "local_incidence_angle"]
  
    for ds_name in datasets:
        outfile = cfg.geom_dir + f"/{ds_name}_full.tif"
        print(f"Creating {outfile}")
        stitched_geom_files.append(outfile)
        # Used to be:
        # /science/SENTINEL1/CSLC/grids/static_layers
        # we might also move this to dolphin if we do use the layers
        ds_path = f"/data/{ds_name}"
        cur_files = [io.format_nc_filename(f, ds_name=ds_path) for f in cfg.geo_files]

        if ds_name == 'layover_shadow_mask':
            no_data = 127
        else:
            no_data = 0

        stitching.merge_images(
            cur_files,
            outfile=outfile,
            driver="GTiff",
            out_bounds=None,
            out_bounds_epsg=None,
            in_nodata=no_data,
            out_nodata=no_data,
            target_aligned_pixels=True,
            strides=strides,
            resample_alg="nearest",
            overwrite=False,
        )

    #matching_file = '/net/kraken/nobak/smirzaee/Folsom/sequential/crop/scratch/unwrapped/20170114_20170120.unw.tif'

    height_file = cfg.geom_dir + "/height.tif"
    stitched_geo_list['height'] = height_file
    if not os.path.exists(height_file):
        print(f"Creating {height_file}")
        stitched_geom_files.append(height_file)
        stitching.warp_to_match(
            input_file=cfg.dem_file,
            match_file=cfg.inp_interferogram,
            output_file=height_file,
            resample_alg="cubic",
        )

    for ds_name in datasets:
        inpfile = cfg.geom_dir + f"/{ds_name}_full.tif"
        outfile = cfg.geom_dir + f"/{ds_name}.tif"
        stitched_geo_list[ds_name] = outfile
        if os.path.exists(outfile):
            continue
        print(f"Creating {outfile}")
        
        stitching.warp_to_match(
            input_file=inpfile,
            match_file=cfg.inp_interferogram,
            output_file=outfile,
            resample_alg="cubic",
        )

    return stitched_geo_list


def dload_grib_files_raider(inps): 
    import importlib
    from RAiDER.processWM import prepareWeatherModel
    module_name = importlib.import_module(f"RAiDER.models.{inps.tropo_model.lower()}")
    weather_obj = getattr(module_name, inps.tropo_model.upper())
    
    grib_files = []

    weather_model = weather_obj()
    for date in inps.date_list:
        date_time_string = f"{date} {inps.hour:02}:00:00"
        date_time = datetime.datetime.strptime(date_time_string, '%Y%m%d %H:%M:%S')
        weather_model.set_wmLoc(inps.weather_dir)
        prepareWeatherModel(weather_model, date_time, ll_bounds=inps.snwe)
        grib_files.append(weather_model.out_file(inps.weather_dir))

    return grib_files

def dload_grib_files(grib_files, tropo_model='ERA5', snwe=None):
    """Download weather re-analysis grib files using PyAPS
    Parameters: grib_files : list of string of grib files
    Returns:    grib_files : list of string
    """
    print('-'*50)
    print('downloading weather model data using PyAPS ...')

    # Get date list to download (skip already downloaded files)
    grib_files_exist = check_exist_grib_file(grib_files, print_msg=True)
    grib_files2dload = sorted(list(set(grib_files) - set(grib_files_exist)))
    date_list2dload = [str(re.findall(r'\d{8}', os.path.basename(i))[0]) for i in grib_files2dload]
    print('number of grib files to download: %d' % len(date_list2dload))
    print('-'*50)

    # Download grib file using PyAPS
    if len(date_list2dload) > 0:
        hour = re.findall(r'\d{8}[-_]\d{2}', os.path.basename(grib_files2dload[0]))[0].replace('-', '_').split('_')[1]
        grib_dir = os.path.dirname(grib_files2dload[0])

        # Check for non-empty account info in PyAPS config file
        check_pyaps_account_config(tropo_model)

        # try 3 times to download, then use whatever downloaded to calculate delay
        i = 0
        while i < 3:
            i += 1
            try:
                if tropo_model in ['ERA5', 'ERAINT']:
                    pa.ECMWFdload(
                        date_list2dload,
                        hour,
                        grib_dir,
                        model=tropo_model,
                        snwe=snwe,
                        flist=grib_files2dload)

                elif tropo_model == 'MERRA':
                    pa.MERRAdload(date_list2dload, hour, grib_dir)

                elif tropo_model == 'NARR':
                    pa.NARRdload(date_list2dload, hour, grib_dir)
            except:
                if i < 3:
                    print(f'WARNING: the {i} attempt to download failed, retry it.\n')
                else:
                    print('\n\n'+'*'*50)
                    print('WARNING: downloading failed for 3 times, stop trying and continue.')
                    print('*'*50+'\n\n')
                pass

    # check potentially corrupted files
    grib_files = check_exist_grib_file(grib_files, print_msg=False)
    return grib_files
    

def snwe2str(snwe: Bbox):
    """Get area extent in string"""
    if not snwe:
        return None
    
    S, N, W, E = add_buffer(snwe)

    area = ''
    area += f'_S{abs(S)}' if S < 0 else f'_N{abs(S)}'
    area += f'_S{abs(N)}' if N < 0 else f'_N{abs(N)}'
    area += f'_W{abs(W)}' if W < 0 else f'_E{abs(W)}'
    area += f'_W{abs(E)}' if E < 0 else f'_E{abs(E)}'

    return area

def add_buffer(snwe):
    s, n, w, e = snwe

    min_buffer=1
    # lat/lon0/1 --> SNWE
    S = np.floor(min(s,n) - min_buffer).astype(int)
    N = np.ceil(max(s,n) + min_buffer).astype(int)
    W = np.floor(min(w,e) - min_buffer).astype(int)
    E = np.ceil(max(w,e) + min_buffer).astype(int)
    return S, N, W, E


def get_grib_filenames(date_list, hour, model, grib_dir, package='pyaps', snwe=None):
    """Get default grib file names based on input info.
    Parameters: date_list  - list of str, date in YYYYMMDD format
                hour       - str, hour in 2-digit with zero padding
                model      - str, global atmospheric model name
                grib_dir   - str, local directory to save grib files
                snwe       - tuple of 4 int, for ERA5 only.
    Returns:    grib_files - list of str, local grib file path
    """
    # area extent
    area = snwe2str(snwe)

    grib_files = []
    for d in date_list:
        if model == 'ERA5':
            if area:
                grib_file = f'ERA5{area}_{d}_{hour}.grb'
            else:
                grib_file = f'ERA5_{d}_{hour}.grb'

        elif model == 'ERAINT': grib_file = f'ERA-Int_{d}_{hour}.grb'
        elif model == 'MERRA' : grib_file = f'merra-{d}-{hour}.nc4'
        elif model == 'NARR'  : grib_file = f'narr-a_221_{d}_{hour}00_000.grb'
        elif model == 'ERA'   : grib_file = f'ERA_{d}_{hour}.grb'
        elif model == 'MERRA1': grib_file = f'merra-{d}-{hour}.hdf'

        if package == 'raider':
            grib_file = grib_file.split('.')[0] + '.nc'

        grib_files.append(os.path.join(grib_dir, grib_file))
       

    return grib_files

def closest_weather_model_hour(sar_acquisition_time, grib_source='ERA5'):
    """Find closest available time of weather product from SAR acquisition time
    Parameters: sar_acquisition_time - str, SAR data acquisition time in seconds
                grib_source          - str, Grib Source of weather reanalysis product
    Returns:    grib_hr              - str, time of closest available weather product
    Example:
        '06' = closest_weather_model_hour(atr['CENTER_LINE_UTC'])
        '12' = closest_weather_model_hour(atr['CENTER_LINE_UTC'], 'NARR')
    """
    # get hour/min of SAR acquisition time
    #sar_time = datetime.strptime(sar_acquisition_time, "%Y-%m-%d %H:%M:%S.%f").hour
    sar_time = sar_acquisition_time.hour
                                 
    # find closest time in available weather products
    grib_hr_list = WEATHER_MODEL_HOURS[grib_source]
    grib_hr = int(min(grib_hr_list, key=lambda x: abs(x-sar_time)))

    # add zero padding
    grib_hr = f"{grib_hr:02d}"
    return grib_hr

def get_grib_info(inps):
    """Read the following info from inps
        inps.grib_dir
        inps.atr
        inps.snwe
        inps.grib_files
    """
    # grib data directory, under weather_dir
    inps.grib_dir = os.path.join(inps.weather_dir, inps.tropo_model)
    if not os.path.isdir(inps.grib_dir):
        os.makedirs(inps.grib_dir)
        print(f'make directory: {inps.grib_dir}')

    acquisition_time = get_zero_doppler_time(inps.slc_files[-1])
    inps.hour = closest_weather_model_hour(acquisition_time, grib_source=inps.tropo_model)
    
    inps.wavelength = get_radar_wavelength(inps.slc_files[-1])
    inps.date_list = os.path.basename(inps.inp_interferogram).split('.')[0].split('_')
    
    wsen = io.get_raster_bounds(inps.inp_interferogram)
    epsg = io.get_raster_crs(inps.inp_interferogram).to_epsg()

    
    if epsg != 4326:
        # x, y to Lat/Lon
        srs_src = osr.SpatialReference()
        srs_src.ImportFromEPSG(epsg)

        srs_wgs84 = osr.SpatialReference()
        srs_wgs84.ImportFromEPSG(4326)

        # Transform the xy to lat/lon
        transformer_xy_to_latlon = osr.CoordinateTransformation(srs_src, srs_wgs84)

        # Stack the x and y
        x_y_pnts_radar = np.stack(([wsen[0], wsen[2]], [wsen[1], wsen[3]]), axis=-1)

        # Transform to lat/lon
        lat_lon_radar = np.array(
            transformer_xy_to_latlon.TransformPoints(x_y_pnts_radar))
        
        inps.snwe = (lat_lon_radar[0,0], lat_lon_radar[1,0], lat_lon_radar[0,1], lat_lon_radar[1,1])
    else:

        inps.snwe = (wsen[1], wsen[3], wsen[0], wsen[2]) 
    

    # grib file list
    if inps.tropo_package == 'pyaps':
        inps.grib_files = get_grib_filenames(
            date_list=inps.date_list,
            hour=inps.hour,
            model=inps.tropo_model,
            grib_dir=inps.grib_dir,
            snwe=inps.snwe)


    return inps

def check_pyaps_account_config(tropo_model):
    """Check for input in PyAPS config file. If they are default values or are empty, then raise error.
    Parameters: tropo_model - str, tropo model being used to calculate tropospheric delay
    Returns:    None
    """
    # Convert MintPy tropo model name to data archive center name
    # NARR model included for completeness but no key required
    MODEL2ARCHIVE_NAME = {
        'ERA5' : 'CDS',
        'ERAI' : 'ECMWF',
        'MERRA': 'MERRA',
        'NARR' : 'NARR',
    }
    SECTION_OPTS = {
        'CDS'  : ['key'],
        'ECMWF': ['email', 'key'],
        'MERRA': ['user', 'password'],
    }

    # Default values in cfg file
    default_values = [
        'the-email-address-used-as-login@ecmwf-website.org',
        'the-user-name-used-as-login@earthdata.nasa.gov',
        'the-password-used-as-login@earthdata.nasa.gov',
        'the-email-adress-used-as-login@ucar-website.org',
        'your-uid:your-api-key',
    ]

    # account file for pyaps3 < and >= 0.3.0
    cfg_file = os.path.join(os.path.dirname(pa.__file__), 'model.cfg')
    rc_file = os.path.expanduser('~/.cdsapirc')

    # for ERA5: ~/.cdsapirc
    if tropo_model == 'ERA5' and os.path.isfile(rc_file):
        pass

    # check account info for the following models
    elif tropo_model in ['ERA5', 'ERAI', 'MERRA']:
        section = MODEL2ARCHIVE_NAME[tropo_model]

        # Read model.cfg file
        cfg_file = os.path.join(os.path.dirname(pa.__file__), 'model.cfg')
        cfg = ConfigParser()
        cfg.read(cfg_file)

        # check all required option values
        for opt in SECTION_OPTS[section]:
            val = cfg.get(section, opt)
            if not val or val in default_values:
                msg = 'PYAPS: No account info found '
                msg += f'for {tropo_model} in {section} section in file: {cfg_file}'
                raise ValueError(msg)

    return


def dload_grib_files(grib_files, tropo_model='ERA5', snwe=None):
    """Download weather re-analysis grib files using PyAPS
    Parameters: grib_files : list of string of grib files
    Returns:    grib_files : list of string
    """
    #Sfrom mintpy.tropo_pyaps3 import check_exist_grib_file
    print('-'*50)
    print('downloading weather model data using PyAPS ...')
    # Get date list to download (skip already downloaded files)
    grib_files_exist = [f for f in grib_files if os.path.exists(f)]
    #grib_files_exist = check_exist_grib_file(grib_files, print_msg=True)
    grib_files2dload = sorted(list(set(grib_files) - set(grib_files_exist)))
    date_list2dload = [str(re.findall(r'\d{8}', os.path.basename(i))[0]) for i in grib_files2dload]
    print('number of grib files to download: %d' % len(date_list2dload))
    print('-'*50)

    SNWE_buffered = add_buffer(snwe)

    # Download grib file using PyAPS
    if len(date_list2dload) > 0:
        hour = re.findall(r'\d{8}[-_]\d{2}', os.path.basename(grib_files2dload[0]))[0].replace('-', '_').split('_')[1]
        grib_dir = os.path.dirname(grib_files2dload[0])

        # Check for non-empty account info in PyAPS config file
        check_pyaps_account_config(tropo_model)

        # try 3 times to download, then use whatever downloaded to calculate delay
        i = 0
        while i < 3:
            i += 1
            try:
                if tropo_model in ['ERA5', 'ERAINT']:
                    pa.ECMWFdload(
                        date_list2dload,
                        hour,
                        grib_dir,
                        model=tropo_model,
                        snwe=SNWE_buffered,
                        flist=grib_files2dload)

                elif tropo_model == 'MERRA':
                    pa.MERRAdload(date_list2dload, hour, grib_dir)

                elif tropo_model == 'NARR':
                    pa.NARRdload(date_list2dload, hour, grib_dir)
            except:
                if i < 3:
                    print(f'WARNING: the {i} attempt to download failed, retry it.\n')
                else:
                    print('\n\n'+'*'*50)
                    print('WARNING: downloading failed for 3 times, stop trying and continue.')
                    print('*'*50+'\n\n')
                pass

    # check potentially corrupted files
    grib_files = [f for f in grib_files if os.path.exists(f)]
    return grib_files


def compute_pyaps(tropo_delay_products, grid, weather_model_params):

    troposphere_delay_datacube = dict()
    
    # X and y for the entire datacube
    y_2d_radar, x_2d_radar = np.meshgrid(grid['ycoord'], grid['xcoord'], indexing='ij')

    # Lat/lon coordinates
    lat_datacube, lon_datacube = transform_xy_to_latlon(grid['epsg'], x_2d_radar, y_2d_radar)
    
    for tropo_delay_product in tropo_delay_products:
        tropo_delay_datacube_list = []
        
        for index, hgt in enumerate(HEIGHT_LEVELS):
            dem_datacube = np.full(lat_datacube.shape, hgt)
            
            # Delay for the reference image
            ref_aps_estimator = pa.PyAPS(weather_model_params['reference_file'],
                                         dem=dem_datacube,
                                         inc=0.0,
                                         lat=lat_datacube,
                                         lon=lon_datacube,
                                         grib=weather_model_params['type'],
                                         humidity='Q',
                                         model=weather_model_params['type'],
                                         verb=False,
                                         Del=tropo_delay_product)
            
            phs_ref = np.zeros((ref_aps_estimator.ny, ref_aps_estimator.nx), dtype=np.float32)
            ref_aps_estimator.getdelay(phs_ref)

            # Delay for the secondary image
            second_aps_estimator = pa.PyAPS(weather_model_params['secondary_file'],
                                            dem=dem_datacube,
                                            inc=0.0,
                                            lat=lat_datacube,
                                            lon=lon_datacube,
                                            grib=weather_model_params['type'],
                                            humidity='Q',
                                            model=weather_model_params['type'],
                                            verb=False,
                                            Del=tropo_delay_product)

            phs_second = np.zeros((second_aps_estimator.ny, second_aps_estimator.nx), dtype=np.float32)
            second_aps_estimator.getdelay(phs_second)

            # Convert the delay in meters to radians
            tropo_delay_datacube_list.append(
                -(phs_ref - phs_second) * 4.0 * np.pi / float(weather_model_params['wavelength']))

            # Tropo delay datacube
        tropo_delay_datacube = np.stack(tropo_delay_datacube_list)
        # Create a maksed datacube that excludes the NaN values
        tropo_delay_datacube_masked = np.ma.masked_invalid(tropo_delay_datacube)

        # Save to the dictionary in memory
        model_type = weather_model_params['type']
        tropo_delay_product_name = f'tropoDelay_pyaps_{model_type}_Zenith_{tropo_delay_product}'
        troposphere_delay_datacube[tropo_delay_product_name]  = tropo_delay_datacube_masked


    return troposphere_delay_datacube


def comput_raider(tropo_delay_products, grid, weather_model_params):
    import xarray as xr
    from RAiDER.llreader import BoundingBox
    from RAiDER.losreader import Zenith
    from RAiDER.delay import tropo_delay as raider_tropo_delay
    from RAiDER.models.hres import HRES

    
    
    def _convert_HRES_to_raider_NetCDF(weather_model_file,
                                       lat_lon_bounds,
                                       weather_model_output_dir):
        '''
        Internal convenience function to convert the ECMWF NetCDF to RAiDER NetCDF

        Parameters
        ----------
        weather_model_file: str
        HRES NetCDF weather model file
        lat_lon_bounds: list
        bounding box of the RSLC
        weather_model_output_dir: str
        the output directory of the RAiDER internal NetCDF file
        Returns
        -------
        the path of the RAiDER internal NetCDF file
        '''
        
        os.makedirs(weather_model_output_dir, exist_ok=True)
        ds = xr.open_dataset(weather_model_file)
        
        # Get the datetime of the weather model file
        weather_model_time = ds.time.values.astype('datetime64[s]').astype(datetime.datetime)[0]
        hres = HRES()
        # Set up the time, Lat/Lon, and working location, where
        # the lat/lon bounds are applied to clip the global
        # weather model to minimize the data processing
        hres.setTime(weather_model_time)
        hres.set_latlon_bounds(ll_bounds = lat_lon_bounds)
        hres.set_wmLoc(weather_model_output_dir)

        # Load the ECMWF NetCDF weather model
        hres.load_weather(weather_model_file)

        # Process the weather model data
        hres._find_e()
        hres._uniform_in_z(_zlevels=None)

        # This function implemented in the RAiDER
        # fills the NaNs with 0
        hres._checkForNans()

        hres._get_wet_refractivity()
        hres._get_hydro_refractivity()
        hres._adjust_grid(hres.get_latlon_bounds())

        # Compute Zenith delays at the weather model grid nodes
        hres._getZTD()

        output_file = hres.out_file(weather_model_output_dir)
        hres._out_name =  output_file

        # Return the ouput file if it exists
        if os.path.exists(output_file):
            return output_file
        else:
            # Write to hard drive
            return hres.write()
        
    reference_weather_model_file = weather_model_params['reference_file']
    secondary_weather_model_file = weather_model_params['secondary_file']
    
    troposphere_delay_datacube = dict()
    
    '''
    # ouput location
    scratch_path = weather_model_params['scratch_dir']
    weather_model_output_dir = os.path.join(scratch_path, 'weather_model_files')
        

    # AOI bounding box
    margin = 0.1
    S, N, W, E = grid['snwe']
    lat_lon_bounds = [S - margin,
                      N + margin,
                      W - margin,
                      E + margin]
    
    

    # Convert to RAiDER NetCDF
    reference_weather_model_file = \
            _convert_HRES_to_raider_NetCDF(weather_model_params['reference_file'],
                                           lat_lon_bounds, weather_model_output_dir)
    

    secondary_weather_model_file = \
            _convert_HRES_to_raider_NetCDF(weather_model_params['secondary_file'],
                                           lat_lon_bounds, weather_model_output_dir)

    '''

    aoi = BoundingBox(grid['snwe'])
    aoi.xpts = grid['xcoord']
    aoi.ypts = grid['ycoord']

    # Zenith
    delay_direction_obj = Zenith()

    # Troposphere delay computation
    # Troposphere delay datacube computation
    tropo_delay_reference, _ = raider_tropo_delay(dt=weather_model_params['reference_time'],
                                                  weather_model_file=reference_weather_model_file,
                                                  aoi=aoi,
                                                  los=delay_direction_obj,
                                                  height_levels=HEIGHT_LEVELS,
                                                  out_proj=grid['epsg'])
    
    tropo_delay_secondary, _ = raider_tropo_delay(dt=weather_model_params['secondary_time'],
                                                  weather_model_file=secondary_weather_model_file,
                                                  aoi=aoi,
                                                  los=delay_direction_obj,
                                                  height_levels=HEIGHT_LEVELS,
                                                  out_proj=grid['epsg'])
    

    for tropo_delay_product in tropo_delay_products:

        # Compute troposphere delay with raider package
        # comb is the summation of wet and hydro components
        if tropo_delay_product == 'comb':
            tropo_delay = tropo_delay_reference['wet'] + tropo_delay_reference['hydro'] - \
                    tropo_delay_secondary['wet'] - tropo_delay_secondary['hydro']
        else:
            tropo_delay = tropo_delay_reference[tropo_delay_product] - \
                    tropo_delay_secondary[tropo_delay_product]

        # Convert it to radians units
        tropo_delay_datacube = -tropo_delay * 4.0 * np.pi / weather_model_params['wavelength']

        # Create a maksed datacube that excludes the NaN values
        tropo_delay_datacube_masked = np.ma.masked_invalid(tropo_delay_datacube)

        # Interpolate to radar grid to keep its dimension consistent with other datacubes
        tropo_delay_interpolator = RegularGridInterpolator((tropo_delay_reference.z,
                                                            tropo_delay_reference.y,
                                                            tropo_delay_reference.x),
                                                           tropo_delay_datacube_masked,
                                                           method='linear', bounds_error=False)
        
        # Interpolate the troposphere delay
        hv, yv, xv = np.meshgrid(HEIGHT_LEVELS,
                                 grid['ycoord'],
                                 grid['xcoord'],
                                 indexing='ij')

        pnts = np.stack(
                (hv.flatten(), yv.flatten(), xv.flatten()), axis=-1)

        # Interpolate
        tropo_delay_datacube = tropo_delay_interpolator(pnts).reshape(hv.shape)


        # Save to the dictionary in memory
        model_type = weather_model_params['type']
        tropo_delay_product_name = f'tropoDelay_raider_{model_type}_Zenith_{tropo_delay_product}'
        troposphere_delay_datacube[tropo_delay_product_name]  = tropo_delay_datacube


    return troposphere_delay_datacube

# def interpolate_grid(data_cube, points)

def write_tropo(tropo_2d, inp_interferogram, out_dir):
    dates = os.path.basename(inp_interferogram).split('.')[0]
    os.makedirs(out_dir, exist_ok=True)

    for key, value in tropo_2d.items():
        output = os.path.join(out_dir, f"{dates}_{key}.tif")
        io.write_arr(arr=value, output_name=output, like_filename=inp_interferogram) 
    return

def read_grid_param(cfg):
    xsize, ysize = io.get_raster_xysize(cfg['inp_interferogram'])
    crs = io.get_raster_crs(cfg['inp_interferogram'])
    gt = io.get_raster_gt(cfg['inp_interferogram'])
    ycoord, xcoord = create_yx_arrays(gt, (ysize, xsize)) # 500 m spacing
    epsg = crs.to_epsg()

    # Note on inc_angle: This was a data cube, we are using a constant now and need to be updated 
    grid = {'xcoord': xcoord,
            'ycoord': ycoord,
            'snwe': cfg['snwe'],
            'epsg': epsg,
            'geotransform': gt,
            'shape': (ysize, xsize),
            'crs': crs.to_wkt()} # cube

    return grid

def read_weather_model_params(cfg):
    grouped_slc = group_by_date([p for p in cfg['slc_files'] if "compressed" not in os.path.basename(p)])
    reference_date, secondary_date = [datetime.datetime.strptime(tt, '%Y%m%d').date() for tt in cfg['date_list']]
    for key, value in grouped_slc.items():
        if key[0] == reference_date:
            reference_time = get_zero_doppler_time(value[0])
        if key[0] == secondary_date:
            secondary_time = get_zero_doppler_time(value[0])    
    
    weather_model_params = {'reference_file':cfg['grib_files'][0],
                    'secondary_file':cfg['grib_files'][1],
                    'scratch_dir': cfg['scratch_dir'],
                    'type':cfg['tropo_model'].upper(),
                    'reference_time': reference_time,
                    'secondary_time': secondary_time,
                    'wavelength':cfg['wavelength']}
    
    return weather_model_params


def get_radar_wavelength(filename: Filename):
    """Get the radar wavelength from the CSLC product.

    Parameters
    ----------
    filename : Filename
        Path to the CSLC product.

    Returns
    -------
    wavelength : float
        Radar wavelength.
    """
    dset = "/metadata/processing_information/input_burst_metadata/wavelength"
    value = _get_dset_and_attrs(filename, dset)[0]
    return value

def get_zero_doppler_time(filename: Filename, type_: str = "start") -> datetime:
    """Get the full acquisition time from the CSLC product.

    Uses `/identification/zero_doppler_{type_}_time` from the CSLC product.

    Parameters
    ----------
    filename : Filename
        Path to the CSLC product.
    type_ : str, optional
        Either "start" or "stop", by default "start".

    Returns
    -------
    str
        Full acquisition time.
    """

    def get_dt(in_str):
        return datetime.datetime.strptime(in_str.decode("utf-8"), DATETIME_FORMAT)

    dset = f"/identification/zero_doppler_{type_}_time"
    value = _get_dset_and_attrs(filename, dset, parse_func=get_dt)[0]
    return value

def _get_dset_and_attrs(
    filename: Filename,
    dset_name: str,
    parse_func: Callable = lambda x: x,
) -> tuple[Any, dict[str, Any]]:
    """Get one dataset's value and attributes from the CSLC product.

    Parameters
    ----------
    filename : Filename
        Path to the CSLC product.
    dset_name : str
        Name of the dataset.
    parse_func : Callable, optional
        Function to parse the dataset value, by default lambda x: x
        For example, could be parse_func=lambda x: x.decode("utf-8") to decode,
        or getting a datetime object from a string.

    Returns
    -------
    dset : Any
        The value of the scalar
    attrs : dict
        Attributes.
    """
    with h5py.File(filename, "r") as hf:
        dset = hf[dset_name]
        attrs = dict(dset.attrs)
        value = parse_func(dset[()])
        return value, attrs


def create_yx_arrays(
    gt: list[float], shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Create the x and y coordinate datasets."""
    ysize, xsize = shape
    # Parse the geotransform
    x_origin, x_res, _, y_origin, _, y_res = gt
    y_end = y_origin + y_res * ysize
    x_end = x_origin + x_res * xsize

    # Make the x/y arrays
    y = np.arange(y_origin, y_end - 500, -500)
    x = np.arange(x_origin, x_end + 500, 500)
    return y, x

def transform_xy_to_latlon(epsg, x, y): #, margin = 0.1):
    '''
    Convert the x, y coordinates in the source projection to WGS84 lat/lon

    Parameters
     ----------
     epsg: int
         epsg code
     x: numpy.ndarray
         x coordinates
     y: numpy.ndarray
         y coordinates
     margin: float
         data cube margin, default is 0.1 degrees

     Returns
     -------
     lat_datacube: numpy.ndarray
         latitude of the datacube
     lon_datacube: numpy.ndarray
         longitude of the datacube
     cube_extent: tuple
         extent of the datacube in south-north-west-east convention
     '''

    # x, y to Lat/Lon
    srs_src = osr.SpatialReference()
    srs_src.ImportFromEPSG(epsg)

    srs_wgs84 = osr.SpatialReference()
    srs_wgs84.ImportFromEPSG(4326)

    if epsg != 4326:
        # Transform the xy to lat/lon
        transformer_xy_to_latlon = osr.CoordinateTransformation(srs_src, srs_wgs84)

        # Stack the x and y
        x_y_pnts_radar = np.stack((x.flatten(), y.flatten()), axis=-1)

        # Transform to lat/lon
        lat_lon_radar = np.array(
            transformer_xy_to_latlon.TransformPoints(x_y_pnts_radar))

        # Lat lon of data cube
        lat_datacube = lat_lon_radar[:, 0].reshape(x.shape)
        lon_datacube = lat_lon_radar[:, 1].reshape(x.shape)
    else:
        lat_datacube = y.copy()
        lon_datacube = x.copy()

    ## Extent of the data cube
    #cube_extent = (np.nanmin(lat_datacube) - margin, np.nanmax(lat_datacube) + margin,
    #               np.nanmin(lon_datacube) - margin, np.nanmax(lon_datacube) + margin)

    return lat_datacube, lon_datacube #, cube_extent


def compute_2d_delay(tropo_delay_cube, grid, geo_files):

    dem_file = geo_files['height']
    
    ysize, xsize = grid['shape']    
    x_origin, x_res, _, y_origin, _, y_res = grid['geotransform']

    left, top = io._apply_gt(gt=grid['geotransform'], x=0, y=0)
    right, bottom = io._apply_gt(gt=grid['geotransform'], x=xsize, y=ysize)

    bounds = (left, bottom, right, top) 

    options = gdal.WarpOptions(dstSRS=grid['crs'],
                               format='MEM',
                               xRes=x_res,
                               yRes=y_res,
                               outputBounds=bounds,
                               outputBoundsSRS=grid['crs'],
                               resampleAlg='near',)
    target_ds = gdal.Warp(
        os.path.abspath(dem_file + '.temp'),
        os.path.abspath(dem_file),
        options=options,
    )

    dem = target_ds.ReadAsArray()
    #dem = io.load_gdal(dem_file)
 
    los_east = io.load_gdal(geo_files['los_east'])
    los_north = io.load_gdal(geo_files['los_north'])
    los_up = (1 - los_east**2 - los_north**2)
    
    mask = los_east > 0

    # Make the x/y arrays
    # Note that these are the center of the pixels, whereas the GeoTransform
    # is the upper left corner of the top left pixel.
    y = np.arange(y_origin, y_origin + y_res * ysize, y_res)
    x = np.arange(x_origin, x_origin + x_res * xsize, x_res)

    yv, xv = np.meshgrid(y, x, indexing='ij')

    delay_2d = {}
    for delay_type in tropo_delay_cube.keys():
        if not delay_type in ['x', 'y', 'z']:
            #tropo_delay_datacube_masked = np.ma.masked_invalid(tropo_delay_cube[delay_type])

            tropo_delay_interpolator = RegularGridInterpolator((HEIGHT_LEVELS,
                                                                grid['ycoord'],
                                                                grid['xcoord']),
                                                                tropo_delay_cube[delay_type],
                                                                method='linear', bounds_error=False)
            
            tropo_delay_2d = np.zeros(dem.shape, dtype=np.float32)
            
            nline = 100
            for i in range(0, dem.shape[1], 100):
                if i+100 > dem.shape[0]:
                    nline = dem.shape[0] - i
                pnts = np.stack((dem[i:i+100, :].flatten(), yv[i:i+100, :].flatten(), xv[i:i+100, :].flatten()), axis=-1)
                tropo_delay_2d[i:i+100, :] = tropo_delay_interpolator(pnts).reshape(nline, dem.shape[1])
            
            out_delay_type = delay_type.replace('Zenith', 'LOS')
            delay_2d[out_delay_type] = (tropo_delay_2d / los_up)*mask
   
    return delay_2d

