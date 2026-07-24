"""This file contains the reprojection functions for the emulator's inputs."""

import numpy as np
import xarray as xr
import pyproj
import logging

# from super_resolution.monitoring import setup_logging
# setup_logging()
logger = logging.getLogger(__name__)



def printMaxMin_XY(varx, vary, unit = 'km'):
    print('Max and min values of x: {0:.2f}{4} -> {1:.2f}{4} and y: {2:.2f}{4} -> {3:.2f}{4}'.format(varx.min().data/1e3, 
                                                                     varx.max().data/1e3, 
                                                                     vary.min().data/1e3, 
                                                                     vary.max().data/1e3, unit))
    
def printMaxMin_LatLon(varlat, varlon, unit = '°'):
    print('Max and min values of lat: {0:.2f}{4} -> {1:.2f}{4} and lon: {2:.2f}{4} -> {3:.2f}{4}'.format(varlat.min().data/1e3, 
                                                                     varlat.max().data/1e3, 
                                                                     varlon.min().data/1e3, 
                                                                     varlon.max().data/1e3, unit))
    
def printShape(var, dim1, dim2, dim3, ds = 'GCM'):
    print('Shape of {6} data: {3}: {0}, {4}: {1}, {5}: {2}\n-----------------'.format(var.shape[0],
                                                                 var.shape[1],
                                                                 var.shape[2], dim1, dim2, dim3, ds))
def print_raster(raster):
    print(
        f"shape: {raster.rio.shape}\n"
        f"resolution: {raster.rio.resolution()}\n"
        f"bounds: {raster.rio.bounds()}\n"
        f"CRS: {raster.rio.crs}\n"
    )


def create_target_grid_from_RACMO(racmo_ds: xr.Dataset, resx_km: float = 14.0, resy_km: float = 28.0) -> tuple:
    """
    Create a uniform grid in EPSG:3031 spanning the RACMO domain at a given resolution.
    In our case, we select a resolution that approximately matches the ERA5 grid spacing at our target latitude of 75°S, 
    which is around 14km in x, and around 28km in y.

    Parameters:
    ----------
    racmo_ds : xarray.Dataset
        The RACMO dataset containing the original grid information.
    resx_km : float
        The desired resolution in the x-direction (in kilometers).
    resy_km : float
        The desired resolution in the y-direction (in kilometers).
    
    Returns:
    -------
    tuple
        A tuple containing the x and y coordinates of the target grid.
    """
    x = racmo_ds.x.values
    y = racmo_ds.y.values

    # get the boundaries of RCM grid
    x_lower, x_upper = float(x.min()), float(x.max())
    y_lower, y_upper = float(y.min()), float(y.max())

    # transform target resolution from km to m
    # METHOD 1: np arange
    resx_m = resx_km * 1000
    resy_m = resy_km * 1000
    # create a grid within the original boundaries at the target resolution
    # gridx = np.arange(x_lower, x_upper + resx_m, resx_m)
    # gridy = np.arange(y_lower, y_upper + resy_m, resy_m)

    # METHOD 2: rescaling 
    # compute the scaling factor between the original grid and the target grid
    factor_x = resx_m/2000
    factor_y = resy_m/2000
    # rescale
    x_new = racmo_ds.x * factor_x
    y_new = racmo_ds.y * factor_y
    # cut to the same extent as the original grid
    gridx = x_new[(x_new <= x_upper) & (x_new > x_lower)]
    gridy = y_new[(y_new <= y_upper) & (y_new > y_lower)]
    #gridx = x_new
    #gridy = y_new

    return gridx, gridy


def reproject_ERA5_to_RACMO_grid(era5_ds: xr.Dataset, gridx: np.ndarray, gridy: np.ndarray) -> xr.Dataset:
    """
    Reproject ERA5 (EPSG:4326, lat/lon 0-360) onto a polar stereographic EPSG:3031 grid.
    Uses the same inverse-transform approach as InterpolateGCM: each target (x,y)
    point is converted back to lat/lon, then used as a lookup coordinate in ERA5.

    Parameters:
    ----------
    era5_ds : xarray.Dataset
        The ERA5 dataset containing the original lat/lon grid.
    gridx : np.ndarray
        The x coordinates of the target grid in EPSG:3031.
    gridy : np.ndarray
        The y coordinates of the target grid in EPSG:3031.  

    Returns:
    -------
    xarray.Dataset
        The ERA5 dataset reprojected onto the target polar stereographic grid.
    """
    source_crs = pyproj.CRS(4326)  # ERA5: WGS84 lat/lon
    target_crs = pyproj.CRS(3031)  # Antarctic polar stereographic

    logger.info("Reprojecting  ERA5 to RACMO grid")

    # Normalise ERA5 longitudes from 0–360 to –180–180
    # This line is needed when ERA5 is downloaded from edh, DDM takes care of this internally.
    # era5 = era5_ds.assign_coords(longitude=((era5_ds.longitude + 180) % 360) - 180).sortby('longitude')

    # Build 2D meshgrid of target x/y and back-transform to lat/lon
    xmesh, ymesh = np.meshgrid(gridx, gridy)
    polar_to_latlon = pyproj.Transformer.from_crs(target_crs, source_crs, always_xy=True)
    lon_grid, lat_grid = polar_to_latlon.transform(xmesh, ymesh)

    lon_da = xr.DataArray(lon_grid, dims=('y', 'x'))
    lat_da = xr.DataArray(lat_grid, dims=('y', 'x'))

    logger.info(f"Interpolating {list(era5_ds.data_vars)} onto {len(gridx)}x{len(gridy)} grid...")

    era5_xy = era5_ds.interp(
        {'x': lon_da, 'y': lat_da},
        method='nearest'
    ).load()

    # Replace lat/lon coords with x/y
    era5_xy = era5_xy.assign_coords({
        'x': (('x',), gridx.data),
        'y': (('y',), gridy.data)
    })
    # .drop_vars(['longitude', 'latitude']).reset_coords()

    era5_xy = era5_xy.rio.write_crs("epsg:3031")

    return era5_xy