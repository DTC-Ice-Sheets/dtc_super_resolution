"""Functions to download data from the Data Download Module."""

import os

import xarray as xr
from shapely.geometry import Polygon, box
from datetime import datetime
import dotenv

from data_download_module.main import download_data
from data_download_module.schemas import DataDownloadRequest
import data_download_module as ddm

dotenv.load_dotenv()

# provisionally we download from edh, but the right dataset will be served from DDM as well
def download_era5_from_edh(start_date: str, end_date: str, output_dir: str) -> xr.Dataset:
    """
    Download ERA5 data from the Earth Data Hub.
    Both dataset cover the time period from 2010-01-01 to 2023-12-31, which was used for training.

    Args:
        start_date (str): The start date for the data download, in format "YYYY-MM-DD".
        end_date (str): The end date for the data download, in format "YYYY-MM-DD".
        output_dir (str): The directory where the downloaded data will be stored.

    Returns:
        str: The path to the downloaded ERA5 data.
    """
    print('Downloading ERA5')
    era5_ds = xr.open_dataset(
        "https://edh:edh_key_IqHSdEIB55PWHT219nGSNqVVEdbPx1MfNevS2wK1oqQ@api.earthdatahub.destine.eu/era5/reanalysis-era5-single-levels-monthly-means-v0.zarr",
        chunks={},
        engine="zarr",
        consolidated=True)

    # cut to Antartica region
    # define antarctica as 90° S to 60° S in latitude and full longitude
    era5_ds = era5_ds.sel(latitude=slice(-40, -90), longitude=slice(0, 360))

    # slice temporally to match RCM data 
    era5_ds = era5_ds.sel(valid_time=slice(start_date, end_date))  

    era5_ds = era5_ds.compute()  # materialize into memory here

    # save to netcdf 
    name = "ERA5_monthly_means_2010-2023.nc"
    era5path = os.path.join(output_dir, name)
    era5_ds.to_netcdf(era5path)

    return era5path

# def download_era5_from_ddm(start_date: str, end_date: str, output_dir: str) -> str:
#     """
#     Download ERA5 data from the Data Download Module.
#Both dataset cover the time period from 2010-01-01 to 2023-12-31, which was used for training.

#     Args:
#         start_date (str): The start date for the data download, in format "YYYY-MM-DD".
#         end_date (str): The end date for the data download, in format "YYYY-MM-DD".
#         output_dir (str): The directory where the downloaded data will be stored.   

#      Returns:
#         str: The path to the downloaded ERA5 data.
#     """
#     antartica_polygon = Polygon([(-180, -60), (180, -60), (180, -90), (-180, -90), (-180, -60)])
#     # greenland_polygon = Polygon([(-73.5, 60), (-12, 60), (-12, 81), (-73.5, 81), (-73.5, 60)])
#     era5_vars = ['d2m', 'msl', 'sst', 'sp', 't2m', 'tp', 'u10', 'v10']
      
#      start_date = datetime.strptime(start_date, "%Y-%m-%d")
#     end_date = datetime.strptime(end_date, "%Y-%m-%d")
    
#     era5_request = DataDownloadRequest(dataset_id="ERA5_REANALYSIS",
#                                   output_parent_path=output_dir,
#                                   start_time=datetime.strptime(start_date, "%Y-%m-%d"),
#                                   end_time=datetime.strptime(end_date, "%Y-%m-%d"),
#                                   epsg4326_polygon=antartica_polygon,
#                                   variables_to_download=era5_vars,
#                                   raw_download=False)

#     downloaded_data_path = download_data(data_download_request=era5_request)

#     return downloaded_data_path
#     # era5_ds = xr.open_dataset(downloaded_data_path, engine="zarr")
#     # return era5_ds


def download_rcm_from_ddm(output_dir: str, start_date: str = "2020-01-01", end_date: str = "2020-02-01") -> str:
    """
    Download RCM data from the Data Download Module.
    
    Args:
        start_date (str): The start date for the data download, in format "YYYY-MM-DD".
        end_date (str): The end date for the data download, in format "YYYY-MM-DD".
        output_dir (str): The directory where the downloaded data will be stored.

    Returns:
        str: The path to the downloaded RCM data.
    """
    print('Downloading RACMO')
    output_dir = os.path.join(output_dir, "RACMO/")
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    request = ddm.schemas.DataDownloadRequest(dataset_id="RACMO",
                                output_parent_path=output_dir,
                                start_time=start_date,
                                end_time=end_date,
                                variables_to_download=["surface_mass_balance_glaciated"],
                                raw_download=False)

    downloaded_data_path = ddm.main.download_data(data_download_request=request)

    return downloaded_data_path


def download_data(start_date: str, end_date: str, output_dir: str) -> dict:
    """
    Download ERA5 and RCM data from the Data Download Module.
    Data from RACMO is not used at inference time, except for the purpose of defining the tiling grid.
    Therefore, the default values are used to download only a small subset of the data.

    Args:
        start_date (str): The start date for the data download, in format "YYYY-MM-DD".
        end_date (str): The end date for the data download, in format "YYYY-MM-DD".
        output_dir (str): The directory where the downloaded data will be stored.

    Returns:
        dict: A dictionary containing paths to the downloaded ERA5 and RCM data.
    """
    dotenv.load_dotenv()

    era5_path = download_era5_from_edh(start_date, end_date, output_dir)
    rcm_path = download_rcm_from_ddm(output_dir)

    return {"era5": era5_path, "rcm": rcm_path}