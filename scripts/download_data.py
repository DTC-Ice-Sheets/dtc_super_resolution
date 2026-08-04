"""Functions to download data from the Data Download Module."""

import pathlib
from datetime import datetime

import dotenv
from data_download_module.main import download_data as ddm_download_data
from data_download_module.schemas import DataDownloadRequest
from shapely.geometry import box

dotenv.load_dotenv()

def download_era5_from_ddm(start_date: str, end_date: str, output_dir: str) -> str:
    """
    Download ERA5 data from the Data Download Module.

    Both dataset cover the time period from 2010-01-01 to 2023-12-31, which was used for training.

    Parameters
    ----------
    start_date: str
        The start date for the data download, in format "YYYY-MM-DD".
    end_date: str
        The end date for the data download, in format "YYYY-MM-DD".
    output_dir: str
        The directory where the downloaded data will be stored.

    Returns
    -------
    str
        The path to the downloaded ERA5 data.
    """
    # antartica_polygon = Polygon([(-180, -60), (180, -60), (180, -90), (-180, -90), (-180, -60)])
    antarctica_polygon = box(-180, -90, 180, -60)
    # greenland_polygon = Polygon([(-73.5, 60), (-12, 60), (-12, 81), (-73.5, 81), (-73.5, 60)])
    era5_vars = ["d2m", "msl", "sst", "sp", "t2m", "tp", "u10", "v10"]

    if isinstance(start_date, str):
        start_date = start_date.strftime("%Y-%m-%d")
    if isinstance(end_date, str):
        end_date = end_date.strftime("%Y-%m-%d")

    output_dir = str(pathlib.Path(output_dir) / "ERA5/")

    era5_request = DataDownloadRequest(
        dataset_id="ERA5_REANALYSIS_MONTHLY_MEANS",
        output_parent_path=output_dir,
        start_time=start_date,
        end_time=end_date,
        epsg4326_polygon=antarctica_polygon,
        variables_to_download=era5_vars,
        raw_download=False,
    )

    return ddm_download_data(data_download_request=era5_request)


def download_rcm_from_ddm(output_dir: str, start_date: str = "2020-01-01", end_date: str = "2020-02-01") -> str:
    """
    Download RCM data from the Data Download Module.

    Parameters
    ----------
    output_dir: str
        The directory where the downloaded data will be stored.
    start_date: str
        The start date for the data download, in format "YYYY-MM-DD".
    end_date: str
        The end date for the data download, in format "YYYY-MM-DD".

    Returns
    -------
    str
        The path to the downloaded RCM data.
    """
    output_dir = str(pathlib.Path(output_dir) / "RACMO/")
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    request = DataDownloadRequest(
        dataset_id="RACMO",
        output_parent_path=output_dir,
        start_time=start_date,
        end_time=end_date,
        variables_to_download=["surface_mass_balance_glaciated"],
        raw_download=False,
    )

    return ddm_download_data(data_download_request=request)


def download_data(start_date: str, end_date: str, output_dir: str) -> dict:
    """
    Download ERA5 and RCM data from the Data Download Module.

    Data from RACMO is not used at inference time, except for the purpose of defining the tiling grid.
    Therefore, the default values are used to download only a small subset of the data.

    Parameters
    ----------
    start_date: str
        The start date for the data download, in format "YYYY-MM-DD".
    end_date: str
        The end date for the data download, in format "YYYY-MM-DD".
    output_dir: str
        The directory where the downloaded data will be stored.

    Returns
    -------
    dict
        A dictionary containing paths to the downloaded ERA5 and RCM data.
    """
    dotenv.load_dotenv()

    era5_path = download_era5_from_ddm(start_date, end_date, output_dir)
    rcm_path = download_rcm_from_ddm(output_dir)

    return {"era5": era5_path, "racmo": rcm_path}
