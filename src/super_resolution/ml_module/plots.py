from matplotlib import pyplot as plt
import matplotlib.path as mpath
import os
import sys
from pathlib import Path
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd

import xarray as xr
import dask

import cartopy
import cartopy.crs as ccrs
import pyproj

MONTHS_DICT = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
               7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"}

def predictions_plot(predictions_da: xr.Dataset, month: int = 0, args=None, savedir: str = '.') -> None:
    """
    Plot the model's predictions for the selected month.

    Parameters:
    -----------
    predictions_da: xarray.Dataset
        The model's predictions as an xarray Dataset.
    month: int
        The month for which to plot predictions.
    args: argparse.Namespace
        The command-line arguments.
    savedir: Path
        The directory where the plot will be saved.

    Returns:
    --------
        None
    """
    if args is not None:
        savedir = args.output_dir
        start_month = int(args.start_date.split('-')[1]) 
        

    # month = 0  # January 
    map_proj = ccrs.SouthPolarStereo(central_longitude=0.0)

    tot_months = len(predictions_da.time)

    for m in range(tot_months):
        plot_month = start_month + m
        if plot_month > 12:
            plot_month = plot_month - 12
        month_name = MONTHS_DICT.get(plot_month, "Unknown")

        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={'projection': map_proj})

        predictions_da.isel(time=m).plot(
            ax=ax, x='x', y='y',
            transform=ccrs.SouthPolarStereo(),
            add_colorbar=True,
            cmap='RdBu_r')
        ax.coastlines('10m')
        ax.gridlines(draw_labels=False)
        ax.set_title(f'Predicted SMB — {month_name} {args.start_date[:4]}', fontsize=10)
        plt.savefig(os.path.join(savedir, f'predicted_smb_month_{args.start_date[:4]}_{m}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    return 