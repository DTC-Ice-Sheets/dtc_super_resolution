from datetime import datetime
from pathlib import Path

import cartopy.crs as ccrs
import xarray as xr
from matplotlib import pyplot as plt

MONTHS_DICT = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}


def predictions_plot(predictions_da: xr.Dataset, start_date: datetime, savedir: str = ".") -> None:
    """
    Plot the model's predictions for the selected month.

    Parameters
    ----------
    predictions_da: xr.Dataset
        The model's predictions as an xarray Dataset.
    start_date: datetime
        The start date from which to start plotting.
    savedir: str
        The directory where the plot will be saved.
    """
    map_proj = ccrs.SouthPolarStereo(central_longitude=0.0)

    tot_months = len(predictions_da.time)
    start_month = start_date.month
    start_year = start_date.year

    for m in range(tot_months):
        plot_month = start_month + m
        if plot_month > 12:
            plot_month = plot_month - 12
        month_name = MONTHS_DICT.get(plot_month, "Unknown")

        _fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"projection": map_proj})

        predictions_da.isel(time=m).plot(
            ax=ax, x="x", y="y", transform=ccrs.SouthPolarStereo(), add_colorbar=True, cmap="RdBu_r"
        )
        ax.coastlines("10m")
        ax.gridlines(draw_labels=False)
        ax.set_title(f"Predicted SMB — {month_name} {start_year}", fontsize=10)
        plt.savefig(Path(savedir) / f"predicted_smb_month_{start_year}_{m}.png", dpi=300, bbox_inches="tight")
        plt.close()
    return
