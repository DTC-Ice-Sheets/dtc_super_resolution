"""Script to run the full pipeline locally."""

import logging
import pathlib
from datetime import datetime
from typing import Annotated, Literal

import cyclopts
from download_data import download_data
from plots import predictions_plot

from super_resolution.ml_module.inference import run_inference
from super_resolution.ml_module.inference_helpers import run_postprocessing
from super_resolution.monitoring import setup_logging
from super_resolution.pre_processing_module.preprocessing import run_preprocessing

setup_logging()
logger = logging.getLogger(__name__)

app = cyclopts.App(help="Run inference on the RCM emulator.")


@app.default
def main(
    start_date: Annotated[datetime, cyclopts.Parameter(help="Start date for data download (YYYY-MM-DD)")],
    end_date: Annotated[datetime, cyclopts.Parameter(help="End date for data download (YYYY-MM-DD)")],
    output_dir: Annotated[str, cyclopts.Parameter(help="Directory to store downloaded data")],
    model_name: Annotated[
        Literal["model_baseline_unet", "model_attention_unet"],
        cyclopts.Parameter(help="Name of the model to use for inference"),
    ],
) -> None:
    """
    Run the high-level RCM emulator pipeline.

    This function orchestrates the entire process,
    from data download to inference and saving the final output.
    """
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

    # 1. LOAD DATA
    logger.info(
        "Downloading ERA5 and RCM data from Data Download Module for the period %s to %s...",
        start_date,
        end_date,
    )

    data_paths = download_data(start_date, end_date, output_dir)

    # 2. PRE-PROCESSING - PREDICTION - POST PROCESSING are done by the RCM emulator module
    logger.info("Running the RCM emulator...")
    input_patches, metadata, tiles, rcm_ds, gcm_ds = run_preprocessing(data_paths["era5"], data_paths["racmo"])

    # 3. PREDICTION
    logger.info("Running inference on the RCM emulator model...")
    monthly_predictions = run_inference(model_name, input_patches, metadata)

    # 4. POST PROCESSING
    final_output_da = run_postprocessing(monthly_predictions, metadata, tiles, rcm_ds, gcm_ds)

    # 5. SAVING OUTPUT
    logger.info("Saving the final output to Zarr format...")
    outdir = output_dir + "/output/"
    output_zarr_path = (
        pathlib.Path(outdir) / f"predicted_smb_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}.zarr"
    )
    final_output_da.to_zarr(output_zarr_path, mode="w")

    # save plots
    logger.info("Saving plots of the predictions...")

    predictions_plot(final_output_da, start_date=start_date, savedir=output_dir)

    logger.info("Saved the final output to %s", output_zarr_path)


if __name__ == "__main__":
    app()
