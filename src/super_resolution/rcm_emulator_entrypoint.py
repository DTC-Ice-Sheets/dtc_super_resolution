"""Entrypoint for the RCM emulator pipeline."""

import argparse
import argparse
import os
import logging

from super_resolution.monitoring import setup_logging
from super_resolution.download_data import download_data
from super_resolution.pre_processing_module.preprocessing import run_preprocessing
from super_resolution.ml_module.inference import run_inference
from super_resolution.ml_module.inference_helpers import run_postprocessing

from super_resolution.ml_module.plots import predictions_plot

# setup_logging()
logger = logging.getLogger(__name__)

RACMO_PATH = "/home/ngaillard/super_resolution/src/super_resolution/data/RACMO/RACMO__start_2020-01-01T00:00:00__end_2020-02-01T00:00:00__polygon_area_None__variables_surface_mass_balance_glaciated__mask_False__raw_False__21e0066e.zarr"


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on the RCM emulator.")
    parser.add_argument("--start_date", type=str, required=True, help="Start date for data download (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, required=True, help="End date for data download (YYYY-MM-DD)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to store downloaded data")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use for inference", 
                        choices=["model_unet", "model_attention_unet"])
    return parser.parse_args()


def main() -> None:
    """
    High-level pipeline for the RCM emulator. This function orchestrates the entire process, 
    from data download to inference and saving the final output.
    """

    # Parse the input arguments
    args = parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 1. LOAD DATA
    logger.info("Downloading ERA5 and RCM data from Data Download Module for the period %s to %s...",
                args.start_date,
                args.end_date)
    
    # data_paths = download_data(args.start_date, args.end_date, args.output_dir)
    data_paths = {"era5": os.path.join(args.output_dir, "ERA5_monthly_means_2010-2023.nc"), "rcm": RACMO_PATH}

    # 2. PRE-PROCESSING
    logger.info("Pre-processing the downloaded data...")
    input_patches, metadata, tiles, rcm_ds, gcm_ds = run_preprocessing(data_paths)

    # 3. PREDICTION
    logger.info("Running inference on the RCM emulator model...")
    monthly_predictions = run_inference(args.model_name, input_patches, metadata)

    # 4. POST PROCESSING
    final_output_da = run_postprocessing(monthly_predictions, metadata, tiles, rcm_ds, gcm_ds)

    # 5. SAVING OUTPUT
    logger.info("Saving the final output to Zarr format...")
    outdir = args.output_dir + "/output/"
    output_zarr_path = os.path.join(outdir, f"predicted_smb_{args.start_date}_{args.end_date}.zarr")
    final_output_da.to_zarr(output_zarr_path, mode="w")

    # save plots
    predictions_plot(final_output_da, month=3, args=args)

    logger.info("Saved the final output to %s", output_zarr_path)
    return


if __name__ == "__main__":  
    main()

