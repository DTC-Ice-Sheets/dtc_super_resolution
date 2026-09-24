"""Entrypoint for the RCM emulator pipeline."""

import logging
from typing import Annotated, Literal

import cyclopts

from super_resolution.monitoring import setup_logging
from super_resolution.rcm_emulator.ml_module.inference import run_inference
from super_resolution.rcm_emulator.ml_module.inference_helpers import run_postprocessing_loc
from super_resolution.rcm_emulator.pre_processing_module.preprocessing import run_preprocessing_loc

setup_logging()
logger = logging.getLogger(__name__)

app = cyclopts.App(help="Run inference on the RCM emulator.")


@app.default
def main(
    gcm_zarr_path: Annotated[str, cyclopts.Parameter(help="Path to the GCM data")],
    rcm_zarr_path: Annotated[str, cyclopts.Parameter(help="Path to the RCM data")],
    location: Annotated[tuple[float, float], cyclopts.Parameter(help="Location coordinates (lat, lon)")],
    output_zarr_path: Annotated[str, cyclopts.Parameter(help="Directory to store downloaded data")],
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
    # 2. PRE-PROCESSING
    input_patches, patch_metadata, tile_meta, rcm_ds, gcm_ds, target_patches = run_preprocessing_loc(
        gcm_zarr_path, rcm_zarr_path, location
    )

    # 3. PREDICTION
    monthly_predictions = run_inference(model_name, input_patches, patch_metadata)

    # 4. POST PROCESSING
    final_output_da = run_postprocessing_loc(
        monthly_predictions, tile_meta, rcm_ds, gcm_ds, input_patches, target_patches
    )

    # 5. SAVING OUTPUT
    final_output_da.to_zarr(output_zarr_path, mode="w")

    return


if __name__ == "__main__":
    app()
