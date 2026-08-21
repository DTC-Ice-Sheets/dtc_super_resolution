"""Script to run inference on the RCM emulator model."""

import pathlib

import torch
import yaml

from super_resolution.rcm_emulator.ml_module.inference_helpers import load_model, predict


def run_inference(model_name: str, input_patches: list, metadata: list) -> dict:
    """
    Run inference on the RCM emulator model.

    Parameters
    ----------
    model_name: str
        The name of model to use for inference.
    input_patches: list
        A list of numpy arrays containing preprocessed input patches for inference.
    metadata: list
        A list of dictionaries containing metadata for each input patch.

    Returns
    -------
    dict
        A dictionary containing the model's predictions.

    Raises
    ------
    ValueError
        If the specified model name is not found in the configuration file.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the model config
    config_path = pathlib.Path(__file__).parent / "models_config.yaml"
    with config_path.open("r") as f:
        config = yaml.safe_load(f)
        try:
            model_config = config[model_name]
        except KeyError as err:
            raise ValueError(f"Model name '{model_name}' not found in the configuration file.") from err

    # instantiate the model and load the weights
    model = load_model(model_config)

    # Create a DataLoader for the input tiles
    return predict(model, input_patches, metadata, device=device, config=model_config)
