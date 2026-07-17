import yaml
import torch

from super_resolution.ml_module.inference_helpers import load_model, predict


def run_inference(model_name: str, 
            input_patches: list, 
            metadata: list) -> dict:
    """
    Run inference on the RCM emulator model.

    Parameters:
    -----------
    model_path: str
        Path to the trained model file.
    input_patches: list
        A list of numpy arrays containing preprocessed input patches for inference.
    metadata: list
        A list of dictionaries containing metadata for each input patch.
    model_type: str
        The type of model to use for inference.

    Returns:
    --------
        dict: A dictionary containing the model's predictions.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the model config
    with open("./ml_module/models_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        try:
            model_config = config[model_name]
        except KeyError:
            raise ValueError(f"Model name '{model_name}' not found in the configuration file.")

    # instantiate the model and load the weights
    model = load_model(model_config)

    # Create a DataLoader for the input tiles
    monthly_predictions = predict(model, input_patches, metadata, device=device, config=model_config)

    return monthly_predictions
