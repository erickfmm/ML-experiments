import torch
import yaml


def save(model, model_filename: str, weights_filename: str):
    """Save model architecture as YAML and weights as state_dict file."""
    # Serialize architecture to YAML
    model_yaml = str(model)
    with open(model_filename, "w") as yaml_file:
        yaml_file.write(model_yaml)
    # Serialize weights
    torch.save(model.state_dict(), weights_filename)
    print("Saved model to disk")
    return model


def load(model_cls, model_filename: str, weights_filename: str, *args, **kwargs):
    """Load model from YAML architecture description and state_dict weights.
    
    Parameters
    ----------
    model_cls : type
        The model class to instantiate.
    model_filename : str
        Path to the YAML architecture file.
    weights_filename : str
        Path to the saved state_dict file.
    *args, **kwargs :
        Arguments passed to model_cls() constructor.
    """
    # Load model architecture info (for reference)
    with open(model_filename, "r") as yaml_file:
        _ = yaml_file.read()
    # Create model instance
    model = model_cls(*args, **kwargs)
    # Load weights
    model.load_state_dict(torch.load(weights_filename, weights_only=True))
    model.eval()
    print("Loaded model from disk")
    return model
