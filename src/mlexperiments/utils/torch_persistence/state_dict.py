import torch


def save(model, filename: str):
    """Save model's state_dict to a file."""
    torch.save(model.state_dict(), filename)
    print("Saved model to disk")
    return model


def load(model_cls, filename: str, *args, **kwargs):
    """Load model's state_dict from a file.
    
    Parameters
    ----------
    model_cls : type
        The model class to instantiate.
    filename : str
        Path to the saved state_dict file.
    *args, **kwargs :
        Arguments passed to model_cls() constructor.
    """
    model = model_cls(*args, **kwargs)
    model.load_state_dict(torch.load(filename, weights_only=True))
    model.eval()
    print("Loaded model from disk")
    return model
