from mlagents.torch_utils import torch, nn, default_device



def has_nan(x):
    """
    Returns true if there are any NaNs in the tensor, false otherwise.
    """
    return torch.isnan(x).any()