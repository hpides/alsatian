import torch


def get_device(device_str: str = None):
    if device_str is not None:
        return torch.device(device_str)
    else:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
