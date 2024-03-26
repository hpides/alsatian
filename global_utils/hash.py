import hashlib

import torch


def get_device(device):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device


def state_dict_hash(state_dict: dict) -> str:
    """
    Calculates a md5 hash of a state dict dependent on the layer names and the corresponding weight tensors.
    :param state_dict: The state dict to create the hash from.
    :return: The md5 hash as a string.
    """
    md5 = hashlib.md5()

    for layer_name, weight_tensor in state_dict.items():
        weight_tensor = weight_tensor.to('cpu')
        numpy_data = weight_tensor.numpy().data
        md5.update(bytes(layer_name, 'utf-8'))
        md5.update(numpy_data)

    return md5.hexdigest()


def tensor_hash(tensor: torch.tensor, device: torch.device = None) -> str:
    """
    Calculates a md5 hash of the given tensor.
    :param tensor: The tensor to hash.
    :param device: The device to execute on.
    :return: The md5 hash as a string.
    """
    md5 = hashlib.md5()

    device = get_device(device)

    tensor = tensor.to(device)
    numpy_data = tensor.detach().numpy().data
    md5.update(numpy_data)

    return md5.hexdigest()

def architecture_hash(model: torch.nn.Module) -> str:
    md5 = hashlib.md5()
    md5.update(bytes(repr(model), 'utf-8'))

    return md5.hexdigest()