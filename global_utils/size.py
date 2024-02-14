def number_parameters(state_dict):
    total_params = 0
    for key, value in state_dict.items():
        total_params += value.numel()  # Get the number of elements in the tensor
    return total_params


def state_dict_size_mb(state_dict):
    num_params = number_parameters(state_dict)
    bytes = num_params * 4
    return bytes / 10 ** 6
