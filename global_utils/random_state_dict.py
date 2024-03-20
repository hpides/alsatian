import torch


def generate_random_state_dict(model):
    random_state_dict = {}
    for key, param in model.state_dict().items():
        if param.dtype == torch.long:
            random_state_dict[key] = torch.randint_like(param, low=-100, high=100)
        else:
            random_state_dict[key] = torch.randn_like(param)
    return random_state_dict
