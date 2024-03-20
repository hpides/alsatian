import torch


def generate_random_state_dict(model):
    random_state_dict = {}
    for key, param in model.state_dict().items():
        if param.dtype == torch.long:
            random_state_dict[key] = torch.randint_like(param, low=1, high=100)
        else:
            random_state_dict[key] = torch.randn_like(param)
    return random_state_dict

def add_noise_to_state_dict(state_dict, noise_scale=1e-5):
    for key in state_dict.keys():
        noise = torch.randn_like(state_dict[key]) * noise_scale
        state_dict[key] += noise
