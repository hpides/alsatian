from statistics import mean

import torch

from custom.models.init_models import initialize_model
from global_utils.benchmark_util import Benchmarker
from global_utils.model_names import RESNET_50
from global_utils.model_operations import get_split_index, merge_n_models, count_parameters

TINY = 'tiny'


def inference(model, random_batches):
    out = []
    with torch.no_grad():
        model = model.to(device)
        for b in random_batches:
            b = b.to(device)
            x = model(b)
            out.append(x)

    return out


def get_tiny_model():
    return torch.nn.Sequential(
        torch.nn.Linear(10, 10),
        torch.nn.Linear(10, 10)
    )


if __name__ == '__main__':
    # model_type = RESNET_50
    model_type = TINY

    measurements = []
    for i in range(10):
        device = torch.device('cuda')
        second_models = []
        if model_type == RESNET_50:
            # here use the regular sequential resnet50
            models = [initialize_model(RESNET_50, sequential_model=True, features_only=True) for _ in range(50)]
            split_indices = [get_split_index(-3, RESNET_50)] * 49

            random_batches = [torch.rand(256, 3, 224, 224) for _ in range(10)]
        elif model_type == TINY:
            models = [get_tiny_model() for _ in range(50)]
            split_indices = [1] * 49

            random_batches = [torch.rand(10, 10) for _ in range(10)]
        else:
            raise ValueError

        merged_model = merge_n_models(models, split_indices)

        merged_model.eval()
        print(count_parameters(merged_model))

        bench = Benchmarker(device)

        bench.warm_up_gpu()
        mes, out = bench.micro_benchmark(inference, merged_model, random_batches)

        measurements.append(mes)

    print(measurements)
    print(f'mean: {mean(measurements)}')
