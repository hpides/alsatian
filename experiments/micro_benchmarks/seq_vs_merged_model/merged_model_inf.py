from statistics import mean

import torch

from custom.models.init_models import initialize_model
from global_utils.benchmark_util import Benchmarker
from global_utils.model_names import RESNET_50
from global_utils.model_operations import get_split_index, merge_n_models, count_parameters


def inference(model, random_batches):
    with torch.no_grad():
        model = model.to(device)
        for b in random_batches:
            b = b.to(device)
            x = model(b)


if __name__ == '__main__':
    measurements = []
    for i in range(10):
        device = torch.device('cuda')
        second_models = []
        # here use the regular sequential resnet50
        models = [initialize_model(RESNET_50, sequential_model=True, features_only=True) for _ in range(50)]
        split_indices = [get_split_index(-3, RESNET_50)] * 49
        merged_model = merge_n_models(models, split_indices)

        merged_model.eval()
        print(count_parameters(merged_model))

        bench = Benchmarker(device)

        random_batches = [torch.rand(256, 3, 224, 224) for _ in range(10)]

        bench.warm_up_gpu()
        mes, out = bench.micro_benchmark(inference, merged_model, random_batches)

        measurements.append(mes)

    print(measurements)
    print(f'mean: {mean(measurements)}')
