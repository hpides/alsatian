import random
from statistics import mean

import torch

from custom.models.init_models import initialize_model
from global_utils.benchmark_util import Benchmarker
from global_utils.model_names import RESNET_50
from global_utils.model_operations import get_split_index, merge_models


def merge_n_models(models, models_indices):
    merged_model = models[0]
    for i, si in enumerate(models_indices):
        merged_model = merge_models(merged_model, models[i + 1], si)


if __name__ == '__main__':
    measurements = []
    for i in range(10):
        random.seed(42)
        models = [initialize_model(RESNET_50, sequential_model=True, features_only=True) for _ in range(50)]
        bench = Benchmarker(torch.device('cpu'))
        split_levels = sorted([random.randint(-15, -1) for _ in range(49)], reverse=True)
        split_indices = [get_split_index(i, RESNET_50) for i in split_levels]

        mes, merge = bench.micro_benchmark(merge_n_models, models,split_indices)
        measurements.append(mes)
    print(measurements)
    print(f'mean: {mean(measurements)}')
