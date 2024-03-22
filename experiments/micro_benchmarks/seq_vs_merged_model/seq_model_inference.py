from statistics import mean

import torch

from custom.models.init_models import initialize_model
from experiments.micro_benchmarks.seq_vs_merged_model.merged_model_inf import TINY, get_tiny_model
from global_utils.benchmark_util import Benchmarker
from global_utils.model_names import RESNET_50
from global_utils.model_operations import get_split_index, split_model, count_parameters


def inference(first_model, second_models, random_batches):
    with torch.no_grad():
        first_model = first_model.to(device)
        second_models = [m.to(device) for m in second_models]
        out = []
        for b in random_batches:
            b = b.to(device)
            x = first_model(b)
            for s_model in second_models:
                y = s_model(x)
                out.append(y)
    return out


if __name__ == '__main__':
    # model_type = RESNET_50
    model_type = TINY

    measurements = []
    param_count = 0
    for i in range(10):
        device = torch.device('cuda')
        second_models = []
        if model_type == RESNET_50:
            # here use the regular sequential resnet50
            model = initialize_model(RESNET_50, sequential_model=True, features_only=True)
            split_index = get_split_index(-3, RESNET_50)
        else:
            model = get_tiny_model()
            split_index = 1

        first_model, second = split_model(model, split_index)

        first_model.eval()
        second.eval()

        second_models.append(second)

        for _ in range(49):
            if model_type == RESNET_50:
                model = initialize_model(RESNET_50, sequential_model=True, features_only=True)
                random_batches = [torch.rand(256, 3, 224, 224) for _ in range(10)]
            else:
                model = get_tiny_model()
                random_batches = [torch.rand(10, 10) for _ in range(10)]

            _, second = split_model(model, split_index)
            second.eval()
            second_models.append(second)

        bench = Benchmarker(device)

        param_count += count_parameters(first_model)
        for sm in second_models:
            param_count += count_parameters(sm)

        bench.warm_up_gpu()
        mes, out = bench.micro_benchmark(inference, first_model, second_models, random_batches)

        measurements.append(mes)

    print(measurements)
    print(f'mean: {mean(measurements)}')
    print(f'num_params: {param_count}')
