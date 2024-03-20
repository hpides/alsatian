from statistics import mean

import torch

from custom.models.init_models import initialize_model
from global_utils.benchmark_util import Benchmarker
from global_utils.model_names import RESNET_50
from global_utils.model_operations import get_split_index, split_model


def inference(first_model, second_models, random_batches):
    with torch.no_grad():
        first_model = first_model.to(device)
        second_models = [m.to(device) for m in second_models]
        for b in random_batches:
            b = b.to(device)
            x = first_model(b)
            for s_model in second_models:
                y = s_model(x)


if __name__ == '__main__':
    measurements = []
    for i in range(10):
        device = torch.device('cuda')
        second_models = []
        # here use the regular sequential resnet50
        model = initialize_model(RESNET_50, sequential_model=True, features_only=True)
        split_index = get_split_index(-3, RESNET_50)
        first_model, second = split_model(model, split_index)

        first_model.eval()
        second.eval()

        second_models.append(second)

        for _ in range(49):
            model = initialize_model(RESNET_50, sequential_model=True, features_only=True)
            _, second = split_model(model, split_index)
            second.eval()
            second_models.append(second)

        random_batches = [torch.rand(256, 3, 224, 224) for _ in range(10)]

        bench = Benchmarker(device)

        bench.warm_up_gpu()
        mes, out = bench.micro_benchmark(inference, first_model, second_models, random_batches)

        measurements.append(mes)

    print(measurements)
    print(f'mean: {mean(measurements)}')
