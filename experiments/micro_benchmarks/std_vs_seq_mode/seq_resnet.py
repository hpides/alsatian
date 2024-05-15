from statistics import mean

import torch

from custom.models.init_models import initialize_model
from global_utils.benchmark_util import Benchmarker
from global_utils.model_names import RESNET_50
from global_utils.model_operations import count_parameters


def inference(model, random_batches):
    with torch.no_grad():
        model.to(device)
        for b in random_batches:
            b = b.to(device)
            _output = model(b)


if __name__ == '__main__':
    measurements = []
    for i in range(10):
        device = torch.device('cuda')
        model = initialize_model(RESNET_50, sequential_model=True)
        model.eval()

        random_batches = [torch.rand(256, 3, 224, 224) for _ in range(30)]

        bench = Benchmarker(device)

        bench.warm_up_gpu()
        mes, out = bench.micro_benchmark(inference, model, random_batches)
        measurements.append(mes)

    print(measurements)
    print(f'mean: {mean(measurements)}')
    print(f'num_params: {count_parameters(model)}')
