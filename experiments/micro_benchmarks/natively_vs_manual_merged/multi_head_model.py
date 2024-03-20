import torch
import torch.nn as nn
from torch import Tensor

from global_utils.benchmark_util import Benchmarker
from global_utils.model_operations import count_parameters, merge_n_models


# Define a custom module for grouped convolutional layer followed by ReLU
class Conv2dReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv2dReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class MultiHeadModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Sequential(
            Conv2dReLU(3, 32, kernel_size=10, padding=1),
            Conv2dReLU(32, 32, kernel_size=10, padding=1),
            Conv2dReLU(32, 64, kernel_size=10, padding=1),

        )

        head_list = []
        for _ in range(50):
            head = nn.Sequential(
                Conv2dReLU(64, 128, kernel_size=10, padding=1),
            )
            head_list.append(head)
        self.heads = nn.ModuleList(head_list)

    def forward(self, x: Tensor) -> Tensor:
        x = self.base(x)
        out = []
        for output_head in self.heads:
            x_i = output_head(x)
            out.append(x_i)
        return out


def inference(model, random_batches):
    device = 'cuda'
    with torch.no_grad():
        model.to(device)
        for b in random_batches:
            b = b.to(device)
            _output = model(b)


if __name__ == '__main__':
    nat_merged = []
    meth_merged = []
    for _ in range(10):
        bench = Benchmarker(torch.device('cuda'))
        # first the naturally merged model

        model = MultiHeadModel()
        print(f'num params: {count_parameters(model)}')
        random_batches = [torch.rand(10, 3, 224, 224) for _ in range(10)]
        bench.warm_up_gpu()
        measurement, out = bench.micro_benchmark(inference, model, random_batches)
        nat_merged.append(measurement)

        # second the model merged using our merge method
        torch.cuda.empty_cache()

        models = []
        for _ in range(50):
            m = nn.Sequential(
                Conv2dReLU(3, 32, kernel_size=10, padding=1),
                Conv2dReLU(32, 32, kernel_size=10, padding=1),
                Conv2dReLU(32, 64, kernel_size=10, padding=1),
                Conv2dReLU(64, 128, kernel_size=10, padding=1)
            )
            models.append(m)
        split_indices = [3] * 49
        merged_model = merge_n_models(models, split_indices)

        print(f'num params: {count_parameters(merged_model)}')
        random_batches = [torch.rand(10, 3, 224, 224) for _ in range(10)]
        bench.warm_up_gpu()
        measurement, out = bench.micro_benchmark(inference, model, random_batches)
        meth_merged.append(measurement)

    print(f'nat merged: {nat_merged}')
    print(f'meth merged: {meth_merged}')

