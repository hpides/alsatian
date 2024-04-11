import torch
from torch.utils.data import DataLoader

from custom.models.init_models import initialize_model
from custom.models.split_indices import SPLIT_INDEXES
from global_utils.benchmark_util import Benchmarker
from global_utils.constants import NUM_PARAMS, CUDA, OUTPUT_SHAPE, GPU_INF_TIMES, OUTPUT_SIZE, NUM_PARAMS_MB, \
    OUTPUT_SIZE_MB
from global_utils.dummy_dataset import DummyDataset
from global_utils.model_names import RESNET_50
from global_utils.model_operations import split_model
from global_utils.size import number_parameters

INF = 10000


def _inference(model, batch, device):
    batch = batch.to(device)

    with torch.no_grad():
        output = model(batch)

    return output


def model_resource_info(model: torch.nn.Module, split_indices: [int], input_shape: [int], batch_size=32,
                        inference_time=False, add_mb_info=True):
    benchmarker = Benchmarker(torch.device(CUDA))
    sorted_split_indices = sorted(split_indices) + [INF]
    prev_idx = None
    result = {}
    # measure all values incrementally by increasing the model size and then taking diffs
    for idx in sorted_split_indices:
        result[idx] = {}

        # get part of model we want to benchmark
        bench_model = _get_bench_model(idx, model)

        # get the number of parameters
        if prev_idx is None:
            result[idx][NUM_PARAMS] = number_parameters(bench_model.state_dict())
        else:
            result[idx][NUM_PARAMS] = number_parameters(bench_model.state_dict()) - result[prev_idx][NUM_PARAMS]

            # measure inference time on GPU
        inf_times = []
        output = None

        # define the amount of data used
        number_items = _number_of_items(batch_size, inference_time)
        dummy_data = DummyDataset(number_items, input_shape, [1], "")
        data_loader = DataLoader(dummy_data, batch_size=batch_size, shuffle=False)

        bench_model.eval()
        bench_model.to(CUDA)
        for inp, _ in data_loader:
            msr, output = benchmarker.micro_benchmark_gpu(_inference, bench_model, inp, CUDA)
            inf_times.append(msr)

        result[idx][OUTPUT_SHAPE] = list(output.shape)
        result[idx][OUTPUT_SIZE] = _multiply_list(list(output.shape))

        if inference_time:
            # trow away first three measurements and take the last 5
            if prev_idx is None:
                result[idx][GPU_INF_TIMES] = inf_times[3:]
            else:
                result[idx][GPU_INF_TIMES] = list(map(float.__sub__, inf_times[3:], result[prev_idx][GPU_INF_TIMES]))

        if add_mb_info:
            result[idx][NUM_PARAMS_MB] = result[idx][NUM_PARAMS] * 4 * 10 ** -6
            result[idx][OUTPUT_SIZE_MB] = result[idx][OUTPUT_SIZE] * 4 * 10 ** -6

        prev_idx = idx

    return result


def _get_bench_model(idx, model):
    if idx < INF:
        # we are always interested in the first half of the model
        bench_model, _ = split_model(model, idx)
    else:
        bench_model = model
    return bench_model


def _number_of_items(batch_size, inference_time):
    if inference_time:
        # make sure we have 8 batches: 3 warm up 5 measurement
        return 8 * batch_size
    else:
        # only need one batch
        return batch_size


def _multiply_list(values):
    result = 1
    for v in values:
        result = result * v
    return result


if __name__ == '__main__':
    model_name = RESNET_50
    model = initialize_model(model_name, pretrained=True, sequential_model=True)
    info = model_resource_info(model, SPLIT_INDEXES[model_name], [3, 224, 224], inference_time=True)
    print(info)
