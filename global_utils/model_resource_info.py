import torch
from torch.utils.data import DataLoader

from custom.models.init_models import initialize_model
from custom.models.split_indices import SPLIT_INDEXES
from global_utils.benchmark_util import Benchmarker
from global_utils.constants import NUM_PARAMS, CUDA, OUTPUT_SHAPE, GPU_INF_TIMES, OUTPUT_SIZE, NUM_PARAMS_MB, \
    OUTPUT_SIZE_MB, CPU
from global_utils.dummy_dataset import DummyDataset
from global_utils.hash import architecture_hash
from global_utils.model_names import RESNET_50
from global_utils.model_operations import split_model
from global_utils.size import number_parameters

INF = 10000


def _inference(model, batch, device):
    batch = batch.to(device)

    with torch.no_grad():
        output = model(batch)

    return output


def layer_output_sizes(model, input_shape: [int]):
    result = {}

    prev_out = torch.randn(size=[1] + input_shape, dtype=torch.float)
    for layer in model:
        out = layer(prev_out)
        output_size = _multiply_list(out.shape)
        result[architecture_hash(layer)] = output_size
        prev_out = out

    return result


def model_resource_info(model: torch.nn.Module, split_indices: [int], input_shape: [int], batch_size=32,
                        inference_time=False, add_mb_info=True):
    device = torch.device(CUDA if torch.cuda.is_available() else CPU)
    benchmarker = Benchmarker(device)
    sorted_split_indices = sorted(split_indices) + [INF]
    prev_number_params = 0
    prev_inf_times = None
    result = {}
    # measure all values incrementally by increasing the model size and then taking diffs
    for idx in sorted_split_indices:
        result[idx] = {}

        # get part of model we want to benchmark
        bench_model = _get_bench_model(idx, model)

        # get the number of parameters
        number_of_params = number_parameters(bench_model.state_dict())
        result[idx][NUM_PARAMS] = number_of_params - prev_number_params
        prev_number_params = number_of_params

        # measure inference time on GPU
        inf_times = []
        output = None

        # define the amount of data used
        number_items = _number_of_items(batch_size, inference_time)
        dummy_data = DummyDataset(number_items, input_shape, [1], "")
        data_loader = DataLoader(dummy_data, batch_size=batch_size, shuffle=False)

        bench_model.eval()
        bench_model.to(device)
        for inp, _ in data_loader:
            msr, output = benchmarker.micro_benchmark(_inference, bench_model, inp, device)
            inf_times.append(msr)

        result[idx][OUTPUT_SHAPE] = list(output.shape)
        result[idx][OUTPUT_SIZE] = _multiply_list(list(output.shape))

        if inference_time:
            # trow away first three measurements and take the last 5
            inf_times = inf_times[3:]
            if prev_inf_times is None:
                result[idx][GPU_INF_TIMES] = inf_times
            else:
                result[idx][GPU_INF_TIMES] = list(map(float.__sub__, inf_times, prev_inf_times))
            prev_inf_times = inf_times

        if add_mb_info:
            result[idx][NUM_PARAMS_MB] = result[idx][NUM_PARAMS] * 4 * 10 ** -6
        result[idx][OUTPUT_SIZE_MB] = result[idx][OUTPUT_SIZE] * 4 * 10 ** -6

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
