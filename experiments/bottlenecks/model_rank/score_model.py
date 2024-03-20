import os
import time

import torch

from custom.data_loaders.custom_image_folder import CustomImageFolder
from custom.dataset_transfroms import imagenet_inference_transform
from custom.models.init_models import initialize_model
from experiments.bottlenecks.model_rank.experiment_args import ExpArgs
from global_utils.benchmark_util import Benchmarker
from global_utils.constants import MODEL_TO_DEVICE, STATE_TO_MODEL, DATA_TO_DEVICE, LOAD_DATA, CUDA, INFERENCE, \
    CALC_PROXY_SCORE, STATE_DICT_SIZE, PARTIAL_STATE_DICT_SIZE, END_TO_END, END_DATA_LOAD, END_EXTRACT_FEATURES
from global_utils.device import get_device
from global_utils.dummy_dataset import get_input_shape, DummyDataset
from global_utils.model_operations import split_model, get_split_index
from global_utils.size import state_dict_size_mb
from search.proxies.nn_proxy import linear_proxy


def score_model_exp(exp_args: ExpArgs):
    results = {}

    device = get_device(exp_args.device)
    bench = Benchmarker(device)

    # take full model first
    model = initialize_model(exp_args.model_name, pretrained=True, features_only=True)
    results[STATE_DICT_SIZE] = state_dict_size_mb(model.state_dict())
    # if split param is given split model

    split_index = get_split_index(exp_args.split_level, exp_args.model_name)
    if split_index is not None:
        initial_model = model
        unused_model, model = split_model(model, split_index)
    state_dict = model.state_dict()
    results[PARTIAL_STATE_DICT_SIZE] = state_dict_size_mb(state_dict)

    if exp_args.dataset_type in ['image_folder', 'imagenette']:
        data_set = CustomImageFolder(os.path.join(exp_args.dataset_path, 'train'), imagenet_inference_transform)
        # artificially making the dataset smaller
        data_set.set_subrange(0, exp_args.num_items)
    elif exp_args.dataset_type == 'imagenette_preprocessed_ssd':
        data_set = DummyDataset(exp_args.num_items, (3, 224, 224), (1,), exp_args.dummy_input_dir)
    else:
        raise NotImplementedError(f'the dataset type {exp_args.dataset_type} is currently not supported')

    # in case we split the model, the model resulting out of the split will not have the same input shape as the
    # original dataset thus we create a new dummy dataset having items of the correct shape
    if split_index is not None and split_index > 0:
        # first get the item shape of the original data without the batch size
        input_shape = data_set[0][0].shape
        model_input_shape = get_input_shape(split_index, initial_model, exp_args.num_items, input_shape)
        label_shape = data_set[0][1].shape if torch.is_tensor(data_set[0][1]) else (1,)
        data_set = DummyDataset(exp_args.num_items, model_input_shape, label_shape, exp_args.dummy_input_dir)

    data_loader = torch.utils.data.DataLoader(data_set, batch_size=exp_args.extract_batch_size, shuffle=False,
                                              num_workers=exp_args.data_workers)

    bench.warm_up_gpu()
    end_to_end, (detailed_measurements, detailed_bench) = bench.benchmark_end_to_end(
        _score_model, model, state_dict, data_loader, device)

    detailed_bench.sync_and_summarize_tasks()
    detailed_measurements[DATA_TO_DEVICE] = detailed_bench.get_task_times(DATA_TO_DEVICE)
    detailed_measurements[INFERENCE] = detailed_bench.get_task_times(INFERENCE)

    results.update(detailed_measurements)
    results[END_TO_END] = end_to_end

    return results


def _score_model(model, state_dict, data, device):
    bench = Benchmarker(device)
    measurements = {}
    # assume we already have an initialized model
    # so, we just have to
    # load the model on the device (GPU)
    measurement, model = bench.micro_benchmark(_load_model_to_device, model, device)
    measurements[MODEL_TO_DEVICE] = measurement
    # load the state dict into the model
    measurement, _ = bench.micro_benchmark(_load_state_dict_in_model, model, state_dict)
    measurements[STATE_TO_MODEL] = measurement

    batch_measures = {
        LOAD_DATA: [],
        DATA_TO_DEVICE: [],
        INFERENCE: []
    }

    bench.add_task(DATA_TO_DEVICE)
    bench.add_task(INFERENCE)

    features = []
    labels = []

    # need to backprop here ...
    with torch.no_grad():
        start = time.perf_counter()
        end_to_end_start = time.perf_counter()

        for i, (inputs, l) in enumerate(data):
            # measure data loading time
            batch_measures[LOAD_DATA].append(time.perf_counter() - start)

            if i == len(data) - 1:
                measurements[END_DATA_LOAD] = time.perf_counter() - end_to_end_start

            bench.register_cuda_start(DATA_TO_DEVICE, i)
            batch = _load_data_to_device(inputs, device)
            bench.register_cuda_end(DATA_TO_DEVICE, i)

            bench.register_cuda_start(INFERENCE, i)
            out = _inference(batch, model)
            bench.register_cuda_end(INFERENCE, i)

            features.append(out)
            labels.append(l)

            start = time.perf_counter()

    torch.cuda.synchronize()
    measurements[END_EXTRACT_FEATURES] = time.perf_counter() - end_to_end_start

    # actually ranking the model with proxy score
    # since we only interested in the duration, but not in the actual ranking -> just use features for train and test
    # also assume 100 classes for now
    measurement, proxy_score = bench.benchmark_end_to_end(
        linear_proxy, features, labels, features, labels, 100, device
    )
    measurements[CALC_PROXY_SCORE] = measurement
    measurements.update(batch_measures)
    return measurements, bench


def _load_model_to_device(model, device):
    if device.type == CUDA:
        model = model.to(device)
    return model


def _load_data_to_device(batch, device):
    # if the device is GPU load, otherwise do nothing
    if device.type == CUDA:
        batch = batch.to(device)
    return batch


def _load_state_dict_in_model(model, state_dict):
    # to allow only a partial update of the state set strict=False
    model.load_state_dict(state_dict, strict=False)


def _inference(batch, model):
    model.eval()
    with torch.set_grad_enabled(False):
        out = model(batch)
        return out
