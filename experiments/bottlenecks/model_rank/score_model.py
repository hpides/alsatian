import os
import time

import torch

from custom.data_loaders.custom_image_folder import CustomImageFolder
from custom.dataset_transfroms import imagenet_inference_transform
from custom.models.init_models import initialize_model
from experiments.bottlenecks.model_rank.experiment_args import ExpArgs
from global_utils.benchmark_util import Benchmarker
from global_utils.constants import MODEL_TO_DEVICE, STATE_TO_MODEL, DATA_TO_DEVICE, LOAD_DATA, CUDA, INFERENCE, \
    CALC_PROXY_SCORE, STATE_DICT_SIZE, PARTIAL_STATE_DICT_SIZE
from global_utils.device import get_device
from global_utils.dummy_dataset import get_input_shape, DummyDataset
from global_utils.size import state_dict_size_mb
from global_utils.split_models import split_model, get_split_index
from search.proxies.nn_proxy import linear_proxy


def score_model_exp(exp_args: ExpArgs):
    results = {}

    device = get_device(exp_args.device)
    bench = Benchmarker(device)

    # take full model first
    model = initialize_model(exp_args.model_name, pretrained=True, features_only=True)
    results[STATE_DICT_SIZE] = state_dict_size_mb(model.state_dict())
    # if split param is given split model

    split_index = get_split_index(exp_args.split_level, model, exp_args.model_name)
    if split_index is not None:
        initial_model = model
        unused_model, model = split_model(model, split_index)
    state_dict = model.state_dict()
    results[PARTIAL_STATE_DICT_SIZE] = state_dict_size_mb(state_dict)

    if exp_args.dataset_type == 'image_folder':
        data_set = CustomImageFolder(os.path.join(exp_args.dataset_path, 'train'), imagenet_inference_transform)
        # artificially making the dataset smaller
        data_set.set_subrange(0, exp_args.num_items)
    else:
        raise NotImplementedError(f'the dataset type {exp_args.dataset_type} is currently not supported')

    # in case we split the model, the model resulting out of the split will not have the same input shape as the
    # original dataset thus we create a new dummy dataset having items of the correct shape
    if split_index is not None and split_index > 0:
        # first get the item shape of the original data without the batch size
        item_shape = data_set[0][0].shape
        model_input_shape = get_input_shape(split_index, initial_model, exp_args.extract_batch_size,
                                            exp_args.num_items, item_shape)
        data_set = DummyDataset(exp_args.extract_batch_size, exp_args.num_items, model_input_shape,
                                exp_args.dummy_input_dir)

    data_loader = torch.utils.data.DataLoader(data_set, batch_size=exp_args.extract_batch_size, shuffle=False,
                                              num_workers=exp_args.data_workers)

    end_to_end, detailed_measurements = bench.benchmark_cpu(
        _score_model, model, state_dict, data_loader, device, bench)

    results.update(detailed_measurements)
    results['end_to_end_time'] = end_to_end
    return results


def _score_model(model, state_dict, data, device, bench):
    measurements = {}
    # assume we already have an initialized model
    # so, we just have to
    # load the model on the device (GPU)
    measurement, model = bench.benchmark_cpu(_load_model_to_device, model, device)
    measurements[MODEL_TO_DEVICE] = measurement
    # load the state dict into the model
    measurement, _ = bench.benchmark_cpu(_load_state_dict_in_model, model, state_dict)
    measurements[STATE_TO_MODEL] = measurement

    batch_measures = {
        LOAD_DATA: [],
        DATA_TO_DEVICE: [],
        INFERENCE: []
    }

    features = []
    labels = []

    # need to backprop here ...
    with torch.no_grad():
        start = time.time_ns()

        for inputs, l in data:
            # measure data loading time
            batch_measures[LOAD_DATA].append(time.time_ns() - start)

            measurement, batch = bench.benchmark_cpu(_load_data_to_device, inputs, device)
            batch_measures[DATA_TO_DEVICE].append(measurement)

            measurement, out = bench.benchmark(_inference, batch, model)
            batch_measures[INFERENCE].append(measurement)

            features.append(out)
            labels.append(l)

    # actually ranking the model with proxy score
    # since we only interested in the duration, but not in the actual ranking -> just use features for train and test
    # also assume 100 classes for now
    measurement, proxy_score = bench.benchmark_cpu(
        linear_proxy, features, labels, features, labels, 100, device
    )
    measurements[CALC_PROXY_SCORE] = measurement

    measurements.update(batch_measures)
    return measurements


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
