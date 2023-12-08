import time

import torch

from global_utils.global_constants import TIME_DATA_LOADING, TIME, VAL, TRAIN


def _calc_features(model, dataloaders, device, repetitions=5):
    # perform multiple iterations because first iteration is usually to warm up GPU

    measurements = {}

    model = model.to(device)

    for rep in range(repetitions):
        print(f'Repetition {rep}/{repetitions - 1}')
        measurements[rep] = {}

        # Each epoch has a training and validation phase
        for phase in [TRAIN, VAL]:
            measurements[rep][phase] = {}
            measurements[rep][phase][TIME_DATA_LOADING] = []
            start = time.time_ns()

            # we only want to extract feature so regardless of the phase always eval
            model.eval()  # Set model to evaluate mode

            # Iterate over data.
            data_load_start = time.time_ns()

            for inputs, _ in dataloaders[phase]:
                measurements[rep][phase][TIME_DATA_LOADING].append(time.time_ns() - data_load_start)
                inputs = inputs.to(device)
                # forward
                # track history if only in train
                # we only want to extract feature so always no grad enabled
                with torch.set_grad_enabled(False):
                    outputs = model(inputs)

                data_load_start = time.time_ns()

            measurements[rep][phase][TIME] = time.time_ns() - start

    return model, measurements


def calc_features(model: torch.nn.Module, datasets: dict, batch_size, device, repetitions=5):
    dataloaders = {}
    dataset_sizes = {}
    for dataset_name, dataset in datasets.items():
        dataloaders[dataset_name] = \
            torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        dataset_sizes[dataset_name] = len(dataset)

    return _calc_features(model, dataloaders, device, repetitions)
