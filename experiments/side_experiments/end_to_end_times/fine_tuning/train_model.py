import time

import torch
from torch import optim
from torch.optim import lr_scheduler

from global_utils.constants import TIME_DATA_LOADING


def train_model(model, criterion, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes, device):
    measurements = {}

    model = model.to(device)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        measurements[epoch] = {}

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            measurements[epoch][phase] = {}
            measurements[epoch][phase][TIME_DATA_LOADING] = []
            start = time.time_ns()
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            data_load_start = time.time_ns()

            for inputs, labels in dataloaders[phase]:
                measurements[epoch][phase][TIME_DATA_LOADING].append(time.time_ns() - data_load_start)
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                data_load_start = time.time_ns()

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            measurements[epoch][phase]['time'] = time.time_ns() - start
            measurements[epoch][phase]['acc'] = f'{epoch_acc:.4f}'
            measurements[epoch][phase]['loss'] = epoch_loss

    return model, measurements


def standard_training(model: torch.nn.Module, datasets: dict, batch_size, device, num_epochs):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    dataloaders = {}
    dataset_sizes = {}
    for dataset_name, dataset in datasets.items():
        dataloaders[dataset_name] = \
            torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        dataset_sizes[dataset_name] = len(dataset)

    return train_model(model, criterion, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes, device)
