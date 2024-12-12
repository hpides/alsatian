import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from custom.data_loaders.custom_image_folder import CustomImageFolder
from custom.dataset_transfroms import imagenet_inference_transform, imagenet_data_transforms
from custom.models.init_models import initialize_model
from global_utils.model_names import RESNET_50
from global_utils.model_operations import split_model, merge_models, split_model_in_two


def measure(surgical_fine_tuning, fine_tuning_start_idx, fine_tuning_end_idx, regular_fine_tuning, freeze_end_idx,
            caching):
    # fixed parameters
    num_epochs = 5
    batch_size = 128
    learning_rate = 0.0001
    num_workers = 12
    model_name = RESNET_50
    training_samples = 2000
    test_samples = 500
    dataset_path = "/mount-ssd/data/imagenette2"
    log_dir = "/mount-ssd/results/fine-tune-time"

    if regular_fine_tuning:
        log_file_name = f'finetune_{freeze_end_idx}_{caching}.txt'
    else:
        log_file_name = f'surg_finetune_{fine_tuning_start_idx}_{fine_tuning_end_idx}_{caching}.txt'

    log_file_path = os.path.join(log_dir, log_file_name)

    # get model
    model = initialize_model(model_name, pretrained=True, sequential_model=True)

    assert not (surgical_fine_tuning and regular_fine_tuning)

    if surgical_fine_tuning:
        # split model individual parts
        model_parts = split_model(model, [fine_tuning_start_idx, fine_tuning_end_idx])
        freeze_parts = [model_parts[0], model_parts[2]]

    elif regular_fine_tuning:
        model_parts = split_model_in_two(model, freeze_end_idx)
        freeze_parts = [model_parts[0]]

    else:
        raise NotImplementedError

    for freeze_part in freeze_parts:
        for param in freeze_part.parameters():
            param.requires_grad = False

    for freeze_part in [model_parts[1]]:
        for param in freeze_part.parameters():
            param.requires_grad = True

    # join model parts to one model and load to GPU
    model = merge_models(model_parts)
    model = model.to('cuda')

    def check_frozen_layers(model):
        for name, param in model.named_parameters():
            layer_status = "Frozen" if not param.requires_grad else "Trainable"
            print(f"Layer: {name} - Status: {layer_status}")

    check_frozen_layers(model)

    # use imagenette dataset
    train_dataset = CustomImageFolder(os.path.join(dataset_path, 'train'), imagenet_data_transforms['train'])
    train_dataset.set_subrange(0, training_samples)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = CustomImageFolder(os.path.join(dataset_path, 'val'), imagenet_inference_transform)
    test_dataset.set_subrange(0, test_samples)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define loss and optimizer (only for parameters that require gradients)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    def evaluate_model(model, data_loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        return 100 * correct / total

    # Training loop
    with open(log_file_path, 'w') as log_file:
        for epoch in range(num_epochs):
            model.train()
            epoch_start_time = time.time()

            # Initialize counters for training accuracy
            correct_train = 0
            total_train = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')  # Move to GPU if available
                model = model.to('cuda')  # Ensure the model is on the correct device

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Calculate accuracy for this batch
                _, predicted = torch.max(outputs, 1)
                correct_train += (predicted == labels).sum().item()
                total_train += labels.size(0)

            # Calculate epoch time and training accuracy
            epoch_time = time.time() - epoch_start_time
            train_accuracy = 100 * correct_train / total_train

            # Evaluate on test set
            test_accuracy = evaluate_model(model, test_loader)

            # Log epoch time, training accuracy, and test accuracy
            log_file.write(f'Epoch {epoch + 1}, Time: {epoch_time:.2f} seconds, '
                           f'Train Accuracy: {train_accuracy:.2f}%, '
                           f'Test Accuracy: {test_accuracy:.2f}%\n')
            print(f'Epoch {epoch + 1}, Time: {epoch_time:.2f} seconds, '
                  f'Train Accuracy: {train_accuracy:.2f}%, '
                  f'Test Accuracy: {test_accuracy:.2f}%')

    print("Training completed and times logged.")
    print()
    print()
    print()


if __name__ == '__main__':
    # changing parameters
    surgical_fine_tuning = False
    fine_tuning_start_idx = 13
    fine_tuning_end_idx = 16

    regular_fine_tuning = True
    freeze_end_idx = 13

    caching = False

    # traditional fine tuning
    split_indexes = [7, 11, 17, 20]
    # full fine tune
    measure(False, 0, 0, True, 0, False)
    # partial fine tune
    for split_index in split_indexes:
        measure(False, 0, 0, True, split_index, False)

    # surgical fine-tuning, only middel layers, only last is already covered above
    split_indexes = [0, 7, 11, 17, 20]
    for i in range(len(split_indexes) - 1):
        measure(True, split_indexes[i], split_indexes[i + 1], False, 0, False)
