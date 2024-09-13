import argparse
import os
import random
import string
import time

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Subset
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from custom.models.init_models import initialize_model
from custom.models.split_indices import SPLIT_INDEXES
from experiments.main_experiments.snapshots.synthetic.generate import TWENTY_FIVE_PERCENT
from experiments.main_experiments.snapshots.synthetic.retrain_distribution import normal_retrain_layer_dist_25
from global_utils.model_names import VISION_MODEL_CHOICES, RESNET_18, EFF_NET_V2_L, RESNET_152, VIT_L_32
from global_utils.model_operations import split_model_in_two
from global_utils.write_results import write_measurements_and_args_to_json_file

STANFORD_CARS = "stanford-cars"

STANFORD_DOGS = "stanford-dogs"

IMAGE_WOOF = "image-woof"

FOOD_101 = "food-101"

CUB_BIRDS_200 = "cub-birds-200"

DUMMY = "dummy"

IMAGENETTE = "imagenette"

NUM_CLASSES = "num_classes"


def set_seeds(seed_value):
    # Set seed for Python random module
    random.seed(seed_value)

    # Set seed for NumPy random generator
    np.random.seed(seed_value)

    # Set seed for PyTorch random generator
    torch.manual_seed(seed_value)

    # If using GPU, set seed for CUDA as well
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)


def generate_4_digit_id():
    characters = string.ascii_letters + string.digits
    return ''.join(random.choices(characters, k=4))


def subsample_random_items(dataset, sample_size):
    # Ensure the sample size does not exceed the dataset size
    if sample_size > len(dataset):
        raise ValueError("Sample size cannot be larger than the dataset size.")

    # Get random indices for the subsample
    random_indices = random.sample(range(len(dataset)), sample_size)

    # Create a subset of the dataset with the sampled indices
    subset = Subset(dataset, random_indices)
    return subset


def get_retrain_index(architecture_name, distribution):
    num_blocks = len(SPLIT_INDEXES[architecture_name])
    if distribution == TWENTY_FIVE_PERCENT:
        retrain_idx = random.choice(normal_retrain_layer_dist_25(num_blocks, 200))
    else:
        raise NotImplementedError

    return retrain_idx


def evaluate_model(model, loader, device, criterion):
    model.eval()
    batches, agg_loss, correct_top1, correct_top5, num_items = 0, 0, 0, 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            batches += 1
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            _, predicted = torch.topk(outputs, k=5, dim=1)
            num_items += labels.size(0)

            loss = criterion(outputs, labels)
            agg_loss += loss.item()
            correct_top1 += (predicted[:, 0] == labels).sum().item()
            correct_top5 += torch.sum(torch.eq(predicted, labels.view(-1, 1)).any(dim=1)).item()

    loss = agg_loss / batches
    top1_accuracy = correct_top1 / num_items
    top5_accuracy = correct_top5 / num_items

    return loss, top1_accuracy, top5_accuracy


def train_eval_model(model, train_loader, val_loader, test_loader, lr, num_epochs, snapshot_files_prefix,
                     snapshot_save_path):
    id_4 = generate_4_digit_id()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    training_stats = []

    os.makedirs(snapshot_save_path, exist_ok=True)

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0
        num_items, correct_top1, correct_top5 = 0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            _, predicted = torch.topk(outputs, k=5, dim=1)

            loss = criterion(outputs, labels)
            correct_top1 += (predicted[:, 0] == labels).sum().item()
            correct_top5 += torch.sum(torch.eq(predicted, labels.view(-1, 1)).any(dim=1)).item()

            num_items += labels.size(0)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        epoch_train_loss = train_loss / len(train_loader)
        train_top1_accuracy = correct_top1 / num_items
        train_top5_accuracy = correct_top5 / num_items

        epoch_val_loss, val_top1_accuracy, val_top5_accuracy = evaluate_model(model, val_loader, device, criterion)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        training_stats.append({
            'epoch': epoch + 1,
            'train_loss': epoch_train_loss,
            'train_top1_accuracy': train_top1_accuracy,
            'train_top5_accuracy': train_top5_accuracy,
            'val_loss': epoch_val_loss,
            'val_top1_accuracy': val_top1_accuracy,
            'val_top5_accuracy': val_top5_accuracy,
            'epoch_duration': epoch_duration
        })

        # save model, save only every 4th snapshot
        if (epoch + 1) % 4 == 0:
            snapshot_filename = f'{snapshot_files_prefix}-id-{id_4}-epoch-{epoch + 1}.pth'
            snapshot_path = os.path.join(snapshot_save_path, snapshot_filename)
            torch.save(model.state_dict(), snapshot_path)

        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Train Loss: {epoch_train_loss:.4f}, "
              f"Train Top1 Accuracy: {train_top1_accuracy:.4f}, "
              f"Train Top5 Accuracy: {train_top5_accuracy:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}, "
              f"Val Top1 Accuracy: {val_top1_accuracy:.4f}, "
              f"Val Top5 Accuracy: {val_top5_accuracy:.4f}, "
              f"Time: {epoch_duration:.4f} seconds")

    # eval model
    test_loss, test_top1_accuracy, test_top5_accuracy = evaluate_model(model, test_loader, device, criterion)

    # save train and eval stats
    results = {
        "training_stats": training_stats,
        "test_stats": {
            'loss': test_loss,
            'top1_accuracy': test_top1_accuracy,
            'top5_accuracy': test_top5_accuracy,
        }
    }

    stats_filename = f'{snapshot_files_prefix}-id-{id_4}-stats'
    write_measurements_and_args_to_json_file(results, args, snapshot_save_path, stats_filename)


def generate_trained_snapshot(model, train_data_loader, val_data_loader, test_data_loader, hyper_params):
    train_eval_model(model, train_data_loader, val_data_loader, test_data_loader, hyper_params["lr"],
                     hyper_params["epochs"], hyper_params["snapshot_files_prefix"], hyper_params["snapshot_save_path"])


def get_datasets(train_data_path, test_data_path):
    imagenet_data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    train_dataset = ImageFolder(train_data_path, transform=imagenet_data_transforms['train'])
    test_dataset = ImageFolder(test_data_path, transform=imagenet_data_transforms['val'])

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # cap datas set sizes
    if len(train_dataset) > 15000 * 0.8:
        train_dataset = subsample_random_items(train_dataset, int(15000 * 0.8))

    if len(val_dataset) > 15000 * 0.2:
        val_dataset = subsample_random_items(val_dataset, int(15000 * 0.2))

    if len(test_dataset) > 10000:
        test_dataset = subsample_random_items(test_dataset, 10000)

    return train_dataset, val_dataset, test_dataset


def main(args):
    # both numbers as figured out to be a good fit for all our models
    batch_size = 128
    num_workers = 12

    # prepare data config and data sets
    dataset_configs = {
        FOOD_101: {NUM_CLASSES: 101},
        STANFORD_DOGS: {NUM_CLASSES: 120},
        STANFORD_CARS: {NUM_CLASSES: 196},
        IMAGE_WOOF: {NUM_CLASSES: 10},
        CUB_BIRDS_200: {NUM_CLASSES: 200},
        IMAGENETTE: {NUM_CLASSES: 10},
        DUMMY: {NUM_CLASSES: 10}
    }

    dataset_config = dataset_configs[args.dataset_name]

    train_dataset, val_dataset, test_dataset = get_datasets(args.train_dataset_path, args.test_dataset_path)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # prepare model by loading base model, then freeze first half of model
    num_classes = dataset_config[NUM_CLASSES]
    base_model = initialize_model(
        args.model_name, pretrained=True, sequential_model=True, new_num_classes=num_classes
    )
    retrain_index = get_retrain_index(args.model_name, args.retrain_distribution)
    split_index = SPLIT_INDEXES[args.model_name][retrain_index]

    first_model, second_model = split_model_in_two(base_model, split_index)
    # freeze all parameters in first half
    for param in first_model.parameters():
        param.requires_grad = False
    model = torch.nn.Sequential(first_model, second_model)
    # for name, param in model.named_parameters():
    #     print(f'{name}: {param.requires_grad}')

    hyper_params = {
        "lr": 0.001,
        "epochs": 20,
        "snapshot_files_prefix": f'{args.model_name}-ri-{retrain_index}',
        "snapshot_save_path": args.snapshot_save_path
    }

    generate_trained_snapshot(model, train_data_loader, val_data_loader, test_data_loader, hyper_params)

    # clear space on GPU for next models
    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train a model on a dataset with optional layer freezing.")
    parser.add_argument('--model_name', type=str, choices=VISION_MODEL_CHOICES)
    parser.add_argument('--train_dataset_path', type=str)
    parser.add_argument('--test_dataset_path', type=str)
    parser.add_argument('--dataset_name', type=int)
    parser.add_argument('--retrain_distribution', type=int, choices=TWENTY_FIVE_PERCENT)
    parser.add_argument('--snapshot_save_path', type=str, default="/mount-fs/trained-snapshots")
    args = parser.parse_args()

    data_paths = {
        DUMMY: "/mount-ssd/data/imagenette-dummy",
        FOOD_101: "/mount-ssd/data/food-101/prepared_data",
        STANFORD_DOGS: "/mount-ssd/data/stanford_dogs/prepared_data",
        STANFORD_CARS: "/mount-ssd/data/stanford-cars/car_data",
        IMAGE_WOOF: "/mount-ssd/data/image-woof/imagewoof2",
        CUB_BIRDS_200: "/mount-ssd/data/cub-birds-200/prepared_data",
        IMAGENETTE: "/mount-ssd/data/imagenette2",
    }

    original_snapshot_save_path = args.snapshot_save_path

    set_seeds(42)

    for model_name in [RESNET_18, RESNET_152, EFF_NET_V2_L, VIT_L_32]:
        args.model_name = model_name
        model_count = 35

        while model_count > 0:
            print("model-count:", model_count)

            for dataset_name in [IMAGE_WOOF, STANFORD_DOGS, STANFORD_CARS, CUB_BIRDS_200, FOOD_101]:
                # for dataset_name in [DUMMY]:
                args.dataset_name = dataset_name
                for retrain_distribution in [TWENTY_FIVE_PERCENT]:
                    args.retrain_distribution = retrain_distribution
                    args.train_dataset_path = os.path.join(data_paths[dataset_name], 'train')
                    if dataset_name in [IMAGENETTE, IMAGE_WOOF]:
                        args.test_dataset_path = os.path.join(data_paths[dataset_name], 'val')
                    else:
                        args.test_dataset_path = os.path.join(data_paths[dataset_name], 'test')
                    args.snapshot_save_path = os.path.join(original_snapshot_save_path, dataset_name)

                    print(args)
                    main(args)

            model_count -= 5
