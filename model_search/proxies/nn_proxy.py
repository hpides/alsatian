import torch
from torch.utils.data import DataLoader

from custom.data_loaders.cache_service_dataset import CacheServiceDataset
from global_utils.deterministic import check_deterministic_env_var_set, set_deterministic


# code adapted from/inspired by: https://github.com/DS3Lab/shift/blob/1db7f15d5fe4261d421f96c1b3a92492c8ca6b07/server/worker_general/general/classifier/_linear.py

def get_input_dimension(batch):
    sample_tensor: torch.Tensor = batch[0]
    return sample_tensor.shape


def linear_proxy(train_data_loader: DataLoader, test_data_loader: DataLoader, num_classes: int,
                 device: torch.device) -> (float, float):
    if check_deterministic_env_var_set():
        set_deterministic()

    train_dataset = train_data_loader.dataset
    test_dataset = test_data_loader.dataset
    assert isinstance(train_dataset, CacheServiceDataset), "we expect cached data"
    # If cached data not given, see olf versions of this implementation to use dataloaders
    assert isinstance(test_dataset, CacheServiceDataset), "we expect cached data"

    # collect ids for features and labels
    train_feature_ids = train_dataset._data_ids
    train_label_ids = train_dataset._label_ids
    test_feature_ids = test_dataset._data_ids
    test_label_ids = test_dataset._label_ids

    caching_service = train_dataset.caching_service

    input_dimension = caching_service.get_item(train_feature_ids[0]).shape

    # init objects
    model = torch.nn.Linear(input_dimension[1], num_classes)
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # the features should be very small and fit into GPU memory, if not we have to adjust this code and load the data
    # multiple times in the training loop below
    features, labels = collect_features_and_labels(caching_service, device, train_feature_ids, train_label_ids)

    # train model on train data
    model.train()
    for i in range(100):
        for feature_batch, label_batch in zip(features, labels):
            optimizer.zero_grad()
            outputs = model(feature_batch)
            loss = torch.nn.CrossEntropyLoss()(outputs, label_batch)
            loss.backward()
            optimizer.step()

    print('done training')

    # eval model on test data
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct_predictions = 0

    features, labels = collect_features_and_labels(caching_service, device, test_feature_ids, test_label_ids)

    loss_func = torch.nn.CrossEntropyLoss()

    for feature_batch, label_batch in zip(features, labels):
        outputs = model(feature_batch)
        loss = loss_func(outputs, label_batch)

        total_loss += loss.item()

        # Calculate top-1 accuracy
        _, predicted = torch.max(outputs, 1)
        total_samples += label_batch.size(0)
        correct_predictions += (predicted == label_batch).sum().item()

    average_loss = total_loss / len(features)
    top1_accuracy = correct_predictions / total_samples

    print('done inference')
    return average_loss, top1_accuracy


def collect_features_and_labels(caching_service, device, train_feature_ids, train_label_ids):
    features = []
    labels = []
    for feature_id, label_id in zip(train_feature_ids, train_label_ids):
        feature_batch = caching_service.get_item(feature_id)
        label_batch = caching_service.get_item(label_id)
        feature_batch, label_batch = feature_batch.to(device), label_batch.to(device)
        feature_batch, label_batch = torch.squeeze(feature_batch), torch.squeeze(label_batch)
        features.append(feature_batch)
        labels.append(label_batch)
    return features, labels
