import torch
from torch.utils.data import DataLoader

from global_utils.deterministic import check_deterministic_env_var_set, set_deterministic


# code adapted from/inspired by: https://github.com/DS3Lab/shift/blob/1db7f15d5fe4261d421f96c1b3a92492c8ca6b07/server/worker_general/general/classifier/_linear.py

def get_input_dimension(batch):
    sample_tensor: torch.Tensor = batch[0]
    return sample_tensor.shape


def linear_proxy(train_data_loader: DataLoader, test_data_loader: DataLoader, num_classes: int,
                 device: torch.device) -> float:
    if check_deterministic_env_var_set():
        set_deterministic()

    item = next(iter(train_data_loader))
    item = train_data_loader.dataset.translate_to_actual_data(item)
    input_dimension = get_input_dimension(item)

    # init objects
    model = torch.nn.Linear(input_dimension[1], num_classes)
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # the features should be very small and fit into GPU memory, if not we have to adjust this code and load the data
    # multiple times in the training loop below
    features = []
    labels = []
    # for feature_batch, label_batch in train_data_loader:
    for i, (batch) in enumerate(train_data_loader):
        batch = train_data_loader.dataset.translate_to_actual_data(batch)
        feature_batch, label_batch = batch
        feature_batch, label_batch = feature_batch.to(device), label_batch.to(device)
        feature_batch, label_batch = torch.squeeze(feature_batch), torch.squeeze(label_batch)
        features.append(feature_batch)
        labels.append(label_batch)

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

    for batch in test_data_loader:
        features, labels = train_data_loader.dataset.translate_to_actual_data(batch)
        features, labels = features.to(device), labels.to(device)
        features, labels = torch.squeeze(features), torch.squeeze(labels)

        outputs = model(features)
        loss = torch.nn.CrossEntropyLoss()(outputs, labels)

        batch_size = features.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    average_loss = total_loss / total_samples

    print('done inference')
    return average_loss
