import torch
from torch.utils.data import DataLoader

from global_utils.deterministic import check_deterministic_env_var_set, set_deterministic


# code adapted form/inspired by: https://github.com/DS3Lab/shift/blob/1db7f15d5fe4261d421f96c1b3a92492c8ca6b07/server/worker_general/general/classifier/_linear.py

def get_input_dimension(batch):
    sample_tensor: torch.Tensor = batch[0]
    return tuple(sample_tensor.shape[1:])


def linear_proxy(train_data_loader: DataLoader, test_data_loader: DataLoader, num_classes: int,
                 device: torch.device) -> float:
    if check_deterministic_env_var_set():
        set_deterministic()

    input_dimension = get_input_dimension(next(iter(train_data_loader)))

    # init objects
    model = torch.nn.Linear(input_dimension[1], num_classes)
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # train model on train data
    model.train()
    for i in range(100):
        for data, labels in train_data_loader:
            batch, labels = data.to(device), labels.to(device)
            batch, labels = torch.squeeze(batch), torch.squeeze(labels)
            optimizer.zero_grad()
            outputs = model(batch)
            loss = torch.nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()

    print('done training')

    # eval model on test data
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for data, labels in test_data_loader:
        batch, labels = data.to(device), labels.to(device)
        batch, labels = torch.squeeze(batch), torch.squeeze(labels)
        outputs = model(batch)
        loss = torch.nn.CrossEntropyLoss()(outputs, labels)

        batch_size = batch.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    average_loss = total_loss / total_samples

    print('done inference')
    return average_loss
