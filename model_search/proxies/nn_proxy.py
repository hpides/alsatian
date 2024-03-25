import numpy as np
import torch


# code adapted form/inspired by: https://github.com/DS3Lab/shift/blob/1db7f15d5fe4261d421f96c1b3a92492c8ca6b07/server/worker_general/general/classifier/_linear.py

def get_input_dimension(tensors: np.ndarray):
    sample_tensor: torch.Tensor = tensors[0]
    return tuple(sample_tensor.shape[1:])


def linear_proxy(train_features, train_labels, test_features, test_labels, num_classes: int, device) -> float:
    input_dimension = get_input_dimension(train_features)
    assert input_dimension == get_input_dimension(test_features), "Make sure train and test have the same shape"

    # init objects
    model = torch.nn.Linear(input_dimension[0], num_classes)
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # train model on train data
    model.train()
    for i in range(100):
        batch, label = _prepare_batch_and_label(device, train_features, train_labels)
        optimizer.zero_grad()
        outputs = model(batch)
        loss = torch.nn.CrossEntropyLoss()(outputs, label)
        # print(loss)
        loss.backward()
        optimizer.step()

    # eval model on test data
    model.eval()
    total_loss = 0.0
    total_samples = 0

    batch, label = _prepare_batch_and_label(device, test_features, test_labels)
    outputs = model(batch)
    loss = torch.nn.CrossEntropyLoss()(outputs, label)

    batch_size = batch.size(0)
    total_loss += loss.item() * batch_size
    total_samples += batch_size

    average_loss = total_loss / total_samples
    return average_loss


def _prepare_batch_and_label(device, test_features, test_labels):
    batch = torch.cat(test_features, dim=0)
    label = torch.flatten(torch.cat(test_labels, dim=0))
    batch = batch.to(device)
    label = label.to(device)
    return batch, label


if __name__ == '__main__':
    size = 100000
    train_features = [torch.randn((size, 2048), dtype=torch.float)] * 10
    train_labels = [torch.randint(low=0, high=1000, size=(size,), dtype=torch.long)] * 10
    test_features = [torch.randn((size, 2048), dtype=torch.float)] * 10
    test_labels = [torch.randint(low=0, high=1000, size=(size,), dtype=torch.long)] * 10
    linear_proxy(train_features, train_labels, test_features, test_labels, 1000, 'cuda')
