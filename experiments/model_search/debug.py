import torch
from torch.utils.data import DataLoader, Dataset


# Example dataset
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


data = [1, 2, 3, 4, 5]
data = [torch.tensor(x) for x in data]

# DataLoader with batch_size=1
data_loader = DataLoader(data, batch_size=1, shuffle=False)

# Iterating through the DataLoader
for item in data_loader:
    print(item)
