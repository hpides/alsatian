import math
import os
import random
import tempfile

import torch

from global_utils.split_models import split_model


def get_input_shape(split_index, model, batch_size, number_items, item_shape):
    with tempfile.TemporaryDirectory() as temp_dir:
        device = torch.device("cpu")

        data = DummyDataset(batch_size=batch_size, number_items=number_items, item_shape=item_shape,
                            directory=temp_dir)
        first, second = split_model(model, split_index)
        shape = _second_model_input_shape(data, first, device)

        return shape


def _second_model_input_shape(data, first, device):
    batch = data.get_random_batch().to(device)
    first.to(device)
    output = first(batch)
    s = output.shape
    return s


class DummyDataset:

    def __init__(self, batch_size, number_items, item_shape, directory, saved_batches=30):
        self.batch_size = batch_size
        self.number_items = number_items
        self.item_shape = item_shape
        self.numer_of_batches = math.ceil(number_items / batch_size)
        self.tmp_data = os.path.abspath(directory)
        self.saved_batches = min(saved_batches, self.numer_of_batches)

        os.makedirs(self.tmp_data, exist_ok=True)
        print(f"Directory '{self.tmp_data}' created.")

        self._save_batches_to_disk(self.saved_batches)

    def __del__(self):
        if os.path.exists(self.tmp_data):
            self._clear_directory()
            os.rmdir(self.tmp_data)
            print(f"Directory '{self.tmp_data}' deleted.")

    def __len__(self):
        return self.number_items

    def __getitem__(self, index):
        return self.get_random_batch()

    def _clear_directory(self):
        for root, dirs, files in os.walk(self.tmp_data):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))

    def get_random_batch(self):
        batch_index = random.randint(0, self.saved_batches - 1)
        return torch.load(self._get_batch_path(batch_index))

    def _save_batches_to_disk(self, number_of_batches):
        for i in range(number_of_batches):
            batch = torch.randn(size=[self.batch_size] + list(self.item_shape), dtype=torch.float)
            batch_path = self._get_batch_path(i)
            torch.save(batch, batch_path)

    def _get_batch_path(self, i):
        return os.path.join(self.tmp_data, f'batch-{i}.pt')


if __name__ == '__main__':
    temp_dir = DummyDataset(16, 50, (224, 224, 3), './dummy_data')
    print('test')
    del temp_dir
