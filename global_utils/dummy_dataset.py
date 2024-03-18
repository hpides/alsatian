import os

import torch

from global_utils.file_names import clean_file_name
from global_utils.split_models import split_model


def get_input_shape(split_index, model, number_items, item_shape):
    device = torch.device("cpu")

    data = DummyDataset(number_items=number_items, input_shape=item_shape, label_shape=(1,),
                        directory='', saved_items=0)
    first, second = split_model(model, split_index)
    shape = _second_model_input_shape(data, first, device)

    return shape


def _second_model_input_shape(data, first, device):
    _input, _label = data.generate_random_batch(1)
    _input = _input.to(device)
    first = first.to(device)
    output = first(_input)
    s = output.shape
    return s[1:]


class DummyDataset:

    def __init__(self, number_items, input_shape, label_shape, directory, saved_items=None, allow_reuse=True,
                 cleanup=False):
        self.number_items = number_items
        self.input_shape = input_shape
        self.label_shape = label_shape
        if saved_items:
            self.saved_items = min(saved_items, self.number_items)
        else:
            self.saved_items = self.number_items
        self.tmp_data = os.path.join(os.path.abspath(directory), f'tmp-{self._dummy_data_id}')
        self._use_cached_data = False
        self.allow_reuse = allow_reuse
        self.cleanup = cleanup

        if self.has_persisted_items:
            if os.path.exists(self.tmp_data) and self.allow_reuse:
                self._use_cached_data = True
            else:
                os.makedirs(self.tmp_data, exist_ok=True)
                print(f"Directory '{self.tmp_data}' created.")
                self._save_items_to_disk()

    @property
    def _dummy_data_id(self):
        return clean_file_name(
            f'{self.number_items}-{"-".join(map(str, self.input_shape))}-{"-".join(map(str, self.label_shape))}-{self.saved_items}')

    @property
    def has_persisted_items(self) -> bool:
        return self.saved_items > 0

    def __del__(self):
        if os.path.exists(self.tmp_data) and self.cleanup:
            self._clear_directory()
            os.rmdir(self.tmp_data)
            print(f"Directory '{self.tmp_data}' deleted.")

    def __len__(self):
        return self.number_items

    def __getitem__(self, index):
        return self.get_item(index)

    def _clear_directory(self):
        for root, dirs, files in os.walk(self.tmp_data):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))

    def get_item(self, index):
        if self.has_persisted_items:
            item_index = index % self.saved_items
            return torch.load(self._get_item_path(item_index))
        else:
            return self._generate_random_item()

    def _save_items_to_disk(self):
        for i in range(self.saved_items):
            item = self._generate_random_item()
            item_path = self._get_item_path(i)
            torch.save(item, item_path)

    def _generate_random_item(self):
        _input = torch.randn(size=list(self.input_shape), dtype=torch.float)
        _label = torch.randint(low=0, high=1, size=list(self.label_shape), dtype=torch.uint8)
        item = (_input, _label)
        return item

    def generate_random_batch(self, batch_size):
        input_batch = torch.randn(size=[batch_size] + list(self.input_shape), dtype=torch.float)
        label_batch = torch.randint(low=0, high=256, size=[batch_size] + list(self.label_shape),
                                    dtype=torch.uint8)
        batch = (input_batch, label_batch)
        return batch

    def _get_item_path(self, i):
        return os.path.join(self.tmp_data, f'item-{i}.pt')


if __name__ == '__main__':
    temp_dir = DummyDataset(16, (224, 224, 3), (1,), './dummy_data')
    test = temp_dir._dummy_data_id
    print('test')
    del temp_dir
