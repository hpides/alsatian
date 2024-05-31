from torch.utils.data import Dataset

from model_search.caching_service import CachingService


class CacheServiceDataset(Dataset):

    def __init__(self, caching_service: CachingService, data_prefixes: [str], label_prefixes: [str] = None):
        self.caching_service: CachingService = caching_service
        self.data_prefixes: [str] = data_prefixes
        self.label_prefixes: [str] = label_prefixes
        self._input_only: bool = label_prefixes is None

        self._collect_items()

    def __len__(self):
        return len(self._data_ids)

    def __getitem__(self, index):
        data_id = self._data_ids[index]
        if self._input_only:
            return self._get_input_only(data_id)
        else:
            label_id = self._label_ids[index]
            return self._get_input_and_label(data_id, label_id)

    def _get_input_only(self, data_id):
        # see comment in method 'translate_to_actual_data' to understand why we do this
        if data_id in self.caching_service._gpu_cache:
            return data_id
        else:
            return self.caching_service.get_item(data_id)

    def _get_input_and_label(self, data_id, label_id):
        # see comment in method 'translate_to_actual_data' to understand why we do this
        if data_id in self.caching_service._gpu_cache:
            return data_id, label_id
        else:
            return self.caching_service.get_item(data_id), self.caching_service.get_item(label_id)

    def _collect_items(self):
        self._data_ids = []
        for pre in self.data_prefixes:
            self._data_ids += self.caching_service.all_ids_with_prefix(pre)

        if not self._input_only:
            self._label_ids = []
            for pre in self.label_prefixes:
                self._label_ids += self.caching_service.all_ids_with_prefix(pre)

            assert len(self._data_ids) == len(self._label_ids)

    def translate_to_actual_data(self, data):
        # currently when loading data that is already in (GPU) memory we only return the id as a string for that item
        # and just directly load it out of the cache. We do this to avoid overhead that is introduced by PyTorch's data
        # loader otherwise. There might be a smoother way to do that (e.g. by writing an own DataLoader) but so far this
        # seems to be an easy solution.
        if is_list_of_strings(data):
            assert len(data) == 1
            data_id = data[0]
            return self.caching_service.get_item(data_id)
        elif is_list_of_tuples_of_strings(data):
            assert len(data) == 2
            feature_id, label_id = data[0][0], data[1][0]
            return self.caching_service.get_item(feature_id), self.caching_service.get_item(label_id)
        elif len(data) == 2:
            return data[0][0], data[1][0]
        else:
            # in other cases that should be a tensor or a list of tensors
            return data[0]


def is_list_of_strings(lst):
    return isinstance(lst, list) and all(isinstance(item, str) for item in lst)


def is_list_of_tuples_of_strings(lst):
    return isinstance(lst, list) and all(isinstance(item[0], str) for item in lst)
