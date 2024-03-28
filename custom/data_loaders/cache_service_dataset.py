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
            return self.caching_service.get_item(data_id)
        else:
            label_id = self._label_ids[index]
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
