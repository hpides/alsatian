from torch.utils.data import Dataset

from model_search.caching_service import TensorCachingService


class CacheServiceDataset(Dataset):

    def __init__(self, caching_service: TensorCachingService, data_prefixes: [str], label_prefixes: [str]):
        self.caching_service: TensorCachingService = caching_service
        self.data_prefixes: [str] = data_prefixes
        self.label_prefixes: [str] = label_prefixes

        self._collect_items()

    def __len__(self):
        return len(self._data_ids)

    def __getitem__(self, index):
        data_id = self._data_ids[index]
        label_id = self._label_ids[index]
        return self.caching_service.get_tensor(data_id), self.caching_service.get_tensor(label_id)

    def _collect_items(self):
        self._data_ids = []
        for pre in self.data_prefixes:
            self._data_ids += self.caching_service.all_ids_with_prefix(pre)

        self._label_ids = []
        for pre in self.label_prefixes:
            self._label_ids += self.caching_service.all_ids_with_prefix(pre)

        assert len(self._data_ids) == len(self._label_ids)
