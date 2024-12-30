import os
from functools import cmp_to_key

import torch.nn.functional

from global_utils.benchmark_util import CPU
from global_utils.constants import CUDA
from model_search.execution.planning.execution_plan import CacheLocation


def _compare(item1: str, item2: str):
    suffix_1 = item1.split("-")[-2:]
    suffix_2 = item2.split("-")[-2:]
    if suffix_1[0] == suffix_2[0]:
        return int(suffix_1[1]) - int(suffix_2[1])
    else:
        return int(suffix_1[0]) - int(suffix_2[0])


class CachingService:

    def __init__(self, persistent_path: os.path):
        """
        :param persistent_path: the path on the file system used to cache items on persistent storage
        """
        self.ssd_path = persistent_path
        self._gpu_cache = {}
        self._cpu_cache = {}
        self._persistent_cache = {}
        self._delete_later = []

    def __del__(self):
        self.remove_referenced_files()
        self.remove_unreferenced_files()

    def remove_referenced_files(self):
        for file_id in self._persistent_cache.keys():
            path = self._get_path(file_id)
            self._delete_file(path)

    def remove_unreferenced_files(self):
        cached_files = set(self._gpu_cache.keys()) | set(self._cpu_cache.keys()) | set(self._persistent_cache.keys())
        if os.path.exists(self.ssd_path):
            for filename in os.listdir(self.ssd_path):
                file_id, file_extension = os.path.splitext(filename)
                if file_extension == ".pt" and file_id not in cached_files:
                    path = os.path.join(self.ssd_path, filename)
                    self._delete_file(path)

    def get_item(self, _id):
        if _id in self._gpu_cache:
            return self._gpu_cache[_id]
        elif _id in self._cpu_cache:
            return self._cpu_cache[_id]
        elif _id in self._persistent_cache:
            return torch.load(self._get_path(_id))
        else:
            raise KeyError(f'{_id} does not exist in any cache')

    def remove_item(self, _id, remove_immediately=False):
        if _id in self._gpu_cache:
            del self._gpu_cache[_id]
        elif _id in self._cpu_cache:
            del self._cpu_cache[_id]
        elif _id in self._persistent_cache:
            if remove_immediately:
                path = self._get_path(_id)
                self._delete_file(path)
            del self._persistent_cache[_id]
        else:
            raise KeyError(f'{_id} does not exist in any cache')

    def cache_on_location(self, _id, data, location: CacheLocation, allow_identical_overwrite=False):
        if location == CacheLocation.GPU:
            self.cache_on_gpu(_id, data, allow_identical_overwrite)
        elif location == CacheLocation.CPU:
            self.cache_on_cpu(_id, data, allow_identical_overwrite)
        elif location == CacheLocation.SSD:
            self.cache_persistent(_id, data, allow_identical_overwrite=allow_identical_overwrite)

    def cache_on_gpu(self, _id, data, allow_identical_overwrite=False):
        if not data.is_cuda:
            data = data.to(CUDA)

        self._check_exists_or_overwrite(_id, allow_identical_overwrite, data)

        self._gpu_cache[_id] = data

    def cache_on_cpu(self, _id, data, allow_identical_overwrite=False):
        if self._is_cuda(data):
            if isinstance(data, list) or isinstance(data, tuple):
                data = [x.to(CPU) for x in data]
            else:
                data = data.to(CPU)

        self._check_exists_or_overwrite(_id, allow_identical_overwrite, data)

        self._cpu_cache[_id] = data

    def _check_exists_or_overwrite(self, _id, allow_identical_overwrite, data):
        if self.id_exists(_id):
            if not allow_identical_overwrite:
                raise KeyError(f'{_id} already cached')
            else:
                saved_data = self.get_item(_id)
                if not torch.equal(data, saved_data):
                    print(f"currently cached data: {saved_data}")
                    print(f"new data: {data}")
                    raise KeyError(
                        f'new data supposed to save to {_id} would override non-identical cached with same id')

    def _is_cuda(self, data):
        if isinstance(data, torch.Tensor):
            return data.is_cuda
        elif isinstance(data, torch.nn.Module):
            return next(data.parameters()).is_cuda
        elif isinstance(data, list) or isinstance(data, tuple):
            return self._is_cuda(data[0])
        else:
            raise NotImplementedError

    def cache_persistent(self, _id, data, is_guaranteed_cpu_data=False, allow_identical_overwrite=False):
        if isinstance(data, list) or isinstance(data, tuple):
            data = [x.to(CPU) for x in data]
        elif (not is_guaranteed_cpu_data) and data.is_cuda:
            data = data.to(CPU)

        self._check_exists_or_overwrite(_id, allow_identical_overwrite, data)

        path = self._get_path(_id)
        torch.save(data, path)

        self._persistent_cache[_id] = path

    def _get_path(self, _id):
        return os.path.join(self.ssd_path, f'{_id}.pt')

    def move_to_cpu(self, _id):
        if _id in self._gpu_cache:
            data = self._gpu_cache.pop(_id).to(CPU)
        elif _id in self._persistent_cache:
            data = self._persistent_cache.pop(_id)
        else:
            raise KeyError(f'{_id} does not exists')

        self.cache_on_cpu(_id, data)

    def move_to_gpu(self, _id):
        if _id in self._persistent_cache:
            data = self._persistent_cache.pop(_id)
        elif _id in self._cpu_cache:
            data = self._cpu_cache.pop(_id)
        else:
            raise KeyError(f'{_id} does not exists')

        self.cache_on_gpu(_id, data)

    def move_to_persistent(self, _id):
        if _id in self._gpu_cache:
            data = self._gpu_cache.pop(_id).to(CPU)
        elif _id in self._cpu_cache:
            data = self._cpu_cache.pop(_id)
        else:
            raise KeyError(f'{_id} does not exists')

        self.cache_persistent(_id, data)

    def _delete_file(self, file_path):
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"File '{file_path}' deleted successfully.")
        else:
            print(f"File '{file_path}' does not exist.")

    def id_exists(self, _id):
        if _id in self._gpu_cache:
            return True
        elif _id in self._cpu_cache:
            return True
        elif _id in self._persistent_cache:
            return True
        else:
            return False

    def all_ids_with_prefix(self, prefix):
        result = []
        for cache in [self._gpu_cache, self._cpu_cache, self._persistent_cache]:
            result += self._all_ids_with_prefix(prefix, cache)
        # NOTE: this might seem expensive, but in most cases we do not expect to have more than 100 items
        result.sort(key=cmp_to_key(_compare))
        return result

    def _all_ids_with_prefix(self, prefix, cache):
        result = []
        for k in cache.keys():
            if prefix in k:
                result.append(k)
        return result

    def remove_all_ids_with_prefix(self, prefix, remove_immediately=False):
        for cache in [self._gpu_cache, self._cpu_cache, self._persistent_cache]:
            for k in list(cache.keys()):
                if prefix in k:
                    self.remove_item(k, remove_immediately)


if __name__ == '__main__':
    cs = CachingService('./')
    for i in range(5):
        t = torch.rand(1024, 3, 224, 224)
        cs.cache_on_gpu(str(i), t)

    for i in range(5, 10):
        t = torch.rand(1024, 3, 224, 224)
        cs.cache_on_cpu(str(i), t)

    cs.move_to_gpu(str(6))

    cs.move_to_cpu(str(2))
    print('end')
