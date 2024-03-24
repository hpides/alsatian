import os

import torch.nn.functional
from torch import Tensor

from global_utils.benchmark_util import CPU
from global_utils.constants import CUDA


class TensorCachingService:

    def __init__(self, persistent_path: os.path):
        """
        :param persistent_path: the path on the file system used to cache tensors on persistent storage
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
        for filename in os.listdir(self.ssd_path):
            file_id, file_extension = os.path.splitext(filename)
            if file_extension == ".pt" and file_id not in cached_files:
                path = os.path.join(self.ssd_path, filename)
                self._delete_file(path)

    def get_tensor(self, _id):
        if _id in self._gpu_cache:
            return self._gpu_cache[_id]
        elif _id in self._cpu_cache:
            return self._cpu_cache[_id]
        elif _id in self._persistent_cache:
            return torch.load(self._get_path(_id))
        else:
            raise KeyError(f'{_id} does not exist in any cache')

    def remove_tensor(self, _id, remove_immediately=False):
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

    def cache_on_gpu(self, _id, data: Tensor):
        self._check_id_not_exists(_id)
        if data.is_cuda:
            self._gpu_cache[_id] = data
        else:
            self._gpu_cache[_id] = data.to(CUDA)

    def cache_on_cpu(self, _id, data: Tensor):
        self._check_id_not_exists(_id)
        if data.is_cuda:
            self._cpu_cache[_id] = data.to(CPU)
        else:
            self._cpu_cache[_id] = data

    def cache_persistent(self, _id, data: Tensor):
        self._check_id_not_exists(_id)
        if data.is_cuda:
            data = data.to(CPU)
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

    def _check_id_not_exists(self, _id):
        if self.id_exists(_id):
            raise KeyError(f'{_id} already cached')


if __name__ == '__main__':
    cs = TensorCachingService('./')
    for i in range(5):
        t = torch.rand(1024, 3, 224, 224)
        cs.cache_on_gpu(str(i), t)

    for i in range(5, 10):
        t = torch.rand(1024, 3, 224, 224)
        cs.cache_on_cpu(str(i), t)

    cs.move_to_gpu(str(6))

    cs.move_to_cpu(str(2))
    print('end')
