import torch.nn.functional
from torch import Tensor

from global_utils.benchmark_util import CPU
from global_utils.constants import CUDA


class CachingService:

    def __init__(self):
        self.gpu_cache = {}
        self.cpu_cache = {}
        self.ssd_cache = {}

    def cache_on_gpu(self, _id, data: Tensor):
        if data.is_cuda:
            self.gpu_cache[_id] = data
        else:
            self.gpu_cache[_id] = data.to(CUDA)

    def cache_on_cpu(self, _id, data: Tensor):
        if data.is_cuda:
            self.cpu_cache[_id] = data.to(CPU)
        else:
            self.cpu_cache[_id] = data

    def cache_on_ssd(self, _id, data: Tensor):
        raise NotImplementedError

    def move_to_cpu(self, _id):
        self.cpu_cache[_id] = self.gpu_cache[_id].to(CPU)
        self.gpu_cache.pop(_id)
        torch.cuda.empty_cache()

    def move_to_gpu(self, _id):
        self.gpu_cache[_id] = self.cpu_cache[_id].to(CUDA)

    def move_to_ssd(self, _id):
        raise NotImplementedError


if __name__ == '__main__':
    cs = CachingService()
    for i in range(5):
        t = torch.rand(1024, 3, 224, 224)
        cs.cache_on_gpu(str(i), t)

    for i in range(5, 10):
        t = torch.rand(1024, 3, 224, 224)
        cs.cache_on_cpu(str(i), t)

    cs.move_to_gpu(str(6))

    cs.move_to_cpu(str(2))
    print('end')
