import unittest

import torch
from torch.utils.data import DataLoader

from custom.data_loaders.cache_service_dataset import CacheServiceDataset
from global_utils.model_operations import tensor_equal
from model_search.caching_service import CachingService
from model_search.execution.planning.execution_plan import CacheLocation


class TestDeterministicOutput(unittest.TestCase):

    def test_load_data_from_gpu_and_cpu(self):

        cache_prefixes = ["pre0", "pre1", "pre2", "pre3"]
        cache_location = [CacheLocation.GPU, CacheLocation.GPU, CacheLocation.CPU, CacheLocation.CPU]
        random_tensors = [torch.randn(3, 3), torch.randn(3, 3), torch.randn(3, 3), torch.randn(3, 3)]
        caching_service = CachingService('./tmp')

        for prefix, random_tensor, location in zip(cache_prefixes, random_tensors, cache_location):
            caching_service.cache_on_location(prefix, random_tensor, location)

        data_set = CacheServiceDataset(
            caching_service,
            cache_prefixes,
        )
        data_loader = DataLoader(data_set, batch_size=1, num_workers=2)

        i = 0
        for data in data_loader:
            # if we get a string back load actual item form caching service (because already in GPU memory)
            d = data_set.translate_to_actual_data(data)
            # make sure tensor is on CPU for comparison
            d = d.to("cpu")
            t = random_tensors[i].to("cpu")
            if not tensor_equal(d, t):
                print(d)
                print(t)
            self.assertTrue(tensor_equal(d, t))
            i += 1

