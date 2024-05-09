import os
import unittest

import torch

from global_utils.constants import CUDA
from model_search.caching_service import CachingService

ID_1 = 'id1'


class TestTensorCachingService(unittest.TestCase):
    def setUp(self):
        self.persistent_path = "./test_data"
        os.makedirs(self.persistent_path, exist_ok=True)
        self.tensor1 = torch.tensor([1, 2, 3])
        self.tensor2 = torch.tensor([4, 5, 6])
        self.tensor3 = torch.tensor([7, 8, 9])

        self.service = CachingService(self.persistent_path)

    def tearDown(self):
        for file in os.listdir(self.persistent_path):
            os.remove(os.path.join(self.persistent_path, file))
        os.rmdir(self.persistent_path)

    def test_cache_on_gpu(self):
        self.service.cache_on_gpu(ID_1, self.tensor1)
        self.assertTrue(ID_1 in self.service._gpu_cache)
        result = self.service.get_item(ID_1)
        tensor_1_cuda = self.tensor1.to(CUDA)
        self.assertTrue(torch.equal(result, tensor_1_cuda))

    def test_cache_on_cpu(self):
        self.service.cache_on_cpu(ID_1, self.tensor1)
        self.assertTrue(ID_1 in self.service._cpu_cache)
        result = self.service.get_item(ID_1)
        self.assertTrue(torch.equal(result, self.tensor1))

    def test_cache_persistent(self):
        self.service.cache_persistent(ID_1, self.tensor1)
        self.assertTrue(ID_1 in self.service._persistent_cache)
        result = self.service.get_item(ID_1)
        self.assertTrue(torch.equal(result, self.tensor1))

    def test_move_gpu_to_cpu(self):
        self.service.cache_on_gpu(ID_1, self.tensor1)
        self.service.move_to_cpu(ID_1)
        self.assertFalse(ID_1 in self.service._gpu_cache)
        self.assertTrue(ID_1 in self.service._cpu_cache)
        result = self.service.get_item(ID_1)
        self.assertTrue(torch.equal(result, self.tensor1))

    def test_move_gpu_to_persistent(self):
        self.service.cache_on_gpu(ID_1, self.tensor1)
        self.service.move_to_persistent(ID_1)
        self.assertFalse(ID_1 in self.service._gpu_cache)
        self.assertTrue(ID_1 in self.service._persistent_cache)
        result = self.service.get_item(ID_1)
        self.assertTrue(torch.equal(result, self.tensor1))

    def test_move_cpu_to_gpu(self):
        self.service.cache_on_cpu(ID_1, self.tensor1)
        self.service.move_to_gpu(ID_1)
        self.assertFalse(ID_1 in self.service._cpu_cache)
        self.assertTrue(ID_1 in self.service._gpu_cache)
        result = self.service.get_item(ID_1)
        tensor_1_cuda = self.tensor1.to(CUDA)
        self.assertTrue(torch.equal(result, tensor_1_cuda))

    def test_add_existing_id(self):
        self.service.cache_on_cpu(ID_1, self.tensor1)
        self.service.move_to_gpu(ID_1)

        with self.assertRaises(KeyError):
            self.service.cache_on_cpu(ID_1, self.tensor2)

    def test_get_tensor_non_existing(self):
        with self.assertRaises(KeyError):
            self.service.get_item('non_existing_id')
