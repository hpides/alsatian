import unittest

import torch

from global_utils.model_operations import state_dict_equal


class TestStateDictEqual(unittest.TestCase):

    def test_empty_dicts(self):
        d1 = {}
        d2 = {}

        self.assertTrue(state_dict_equal(d1, d2))

    def test_same_dicts(self):
        tensor = torch.rand(3, 300, 400)

        d1 = {'test': tensor}

        self.assertTrue(state_dict_equal(d1, d1))

    def test_equal_dicts(self):
        tensor = torch.rand(3, 300, 400)

        d1 = {'test': tensor}
        d2 = {'test': tensor}

        self.assertTrue(state_dict_equal(d1, d2))

    def test_different_keys(self):
        tensor = torch.rand(3, 300, 400)

        d1 = {'test1': tensor}
        d2 = {'test2': tensor}

        self.assertFalse(state_dict_equal(d1, d2))

    def test_different_tensor(self):
        tensor1 = torch.rand(3, 300, 400)
        tensor2 = torch.rand(3, 300, 400)

        d1 = {'test': tensor1}
        d2 = {'test': tensor2}

        self.assertFalse(state_dict_equal(d1, d2))
