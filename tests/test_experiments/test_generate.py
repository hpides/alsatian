import os
import random
import unittest

import numpy as np
import torch
from torchvision.models import resnet18

from custom.models.init_models import initialize_model
from custom.models.split_indices import SPLIT_INDEXES
from experiments.main_experiments.snapshots.synthetic.generate import generate_snapshots, RetrainDistribution
from global_utils.model_names import RESNET_18
from global_utils.model_operations import state_dict_equal, split_model_in_two

ID_1 = 'id1'


class TestGenerateSnapshots(unittest.TestCase):

    def setUp(self):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        self.persistent_path = "./test_data"
        os.makedirs(self.persistent_path, exist_ok=True)

    def tearDown(self):
        for file in os.listdir(self.persistent_path):
            os.remove(os.path.join(self.persistent_path, file))
        os.rmdir(self.persistent_path)

    def test_with_hardcoded_layer_indices(self):
        pre_trained_model = initialize_model(RESNET_18, features_only=True, pretrained=True)
        retrain_idxs = [3, 5, 7]
        split_idxs = [SPLIT_INDEXES[RESNET_18][i] for i in retrain_idxs]
        snaps = generate_snapshots(RESNET_18, 4, RetrainDistribution.HARD_CODED, save_path=self.persistent_path,
                                   retrain_idxs=retrain_idxs, use_same_base=True)

        models = [snap.init_model_from_snapshot() for snap in snaps]
        self.assertTrue(state_dict_equal(pre_trained_model.state_dict(), models[0].state_dict()))

        for i in range(1, 4):
            self.assertFalse(state_dict_equal(pre_trained_model.state_dict(), models[i].state_dict()))
            self._compare_base_and_new(pre_trained_model, split_idxs, models, i)

    def _compare_base_and_new(self, pre_trained_model, split_idxs, models, idx):
        base_first, base_second = split_model_in_two(pre_trained_model, split_idxs[idx - 1])
        new_first, new_second = split_model_in_two(models[idx], split_idxs[idx - 1])
        self.assertTrue(state_dict_equal(base_first.state_dict(), new_first.state_dict()))
        self.assertFalse(state_dict_equal(base_second.state_dict(), new_second.state_dict()))

    def test_deterministic(self):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        pre_trained_model1 = resnet18()

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        pre_trained_model2 = resnet18()
        self.assertTrue(state_dict_equal(pre_trained_model1.state_dict(), pre_trained_model2.state_dict()))
