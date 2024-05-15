import random
import unittest

import numpy as np
import torch

from custom.models.init_models import initialize_model
from experiments.snapshots.generate import generate_snapshots, RetrainDistribution
from global_utils.json_operations import write_json_to_file, read_json_to_dict
from global_utils.model_names import RESNET_18
from global_utils.model_operations import state_dict_equal
from model_search.model_management.model_store import ModelStore, model_store_from_dict


class TestModelStore(unittest.TestCase):

    def setUp(self):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        # Define test save path
        self.save_path = '/mount-fs/tmp-dir'

        pre_trained_model = initialize_model(RESNET_18, features_only=True, pretrained=True)
        retrain_idxs = [5, 7, 9]
        self.snapshots = generate_snapshots(RESNET_18, 4, RetrainDistribution.HARD_CODED, save_path=self.save_path,
                                            retrain_idxs=retrain_idxs, use_same_base=True)

        self.model_store = ModelStore(self.save_path)

    def test_get_snapshot(self):
        self.model_store.add_snapshot(self.snapshots[0])
        self._test_get_snapshot_0(self.model_store)

    def _test_get_snapshot_0(self, model_store):
        retrieved_snapshot = model_store.get_snapshot(self.snapshots[0].id)
        self.assertEqual(self.snapshots[0], retrieved_snapshot)

    def test_get_model(self):
        self.model_store.add_snapshot(self.snapshots[0])
        self._test_get_model_0(self.model_store)

    def _test_get_model_0(self, model_store):
        retrieved_snapshot = model_store.get_snapshot(self.snapshots[0].id)
        sd1 = self.snapshots[0].init_model_from_snapshot().state_dict()
        sd2 = retrieved_snapshot.init_model_from_snapshot().state_dict()
        self.assertTrue(state_dict_equal(sd1, sd2))

    def test_get_composed_model(self):
        self.model_store.add_snapshot(self.snapshots[0])
        self._test_composed_model_0(self.model_store)

    def _test_composed_model_0(self, model_store):
        retrieved_snapshot = model_store.get_snapshot(self.snapshots[0].id)
        sd1 = self.snapshots[0].init_model_from_snapshot().state_dict()
        retrieved_model = model_store.get_composed_model([ls.id for ls in retrieved_snapshot.layer_states])
        self.assertTrue(state_dict_equal(sd1, retrieved_model.state_dict()))

    def test_recover_from_dict(self):
        for snap in self.snapshots:
            self.model_store.add_snapshot(snap)

        model_store_dict = self.model_store.to_dict()
        path = '/mount-fs/tmp-dir/test-json-file.json'
        write_json_to_file(model_store_dict, path)
        model_store_dict = read_json_to_dict(path)
        model_store = model_store_from_dict(model_store_dict)

        self._test_get_snapshot_0(model_store)
        self._test_get_model_0(model_store)
        self._test_composed_model_0(model_store)


if __name__ == '__main__':
    unittest.main()
