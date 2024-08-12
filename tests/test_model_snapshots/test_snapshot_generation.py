import os
import random
import shutil
import unittest

from experiments.main_experiments.snapshots.synthetic.generate import RetrainDistribution
from experiments.main_experiments.snapshots.synthetic.generate_set import generate_snapshot_set
from global_utils.model_names import RESNET_18
from model_search.model_snapshots.multi_model_snapshot import MultiModelSnapshot


class TestSnapshotGeneration(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp_dir = './tmp_dir'
        self.model_name = RESNET_18
        os.makedirs(self.tmp_dir, exist_ok=True)
        random.seed(42)

    def tearDown(self) -> None:
        # Remove the directory and all its contents
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    def test_generation_top_layers(self):
        retrain_idxs = [1, 2, 1, 1]
        self._test_generation_layers(retrain_idxs)

    def test_generation_top_25_percent_layers(self):
        retrain_idxs = [2, 4, 3, 3]
        self._test_generation_layers(retrain_idxs)

    def test_generation_top_50_percent_layers(self):
        retrain_idxs = [4, 6, 5, 5]
        self._test_generation_layers(retrain_idxs)

    def _test_generation_layers(self, retrain_idxs):
        snapshots, model_store = generate_snapshot_set(
            self.model_name, len(retrain_idxs) + 1, RetrainDistribution.HARD_CODED, self.tmp_dir,
            retrain_idxs=retrain_idxs)
        # see how many unique blocks we have
        # get how many blocks does one model have
        first_model = list(model_store.models.values())[0]
        first_model_blocks = len(first_model.layer_states)
        unique_blocks = \
            set([(layer.architecture_hash, layer.state_dict_hash) for layer in list(model_store.layers.values())])
        mm_snapshot = MultiModelSnapshot()
        for snapshot in snapshots:
            mm_snapshot.add_snapshot(model_store.get_snapshot(snapshot.id))
        self.assertEqual(sum(retrain_idxs) + first_model_blocks, len(unique_blocks))
