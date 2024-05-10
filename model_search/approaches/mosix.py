from custom.data_loaders.custom_image_folder import CustomImageFolder
from custom.models.init_models import initialize_model
from experiments.snapshots.generate import generate_snapshots, RetrainDistribution
from global_utils.global_constants import TRAIN, TEST
from global_utils.model_names import RESNET_18
from model_search.approaches.shift import get_data_ranges
from model_search.caching_service import CachingService
from model_search.execution.engine.mosix_execution_engine import MosixExecutionEngine
from model_search.execution.planning.mosix_planner import MosixExecutionPlanner, MosixPlannerConfig
from model_search.model_management.model_store import ModelStore
from model_search.model_snapshots.multi_model_snapshot import MultiModelSnapshot


def divide_snapshots(ranking):
    cut = len(ranking) // 2
    return ranking[cut:], ranking[:cut]


if __name__ == '__main__':
    save_path = '/mount-fs/tmp-dir'

    # generate some dummy snapshots
    pre_trained_model = initialize_model(RESNET_18, features_only=True, pretrained=True)
    retrain_idxs = [5, 7, 9]
    split_idxs = [len(pre_trained_model) - i for i in retrain_idxs]
    model_snapshots = generate_snapshots(RESNET_18, 4, RetrainDistribution.HARD_CODED, save_path=save_path,
                                         retrain_idxs=retrain_idxs, use_same_base=True)

    # add the snapshots to a model store
    model_store = ModelStore(save_path)
    for snapshot in model_snapshots:
        model_store.add_snapshot(snapshot)

    # add all the snapshots to a multi-model snapshot
    mm_snapshot = MultiModelSnapshot()
    for snapshot in model_snapshots:
        # get snapshot from model store to have access to the rich model snapshot
        mm_snapshot.add_snapshot(model_store.get_snapshot(snapshot.id))

    # setup datasets
    dataset_paths = {
        TRAIN: '/tmp/pycharm_project_924/data/imagenette-dummy/train',
        TEST: '/tmp/pycharm_project_924/data/imagenette-dummy/val'
    }
    train_data = CustomImageFolder(dataset_paths[TRAIN])
    data_ranges = get_data_ranges(len(model_snapshots), len(train_data))

    # initialize execution planner
    planner_config = MosixPlannerConfig(12, 128)
    planner = MosixExecutionPlanner(planner_config)

    # setup execution engine
    caching_path = '/mount-ssd/cache-dir'
    tensor_caching_service = CachingService(caching_path)
    model_caching_service = CachingService(caching_path)
    exec_engine = MosixExecutionEngine(tensor_caching_service, model_caching_service, model_store)

    # executing by iterating over the data ranges
    first_iteration = True
    for data_range in data_ranges:
        execution_plan = planner.generate_execution_plan(mm_snapshot, dataset_paths, data_range, True)
        snapshot_ranking = exec_engine.execute_plan(execution_plan)
        pruned_snapshot_ids, keep_snapshot_ids = divide_snapshots(snapshot_ranking)
        mm_snapshot.prune_snapshots(pruned_snapshot_ids)
        first_iteration = False

    print('done')
