import os

from custom.data_loaders.custom_image_folder import CustomImageFolder
from experiments.model_search.benchmark_level import BenchmarkLevel
from experiments.prevent_caching.watch_utils import clear_caches_and_check_io_limit
from global_utils.constants import GENERATE_MM_SNAP, SH_RANK_ITERATION_, RANK_ITERATION_DETAILS_, GEN_EXEC_PLAN, \
    EXEC_STEP_MEASUREMENTS, CLEAR_CACHES_
from global_utils.deterministic import DETERMINISTIC_EXECUTION, check_deterministic_env_var_set, set_deterministic
from global_utils.global_constants import TRAIN, TEST
from model_search.approaches.dummy_snapshots import dummy_snap_and_mstore_four_models
from model_search.approaches.shift import get_data_ranges, divide_snapshots, get_sorted_model_scores, _init_benchmarker
from model_search.caching_service import CachingService
from model_search.execution.data_handling.data_information import DatasetClass
from model_search.execution.engine.mosix_execution_engine import MosixExecutionEngine
from model_search.execution.planning.mosix_planner import MosixExecutionPlanner
from model_search.execution.planning.planner_config import PlannerConfig
from model_search.model_management.model_store import ModelStore
from model_search.model_snapshots.base_snapshot import ModelSnapshot
from model_search.model_snapshots.multi_model_snapshot import MultiModelSnapshot


def find_best_model(model_snapshots: [ModelSnapshot], train_data_length, planner_config,
                    caching_path, model_store: ModelStore, benchmark_level: BenchmarkLevel = None):
    measurements = {}
    benchmarker = _init_benchmarker('find_best_model', benchmark_level)

    # add all the snapshots to a multi-model snapshot
    measurement, mm_snapshot = benchmarker.micro_benchmark_cpu(_gen_mm_snapshot, model_snapshots, model_store)
    measurements[GENERATE_MM_SNAP] = measurement

    # initialize execution planner
    planner = MosixExecutionPlanner(planner_config)

    # initialize execution engine
    tensor_caching_service = CachingService(caching_path)
    model_caching_service = CachingService(caching_path)
    exec_engine = MosixExecutionEngine(tensor_caching_service, model_caching_service, model_store)

    ###########################################################
    # executing by iterating over the data ranges
    ###########################################################
    data_ranges = get_data_ranges(len(model_snapshots), train_data_length)
    ranking = None
    first_iteration = True
    for i, data_range in enumerate(data_ranges):
        # make sure we have expected I/O speed
        measurement, _ = benchmarker.micro_benchmark_cpu(clear_caches_and_check_io_limit)
        measurements[f'{CLEAR_CACHES_}{i}'] = measurement

        measurement, (measure, ranking) = benchmarker.micro_benchmark_gpu(
            _sh_iteration, data_range, exec_engine, first_iteration, mm_snapshot, planner, ranking, benchmark_level)
        measurements[f'{SH_RANK_ITERATION_}{i}'] = measurement
        measurements[f'{RANK_ITERATION_DETAILS_}{i}'] = measure
        first_iteration = False
    return measurements, ranking


def _sh_iteration(data_range, exec_engine, first_iteration, mm_snapshot, planner, ranking, benchmark_level):
    measurements = {}
    benchmarker = _init_benchmarker('_sh_iteration', benchmark_level)

    measure, execution_plan = benchmarker.micro_benchmark_cpu(
        planner.generate_execution_plan, mm_snapshot, data_range, first_iteration)
    measurements[GEN_EXEC_PLAN] = measure

    measurements[EXEC_STEP_MEASUREMENTS] = exec_engine.execute_plan(execution_plan, benchmark_level=benchmark_level)
    pruned_snapshot_ids, keep_snapshot_ids = divide_snapshots(execution_plan.execution_steps)
    mm_snapshot.prune_snapshots(pruned_snapshot_ids)
    ranking = get_sorted_model_scores(execution_plan.execution_steps)

    return measurements, ranking


def _gen_mm_snapshot(model_snapshots, model_store):
    mm_snapshot = MultiModelSnapshot()
    for snapshot in model_snapshots:
        # get snapshot from model store to have access to the rich model snapshot
        mm_snapshot.add_snapshot(model_store.get_snapshot(snapshot.id))
    return mm_snapshot


if __name__ == '__main__':
    # basically for this we do not need deterministic execution, leave the flag here if we want to debug
    os.environ[DETERMINISTIC_EXECUTION] = ""
    # os.environ[DETERMINISTIC_EXECUTION] = TRUE

    if check_deterministic_env_var_set():
        num_workers = 0
        set_deterministic()
    else:
        num_workers = 12

    save_path = '/mount-fs/tmp-dir'
    model_snapshots, model_store = dummy_snap_and_mstore_four_models(save_path)

    # datasets
    dataset_paths = {
        TRAIN: '/tmp/pycharm_project_924/data/imagenette-dummy/train',
        TEST: '/tmp/pycharm_project_924/data/imagenette-dummy/val'
    }
    train_data = CustomImageFolder(dataset_paths[TRAIN])

    planner_config = PlannerConfig(num_workers, 128, 100, DatasetClass.CUSTOM_IMAGE_FOLDER, dataset_paths)
    persistent_caching_path = '/mount-ssd/cache-dir'

    find_best_model(model_snapshots, len(train_data), planner_config, persistent_caching_path, model_store)
