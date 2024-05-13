from custom.data_loaders.imagenet_transfroms import inference_transform
from global_utils.constants import TRAIN, TEST, INPUT
from model_search.execution.data_handling.data_information import DatasetInformation, DataInfo, DatasetClass, \
    CachedDatasetInformation
from model_search.execution.planning.execution_plan import ExecutionPlanner, ExecutionPlan, ScoreModelStep, \
    CacheLocation, ScoringMethod
from model_search.execution.planning.shift_planner import SCORE

from model_search.model_snapshots.dfs_traversal import dfs_execution_plan
from model_search.model_snapshots.multi_model_snapshot import MultiModelSnapshot, MultiModelSnapshotEdge
from model_search.model_snapshots.rich_snapshot import LayerState

END = 'end'


class MosixPlannerConfig:
    def __init__(self, num_workers: int, batch_size: int):
        self.num_workers = num_workers
        self.batch_size = batch_size


class CachingConfig:
    def __init__(self, id_prefix: str = None, location: CacheLocation = CacheLocation.SSD):
        """
        Specifying where and under what id to cache
        :param id_prefix: is a string that defines the prefix of cached features, e.g. prefix="feat" ->
        all features could be cached with ids: "feat-1", "feat-2", ...
        :param location: defines where to cache e.g. on SSD, in RAM or on GPU
        """
        self.id_prefix = id_prefix
        self.location = location


class MosixExtractFeaturesStep:
    def __init__(
            self,
            _id: str,
            input_node_id: str,
            output_node_id: str,
            data_info: DataInfo,
            data_range: [int],
            layers: [LayerState],
            cache_location: CacheLocation,
            extract_labels: bool

    ):
        self._id = _id
        self.input_node_id = input_node_id
        self.output_node_id = output_node_id
        self.data_info = data_info
        self.data_range = data_range
        self.layers = layers
        # we use the id of the last layer as the output cache id
        self.cache_config = CachingConfig(output_node_id, cache_location)
        self.extract_labels = extract_labels


class MosixExecutionPlanner(ExecutionPlanner):

    def __init__(self, config: MosixPlannerConfig):
        self.config: MosixPlannerConfig = config

    def generate_execution_plan(self, mm_snapshot: MultiModelSnapshot, dataset_paths: dict,
                                train_dataset_range: [int] = None, first_iteration=False,
                                strategy="DFS") -> ExecutionPlan:
        execution_units = dfs_execution_plan(mm_snapshot.root)

        execution_steps = []

        extract_labels = True
        for exec_unit in execution_units:
            # TODO the location of the caching should be adjusted to e.g. GPU

            # if first, iteration we also have to extract the test features
            if first_iteration:
                ext_test_step = self._create_feature_ext_step(
                    exec_unit, dataset_paths, TEST, self.config.num_workers, self.config.batch_size, None,
                    CacheLocation.SSD, extract_labels)
                execution_steps.append(ext_test_step)

            # add extract feature step on subset of train data
            ext_train_step = self._create_feature_ext_step(
                exec_unit, dataset_paths, TRAIN, self.config.num_workers, self.config.batch_size, train_dataset_range,
                CacheLocation.SSD, extract_labels)
            execution_steps.append(ext_train_step)

            # if the execution unit contains a leaf we also have to add step to score the model based on the extracted features
            last_node_in_unit_child = exec_unit[-1].child
            if last_node_in_unit_child.layer_state.is_leaf:
                score_step = ScoreModelStep(
                    _id="",
                    scoring_method=ScoringMethod.FC,
                    # test_feature_cache_prefixes=f'{TEST}-{LABEL}',
                    test_feature_cache_prefixes=[f'{ext_train_step.cache_config.id_prefix}-{TEST}-{INPUT}'],
                    train_feature_cache_prefixes=[f'{ext_train_step.cache_config.id_prefix}-{TRAIN}-{INPUT}'],
                    num_classes=100,
                    scored_models=last_node_in_unit_child.snapshot_ids)
                execution_steps.append(score_step)

            extract_labels = False

        return ExecutionPlan(execution_steps)

    def _create_feature_ext_step(self, exec_unit: [MultiModelSnapshotEdge], dataset_paths, dataset_type, num_workers,
                                 batch_size, data_range, cache_location, extract_labels):
        data_set_class = DatasetClass.CUSTOM_IMAGE_FOLDER

        # we have to execute all nodes that are children
        exec_snapshot_nodes = [u.child for u in exec_unit]
        layers = [n.layer_state for n in exec_snapshot_nodes]

        # the input id is the id of the first layer (that we not execute)
        input_node_id = exec_unit[0].parent.layer_state.id

        # the output_id is the id of the last layer
        output_node_id = exec_unit[-1].child.layer_state.id

        if extract_labels:
            data_info = DatasetInformation(
                data_set_class, dataset_paths[dataset_type], num_workers, batch_size, dataset_type, inference_transform)
        else:
            data_info = CachedDatasetInformation(num_workers, batch_size, dataset_type)

        step = MosixExtractFeaturesStep(
            _id=f'{input_node_id}--{output_node_id}--{dataset_type}',
            input_node_id=input_node_id,
            output_node_id=output_node_id,
            layers=layers,
            data_info=data_info,
            data_range=data_range,
            cache_location=cache_location,
            extract_labels=extract_labels
        )

        return step



