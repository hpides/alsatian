from custom.data_loaders.imagenet_transfroms import inference_transform
from global_utils.constants import TRAIN, TEST, INPUT, LABEL
from model_search.execution.data_handling.data_information import DatasetInformation, DataInfo, CachedDatasetInformation
from model_search.execution.planning.execution_plan import ExecutionPlan, ScoreModelStep, \
    CacheLocation, ScoringMethod
from model_search.execution.planning.planner_config import PlannerConfig

from model_search.model_snapshots.dfs_traversal import dfs_execution_plan
from model_search.model_snapshots.multi_model_snapshot import MultiModelSnapshot, MultiModelSnapshotEdge
from model_search.model_snapshots.rich_snapshot import LayerState

END = 'end'


class CacheConfig:
    def __init__(self, location: CacheLocation = CacheLocation.SSD, id_prefix: str = None):
        """
        Specifying where and under what id to cache
        :param location: defines where to cache e.g. on SSD, in RAM or on GPU
        :param id_prefix: is a string that defines the prefix of cached features, e.g. prefix="feat" ->
        all features could be cached with ids: "feat-1", "feat-2", ...
        """
        self.location = location
        self.id_prefix = id_prefix


class MosixExtractFeaturesStep:
    def __init__(
            self,
            _id: str,
            input_node_id: str,
            output_node_id: str,
            data_info: DataInfo,
            data_range: [int],
            layers: [LayerState],
            inp_read_cache_config: CacheConfig,
            lbl_read_cache_config: CacheConfig,
            inp_write_cache_config: CacheConfig,
            label_write_cache_config: CacheConfig,
            extract_labels: bool

    ):
        self._id = _id
        self.input_node_id = input_node_id
        self.output_node_id = output_node_id
        self.data_info = data_info
        self.data_range = data_range
        self.layers = layers
        # we use the id of the last layer as the output cache id
        self.inp_read_cache_config = inp_read_cache_config
        self.lbl_read_cache_config = lbl_read_cache_config
        self.inp_write_cache_config = inp_write_cache_config
        self.label_write_cache_config = label_write_cache_config
        self.extract_labels = extract_labels

    @property
    def contains_leaf(self):
        return self.layers[-1].is_leaf


def _get_input_cache_config(output_node_id, dataset_type, inp_lbl, data_range=None, write_cache_location=None):
    prefix = f'{output_node_id}-{dataset_type}-{inp_lbl}'
    if dataset_type == TRAIN and data_range is not None:
        prefix += f'-{data_range[0]}'

    return CacheConfig(location=write_cache_location, id_prefix=prefix)


def _get_label_cache_config(dataset_type, inp_lbl, data_range=None, write_cache_location=None):
    prefix = f'{dataset_type}-{inp_lbl}'
    if dataset_type == TRAIN and data_range is not None:
        prefix += f'-{data_range[0]}'

    return CacheConfig(location=write_cache_location, id_prefix=prefix)


class MosixExecutionPlanner:

    def __init__(self, config: PlannerConfig):
        self.config: PlannerConfig = config

    def generate_execution_plan(self, mm_snapshot: MultiModelSnapshot, train_dataset_range: [int] = None,
                                first_iteration=False, strategy="DFS") -> ExecutionPlan:
        execution_units = dfs_execution_plan(mm_snapshot.root)

        execution_steps = []

        extract_labels = True
        for exec_unit in execution_units:
            # TODO the location of the caching should be adjusted to e.g. GPU

            # if first, iteration we also have to extract the test features
            if first_iteration:
                ext_test_step = self._create_feature_ext_step(
                    exec_unit=exec_unit,
                    dataset_type=TEST,
                    data_range=None,  # use all data
                    extract_labels=extract_labels,
                    write_cache_location=CacheLocation.SSD
                )
                execution_steps.append(ext_test_step)

            # add extract feature step on subset of train data
            ext_train_step = self._create_feature_ext_step(
                exec_unit=exec_unit,
                dataset_type=TRAIN,
                data_range=train_dataset_range,
                extract_labels=extract_labels,
                write_cache_location=CacheLocation.SSD
            )
            execution_steps.append(ext_train_step)

            # if the execution unit contains a leaf we also have to add step to score the model based on the extracted features
            last_node_in_unit_child = exec_unit[-1].child
            if last_node_in_unit_child.layer_state.is_leaf:
                score_step = ScoreModelStep(
                    _id="",
                    scoring_method=ScoringMethod.FC,
                    test_feature_cache_prefixes=[
                        _get_input_cache_config(ext_train_step.output_node_id, TEST, INPUT).id_prefix],
                    train_feature_cache_prefixes=[
                        _get_input_cache_config(ext_train_step.output_node_id, TRAIN, INPUT).id_prefix],
                    num_classes=self.config.target_classes,
                    scored_models=last_node_in_unit_child.snapshot_ids)
                execution_steps.append(score_step)

            extract_labels = False

        assert sum([len(s.scored_models) for s in execution_steps if isinstance(s, ScoreModelStep)]) == len(mm_snapshot.root.snapshot_ids)

        return ExecutionPlan(execution_steps)

    def _create_feature_ext_step(self, exec_unit: [MultiModelSnapshotEdge], dataset_type, data_range, extract_labels,
                                 write_cache_location):

        # we have to execute all nodes that are children
        exec_snapshot_nodes = [u.child for u in exec_unit]
        layers = [n.layer_state for n in exec_snapshot_nodes]

        # the input id is the id of the first layer that we do not execute
        # its output should be cached by a previous step
        input_node_id = exec_unit[0].parent.layer_state.id

        # the output_id is the id of the last layer
        # its output should be cached during its execution to be reused by future steps
        output_node_id = exec_unit[-1].child.layer_state.id

        data_info = self._create_dataset_info(dataset_type, extract_labels)

        step = MosixExtractFeaturesStep(
            _id=f'{input_node_id}--{output_node_id}--{dataset_type}',
            input_node_id=input_node_id,
            output_node_id=output_node_id,
            layers=layers,
            data_info=data_info,
            data_range=data_range,
            inp_read_cache_config=_get_input_cache_config(input_node_id, dataset_type, INPUT, data_range=data_range),
            lbl_read_cache_config=_get_label_cache_config(dataset_type, LABEL, data_range=data_range),
            inp_write_cache_config=_get_input_cache_config(
                output_node_id, dataset_type, INPUT, data_range=data_range, write_cache_location=write_cache_location),
            label_write_cache_config=_get_label_cache_config(
                dataset_type, LABEL, data_range=data_range, write_cache_location=write_cache_location),
            extract_labels=extract_labels
        )

        return step

    def _create_dataset_info(self, dataset_type, extract_labels):
        if extract_labels:
            data_info = DatasetInformation(
                self.config.dataset_class,
                self.config.dataset_paths[dataset_type],
                self.config.num_workers,
                self.config.batch_size,
                dataset_type,
                inference_transform
            )
        else:
            data_info = CachedDatasetInformation(
                self.config.num_workers,
                self.config.batch_size,
                dataset_type)
        return data_info
