from custom.data_loaders.imagenet_transfroms import inference_transform
from global_utils.constants import TRAIN, TEST, INPUT, LABEL
from model_search.execution.data_handling.data_information import DatasetInformation, DataInfo, CachedDatasetInformation
from model_search.execution.planning.execution_plan import ExecutionPlan, ScoreModelStep, \
    CacheLocation, ScoringMethod, ModifyCacheStep
from model_search.execution.planning.execution_tree import execution_tree_from_mm_snapshot, Computation, Release
from model_search.execution.planning.planner_config import PlannerConfig
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
    # if dataset_type == TRAIN and data_range is not None:
    if data_range is not None:
        prefix += f'-{data_range[0]}'

    return CacheConfig(location=write_cache_location, id_prefix=prefix)


def _get_label_cache_config(dataset_type, inp_lbl, data_range=None, write_cache_location=None):
    prefix = f'{dataset_type}-{inp_lbl}'
    # if dataset_type == TRAIN and data_range is not None:
    if data_range is not None:
        prefix += f'-{data_range[0]}'

    return CacheConfig(location=write_cache_location, id_prefix=prefix)


def _contains_leaf(exec_unit):
    last_node_in_unit_child = exec_unit[-1].child
    return last_node_in_unit_child.layer_state.is_leaf


def _split_up_data_range(dataset_range, num_items):
    all_data_used = False
    new_ranges = []
    global_end = dataset_range[1]

    current_start = dataset_range[0]
    current_end = min(dataset_range[0] + num_items, global_end)
    while not all_data_used:
        new_ranges.append([current_start, current_end])
        all_data_used = current_end == global_end

        current_start = current_end
        current_end = min(current_end + num_items, global_end)

    return new_ranges


def _is_last_iteration(test_ranges, train_ranges):
    result = ((len(test_ranges) == 0 and len(train_ranges) == 0))
    return result


def split_num_items_budget(train_ranges, test_ranges):
    # NOTE: there might be some room for optimization here in case the sum of train and test ranges for one step is
    # below the available budget
    # the easies way to split is by for the first n iterations give train no budget
    zero_budgets = [[0, 0]] * len(test_ranges)
    train_ranges = zero_budgets + train_ranges
    return train_ranges, test_ranges


class MosixExecutionPlanner:

    def __init__(self, config: PlannerConfig):
        self.config: PlannerConfig = config

    def generate_execution_plan(self, mm_snapshot: MultiModelSnapshot, train_dataset_range: [int], len_test_data: int,
                                first_iteration=False, strategy="DFS", model_input_size=3 * 224 * 224) -> ExecutionPlan:

        execution_tree = execution_tree_from_mm_snapshot(mm_snapshot, 0)

        # For now, assume that model always fits on GPU, leave out of memory models for future work
        available_budget = self.config.cache_size
        # returns a map with keys: max items per iteration, and values: the node_sequence and edge_sequences
        traversal_groups = execution_tree.best_traversal(available_budget)

        execution_steps = []
        for max_num_items, (node_sequence, edge_sequence) in traversal_groups.items():
            # if we only extract train features, we take the current dataset range and split it up into multiple
            # sub-ranges to stay below the maximum number of items that still meet our budget
            train_ranges = _split_up_data_range(train_dataset_range, max_num_items)
            test_ranges = []

            if first_iteration:
                # if it is the first iteration, we also have to extract test features
                # this means that we can not use all our "budget" in terms of items to process in one iteration for
                # the train data. Thus , we split the budget of items to process at once equally between train and test
                test_ranges = _split_up_data_range([0, len_test_data], max_num_items)
                train_ranges, test_ranges = split_num_items_budget(train_ranges, test_ranges)

            # generate execution steps based on generated ranges
            execution_steps = []
            execution_steps += self._execution_steps_for_data_ranges(
                mm_snapshot, edge_sequence, test_ranges, train_ranges)

        return ExecutionPlan(execution_steps)

    def _execution_steps_for_data_ranges(self, mm_snapshot, edge_sequence, test_ranges, train_ranges):
        execution_steps = []

        while len(test_ranges) > 0 or len(train_ranges) > 0:
            extract_labels = True

            test_data_range = test_ranges.pop(0) if len(test_ranges) > 0 else None
            train_data_range = train_ranges.pop(0) if len(train_ranges) > 0 else None

            for execution_step in edge_sequence:
                # TODO the location of the caching should be dynamically adjusted, for now always use default_cache_location

                if isinstance(execution_step, Release):
                    self._create_modify_cache_step(execution_step, execution_steps)
                elif isinstance(execution_step, Computation):
                    # create actual execution steps
                    exec_unit = execution_step.execution_unit

                    # first steps for test data
                    ext_test_step = None
                    if test_data_range:
                        ext_test_step = self._test_feature_step(exec_unit, extract_labels, test_data_range)
                        execution_steps.append(ext_test_step)

                    ext_train_step = None
                    if train_data_range:
                        ext_train_step = self._train_feature_step(exec_unit, extract_labels, train_data_range)
                        execution_steps.append(ext_train_step)

                    # if the execution unit contains a leaf and we in the last iteration that has train or test data
                    # we also have to add step to score the model based on the extracted features
                    if _contains_leaf(exec_unit) and _is_last_iteration(test_ranges, train_ranges):
                        score_step = self._score_step(exec_unit, [ext_test_step, ext_train_step])
                        execution_steps.append(score_step)

                    extract_labels = False

                elif execution_step is None:
                    pass
                else:
                    raise NotImplementedError

        # check that we have as many Score model steps as snapshot IDs, basically make sure every model is scored
        assert (sum([len(s.scored_models) for s in execution_steps if isinstance(s, ScoreModelStep)])
                == len(mm_snapshot.root.snapshot_ids))

        return execution_steps

    def _score_step(self, exec_unit, ext_test_train_steps):
        test_step, step = ext_test_train_steps
        if step is None:
            step = test_step

        last_node_in_unit_child = exec_unit[-1].child
        score_step = ScoreModelStep(
            _id="",
            scoring_method=ScoringMethod.FC,
            test_feature_cache_prefixes=[
                _get_input_cache_config(step.output_node_id, TEST, INPUT).id_prefix],
            train_feature_cache_prefixes=[
                _get_input_cache_config(step.output_node_id, TRAIN, INPUT).id_prefix],
            num_classes=self.config.target_classes,
            scored_models=last_node_in_unit_child.snapshot_ids)
        return score_step

    def _train_feature_step(self, exec_unit, extract_labels, train_dataset_range):
        ext_train_step = self._create_feature_ext_step(
            exec_unit=exec_unit,
            dataset_type=TRAIN,
            data_range=train_dataset_range,
            extract_labels=extract_labels,
            write_cache_location=self.config.default_cache_location
        )
        return ext_train_step

    def _test_feature_step(self, exec_unit, extract_labels, test_data_range=None):
        ext_test_step = self._create_feature_ext_step(
            exec_unit=exec_unit,
            dataset_type=TEST,
            data_range=test_data_range,  # if use all data -> None
            extract_labels=extract_labels,
            write_cache_location=self.config.default_cache_location
        )
        return ext_test_step

    def _create_modify_cache_step(self, exec_unit: Release, execution_steps):
        step = ModifyCacheStep("", cache_evict_ids=[exec_unit.intermediate._id])
        execution_steps.append(step)

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
