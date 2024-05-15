from custom.data_loaders.imagenet_transfroms import inference_transform
from global_utils.constants import TEST, TRAIN, SCORE
from model_search.execution.data_handling.data_information import DatasetInformation
from model_search.execution.planning.execution_plan import ExecutionPlanner, ExecutionPlan, CacheLocation, \
    BaselineExtractFeaturesStep, ScoreModelStep, ScoringMethod
from model_search.execution.planning.planner_config import AdvancedPlannerConfig
from model_search.model_snapshots.base_snapshot import ModelSnapshot


class ShiftExtractFeaturesStep(BaselineExtractFeaturesStep):

    def __init__(self, _id: str, model_snapshot: ModelSnapshot, data_info: DatasetInformation,
                 feature_cache_prefix: str, data_range: [int], cache_locations=CacheLocation.SSD):
        super().__init__(_id, model_snapshot, data_info, feature_cache_prefix, cache_locations)
        self.data_range: [int] = data_range


class ShiftExecutionPlanner(ExecutionPlanner):

    def __init__(self, config: AdvancedPlannerConfig):
        self.config: AdvancedPlannerConfig = config
        self._train_feature_prefixes = {}

    def generate_execution_plan(self, model_snapshots: [ModelSnapshot], train_dataset_range: [int] = None,
                                first_iteration=False) -> ExecutionPlan:
        # the shift execution plan is a sequential iteration over the models
        # the one core optimization that Shift applies compares to the baseline is successive halving (SH)
        # this after very iteration, we double the amount of data and prune half of the models
        # so the sub-steps are
        # 1) extract test features
        # 2) extract train features (subset of data)
        # 3) score model based on features
        # 4) prune models, continue at step 2 with more data

        # with this method we always plan one iteration of SH

        data_set_class = self.config.dataset_class
        num_workers = self.config.num_workers
        batch_size = self.config.batch_size

        execution_steps = []
        if first_iteration:
            # we only need to extract the test features if we are in the first iteration of SH
            for snapshot in model_snapshots:
                # extract test features
                test_feature_prefix = f'{snapshot.id}-{TEST}'
                execution_steps.append(
                    BaselineExtractFeaturesStep(
                        _id=f'{snapshot.id}-extract-test-0',
                        model_snapshot=snapshot,
                        data_info=DatasetInformation(
                            data_set_class, self.config.dataset_paths[TEST], num_workers, batch_size, TEST, inference_transform),
                        cache_locations=CacheLocation.SSD,
                        feature_cache_prefix=test_feature_prefix
                    )
                )
        # for train data we always have to extract features
        for snapshot in model_snapshots:
            # extract train features
            train_feature_prefix = self._register_train_feature_prefix(snapshot, train_dataset_range)
            execution_steps.append(
                ShiftExtractFeaturesStep(
                    _id=f'{snapshot.id}-extract-train-{train_dataset_range[0]}-{train_dataset_range[1]}',
                    model_snapshot=snapshot,
                    data_info=DatasetInformation(
                        data_set_class, self.config.dataset_paths[TRAIN], num_workers, batch_size, TRAIN, inference_transform),
                    cache_locations=CacheLocation.SSD,
                    feature_cache_prefix=train_feature_prefix,
                    data_range=train_dataset_range
                )
            )
            # score model based on train and test features
            test_feature_prefix = f'{snapshot.id}-{TEST}'
            execution_steps.append(
                ScoreModelStep(
                    _id=f'{snapshot.id}-{SCORE}',
                    scoring_method=ScoringMethod.FC,
                    test_feature_cache_prefixes=[test_feature_prefix],
                    train_feature_cache_prefixes=self._train_feature_prefixes[snapshot.id],
                    num_classes=100,
                    scored_models=[snapshot.id]
                )
            )

        return ExecutionPlan(execution_steps)

    def _register_train_feature_prefix(self, snapshot, train_dataset_range):
        train_feature_prefix = f'{snapshot.id}-{TRAIN}-{train_dataset_range[0]}-{train_dataset_range[1]}'
        if not snapshot.id in self._train_feature_prefixes:
            self._train_feature_prefixes[snapshot.id] = []
        self._train_feature_prefixes[snapshot.id].append(train_feature_prefix)
        return train_feature_prefix


