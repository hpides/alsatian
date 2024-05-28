from custom.data_loaders.imagenet_transfroms import inference_transform
from global_utils.constants import TRAIN, TEST
from model_search.execution.data_handling.data_information import DatasetInformation
from model_search.execution.planning.execution_plan import ExecutionPlan, CacheLocation, \
    BaselineExtractFeaturesStep, ScoreModelStep, ScoringMethod
from model_search.execution.planning.planner_config import PlannerConfig
from model_search.model_snapshots.base_snapshot import ModelSnapshot


class BaselineExecutionPlanner:
    def __init__(self, config: PlannerConfig):
        self.config: PlannerConfig = config

    def generate_execution_plan(self, model_snapshots: [ModelSnapshot]) -> ExecutionPlan:
        # the baseline execution plan is a sequential iteration over the models with no reuse of model intermediates or
        # technique to reduce the amount of data processed by the models (e.g. sucessive halving)
        # so the sub-steps are
        # 1) extract test features
        # 2) extract train features
        # 3) score model based on features

        data_set_class = self.config.dataset_class
        num_workers = self.config.num_workers
        batch_size = self.config.batch_size

        execution_steps = []
        for snapshot in model_snapshots:
            # extract test features
            test_feature_prefix = f'{snapshot.id}-{TEST}'
            execution_steps.append(
                BaselineExtractFeaturesStep(
                    _id=f'{snapshot.id}-extract-test-0',
                    model_snapshot=snapshot,
                    data_info=DatasetInformation(
                        data_set_class, self.config.dataset_paths[TEST], num_workers, batch_size, TEST,
                        inference_transform),
                    cache_locations=CacheLocation.SSD,
                    feature_cache_prefix=test_feature_prefix
                )
            )
            # extract train features
            train_feature_prefix = f'{snapshot.id}-{TRAIN}'
            execution_steps.append(
                BaselineExtractFeaturesStep(
                    _id=f'{snapshot.id}-extract-train-0',
                    model_snapshot=snapshot,
                    data_info=DatasetInformation(
                        data_set_class, self.config.dataset_paths[TRAIN], num_workers, batch_size, TRAIN,
                        inference_transform),
                    cache_locations=CacheLocation.SSD,
                    feature_cache_prefix=train_feature_prefix,
                )
            )
            # score model based on train and test features
            execution_steps.append(
                ScoreModelStep(
                    _id=f'{snapshot.id}-score',
                    scoring_method=ScoringMethod.FC,
                    test_feature_cache_prefixes=[test_feature_prefix],
                    train_feature_cache_prefixes=[train_feature_prefix],
                    num_classes=self.config.target_classes,
                    scored_models=[snapshot.id]
                )
            )

        return ExecutionPlan(execution_steps)
