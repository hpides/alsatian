from custom.data_loaders.imagenet_transfroms import inference_transform
from global_utils.constants import TRAIN, TEST
from model_search.execution.data_handling.data_information import DatasetInformation, DatasetClass
from model_search.execution.planning.execution_plan import ExecutionPlanner, ExecutionPlan, CacheLocation, \
    BaselineExtractFeaturesStep, ScoreModelStep, ScoringMethod
from model_search.execution.planning.planner_config import BaselinePlannerConfig
from model_search.model_snapshots.base_snapshot import ModelSnapshot


class BaselineExecutionPlanner(ExecutionPlanner):
    def __init__(self, config: BaselinePlannerConfig):
        self.config: BaselinePlannerConfig = config

    def generate_execution_plan(self, model_snapshots: [ModelSnapshot], dataset_paths: dict) -> ExecutionPlan:
        # the baseline execution plan is a sequential iteration over the models with no reuse of model intermediates or
        # technique to reduce the amount of data processed by the models (e.g. sucessive halving)
        # so the sub-steps are
        # 1) extract test features
        # 2) extract train features
        # 3) score model based on features

        data_set_class = DatasetClass.CUSTOM_IMAGE_FOLDER
        num_workers = self.config.num_workers
        batch_size = self.config.batch_size

        for snapshot in model_snapshots:
            execution_steps = []
            # extract test features
            test_feature_prefix = f'{snapshot.id}-{TEST}'
            execution_steps.append(
                BaselineExtractFeaturesStep(
                    _id=f'{snapshot.id}-extract-test-0',
                    model_snapshot=snapshot,
                    data_info=DatasetInformation(
                        data_set_class, dataset_paths[TEST], num_workers, batch_size, TEST, inference_transform),
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
                        data_set_class, dataset_paths[TRAIN], num_workers, batch_size, TRAIN, inference_transform),
                    cache_locations=CacheLocation.SSD,
                    feature_cache_prefix=train_feature_prefix
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
                )
            )

        return ExecutionPlan(execution_steps)
