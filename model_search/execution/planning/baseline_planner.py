from custom.data_loaders.imagenet_transfroms import inference_transform
from model_search.execution.data_handling.data_information import DatasetInformation, DatasetClass
from model_search.execution.planning.execution_plan import ExecutionPlanner, ExecutionPlan, CacheLocation, \
    BaselineExtractFeaturesStep, ScoreModelStep, ScoringMethod
from model_search.model_snapshot import ModelSnapshot

TRAIN = 'train'

TEST = 'test'


class BaselineExecutionPlanner(ExecutionPlanner):
    def generate_execution_plan(self, model_snapshots: [ModelSnapshot], dataset_paths: dict) -> ExecutionPlan:
        # the baseline execution plan is a sequential iteration over the models with no reuse of model intermediates or
        # technique to reduce the amount of data processed by the models (e.g. sucessive halving)
        # so the sub-steps are
        # 1) extract test features
        # 2) extract train features
        # 3) score model based on features

        # TODO take this information as parameters
        data_set_class = DatasetClass.CUSTOM_IMAGE_FOLDER
        num_workers = 12
        batch_size = 128

        for snapshot in model_snapshots:
            execution_steps = []
            # extract test features
            test_feature_prefix = f'{snapshot._id}-{TEST}'
            execution_steps.append(
                BaselineExtractFeaturesStep(
                    _id=f'{snapshot._id}-extract-test-0',
                    model_snapshot=snapshot,
                    data_info=DatasetInformation(
                        data_set_class, dataset_paths[TEST], num_workers, batch_size, inference_transform),
                    cache_locations=CacheLocation.SSD,
                    feature_cache_prefix=test_feature_prefix
                )
            )
            # extract train features
            train_feature_prefix = f'{snapshot._id}-{TRAIN}'
            execution_steps.append(
                BaselineExtractFeaturesStep(
                    _id=f'{snapshot._id}-extract-train-0',
                    model_snapshot=snapshot,
                    data_info=DatasetInformation(
                        data_set_class, dataset_paths[TRAIN], num_workers, batch_size, inference_transform),
                    cache_locations=CacheLocation.SSD,
                    feature_cache_prefix=train_feature_prefix
                )
            )
            # score model based on train and test features
            execution_steps.append(
                ScoreModelStep(
                    _id=f'{snapshot._id}-score',
                    scoring_method=ScoringMethod.FC,
                    test_feature_cache_prefixes=[test_feature_prefix],
                    train_feature_cache_prefixes=[train_feature_prefix],
                    num_classes=100
                )
            )

            return ExecutionPlan(execution_steps)
