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
            test_feature_prefix = f'{snapshot.id}-{TEST}'
            execution_steps.append(
                BaselineExtractFeaturesStep(
                    model_snapshot=snapshot,
                    data_info=DatasetInformation(data_set_class, dataset_paths[TEST], num_workers, batch_size),
                    cache_locations=CacheLocation.SSD,
                    feature_cache_prefix=test_feature_prefix
                )
            )
            # extract train features
            train_feature_prefix = f'{snapshot.id}-{TRAIN}'
            execution_steps.append(
                BaselineExtractFeaturesStep(
                    model_snapshot=snapshot,
                    data_info=DatasetInformation(data_set_class, dataset_paths[TRAIN], num_workers, batch_size),
                    cache_locations=CacheLocation.SSD,
                    feature_cache_prefix=snapshot.id
                )
            )
            # score model based on train and test features
            execution_steps.append(
                ScoreModelStep(
                    scoring_method=ScoringMethod.FC,
                    test_feature_cache_prefixes=test_feature_prefix,
                    train_feature_cache_prefixes=train_feature_prefix,
                )
            )
