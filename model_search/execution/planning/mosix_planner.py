from custom.data_loaders.imagenet_transfroms import inference_transform
from global_utils.constants import TEST, TRAIN
from model_search.execution.data_handling.data_information import DatasetInformation, DatasetClass
from model_search.execution.planning.execution_plan import ExecutionPlanner, ExecutionPlan, CacheLocation, \
    BaselineExtractFeaturesStep, ScoreModelStep, ScoringMethod
from model_search.model_snapshot import ModelSnapshot, RichModelSnapshot


class MosixPlannerConfig:
    def __init__(self, num_workers: int, batch_size: int):
        self.num_workers = num_workers
        self.batch_size = batch_size


class MosixExtractFeaturesStep(BaselineExtractFeaturesStep):
    def __init__(
            self,
            _id: str,
            model_snapshot: RichModelSnapshot,
            data_info: DatasetInformation,
            feature_cache_prefix: str,
            data_range: [int],
            cache_indices: [int],
            start_processing_idx: int,
            cache_locations: {int: CacheLocation}
    ):
        super().__init__(_id, model_snapshot, data_info, feature_cache_prefix)
        self.data_range: [int] = data_range
        self.cache_indices: [int] = cache_indices
        self.start_processing_idx: int = start_processing_idx
        self.cache_indices = cache_indices
        self.cache_locations: {int: CacheLocation} = cache_locations


class MosixExecutionPlanner(ExecutionPlanner):

    def __init__(self, config: MosixPlannerConfig):
        self.config: MosixPlannerConfig = config
        self._train_feature_prefixes = {}

    def generate_execution_plan(self, model_snapshots: [ModelSnapshot], dataset_paths: dict,
                                train_dataset_range: [int] = None, first_iteration=False) -> ExecutionPlan:
        # TODO for now there is no planning logic implemented, we just give back one hardcoded execution plan
        #  the caching and reuse indices are made up to test the execution engine

        data_set_class = DatasetClass.CUSTOM_IMAGE_FOLDER
        num_workers = self.config.num_workers
        batch_size = self.config.batch_size

        execution_steps = []
        if first_iteration:
            snapshot = model_snapshots[0]
            test_feature_prefix = f'{snapshot._id}-{TEST}'
            step = MosixExtractFeaturesStep(
                _id=f'{snapshot._id}-extract-test-0',
                model_snapshot=snapshot,
                data_info=DatasetInformation(
                    data_set_class, dataset_paths[TEST], num_workers, batch_size, inference_transform),
                feature_cache_prefix=test_feature_prefix,
                data_range=None,
                cache_indices=[5, 10, 15],
                start_processing_idx=0,
                cache_locations={
                    5: CacheLocation.SSD,
                    10: CacheLocation.CPU,
                    15: CacheLocation.GPU
                }
            )
            execution_steps.append(step)

            start_indices = [5, 10, 15]
            for i in range(3):
                snapshot = model_snapshots[i + 1]
                test_feature_prefix = f'{snapshot._id}-{TEST}'
                step = MosixExtractFeaturesStep(
                    _id=f'{snapshot._id}-extract-test-0',
                    model_snapshot=snapshot,
                    data_info=DatasetInformation(
                        data_set_class, dataset_paths[TEST], num_workers, batch_size, inference_transform),
                    feature_cache_prefix=test_feature_prefix,
                    data_range=None,
                    cache_indices=[],
                    start_processing_idx=start_indices[i],
                    cache_locations={}
                )
                execution_steps.append(step)

        # # for train data we always have to extract features
        # for snapshot in model_snapshots:
        #     # extract train features
        #     train_feature_prefix = self._register_train_feature_prefix(snapshot, train_dataset_range)
        #     execution_steps.append(
        #         MosixExtractFeaturesStep(
        #             _id=f'{snapshot._id}-extract-train-{train_dataset_range[0]}-{train_dataset_range[1]}',
        #             model_snapshot=snapshot,
        #             data_info=DatasetInformation(
        #                 data_set_class, dataset_paths[TRAIN], num_workers, batch_size, inference_transform),
        #             cache_locations=CacheLocation.SSD,
        #             feature_cache_prefix=train_feature_prefix,
        #             data_range=train_dataset_range
        #         )
        #     )
        #     # score model based on train and test features
        #     test_feature_prefix = f'{snapshot._id}-{TEST}'
        #     execution_steps.append(
        #         ScoreModelStep(
        #             _id=f'{snapshot._id}-{SCORE}',
        #             scoring_method=ScoringMethod.FC,
        #             test_feature_cache_prefixes=[test_feature_prefix],
        #             train_feature_cache_prefixes=self._train_feature_prefixes[snapshot._id],
        #             num_classes=100
        #         )
        #     )

        return ExecutionPlan(execution_steps)

    def get_sorted_model_scores(self, plan: ExecutionPlan):
        scores = []
        for step in plan.execution_steps:
            if isinstance(step, ScoreModelStep):
                scores.append([step.execution_result[SCORE], step._id.replace(f'-{SCORE}', '')])

        return sorted(scores)

    def _register_train_feature_prefix(self, snapshot, train_dataset_range):
        train_feature_prefix = f'{snapshot._id}-{TRAIN}-{train_dataset_range[0]}-{train_dataset_range[1]}'
        if not snapshot._id in self._train_feature_prefixes:
            self._train_feature_prefixes[snapshot._id] = []
        self._train_feature_prefixes[snapshot._id].append(train_feature_prefix)
        return train_feature_prefix
