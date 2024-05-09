from custom.data_loaders.imagenet_transfroms import inference_transform
from global_utils.constants import TEST, TRAIN
from model_search.execution.data_handling.data_information import DatasetInformation, DatasetClass, \
    CachedDatasetInformation, DataInfo
from model_search.execution.planning.execution_plan import ExecutionPlanner, ExecutionPlan, CacheLocation, \
    ScoreModelStep, ScoringMethod
from model_search.execution.planning.shift_planner import SCORE

from model_search.model_snapshots.dfs_traversal import dfs_execution_plan
from model_search.model_snapshots.multi_model_snapshot import MultiModelSnapshot
from model_search.model_snapshots.rich_snapshot import RichModelSnapshot

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
            model_snapshot: RichModelSnapshot,
            data_info: DataInfo,
            layer_range: [int] = None,
            data_range: [int] = None,
            cache_config: CachingConfig = None,
    ):
        """
        Defines one sub-step of the feature extraction process by defining ...
        :param _id:
        :param model_snapshot: what snapshot should be used for the model
        :param data_info: what data should be used as input
        :param data_range: if only a subset of the data should be used -> specify start and end index of the dataset
        :param layer_range: if only a subpart of the model should be used -> specify the start/end layer/block indices
        if only one number is given it defines the start index with and open end
        :param cache_config: if output shall be cached specify how and where to cache
        """
        self._id = _id
        self.model_snapshot = model_snapshot
        self.data_info = data_info
        self.data_range = data_range
        self.layer_range = layer_range
        self.cache_config = cache_config

    @property
    def cache_labels(self) -> bool:
        # we want to cache also the labels if we actually have to load the original data
        return self.layer_range[0] == 0 and isinstance(self.data_info, DatasetInformation)

    @property
    def open_layer_range(self) -> bool:
        return len(self.layer_range) == 1


class MosixExecutionPlanner(ExecutionPlanner):

    def __init__(self, config: MosixPlannerConfig):
        self.config: MosixPlannerConfig = config
        self._train_feature_prefixes = {}

    def generate_execution_plan(self, mm_snapshot:MultiModelSnapshot) -> ExecutionPlan:

        order = dfs_execution_plan(mm_snapshot.root)
        print('test')

        # determine traversal trough multi model graph

        # then transform traversal into feature extraction and scoring steps



    # def generate_execution_plan(self, model_snapshots: [ModelSnapshot], dataset_paths: dict,
    #                             train_dataset_range: [int] = None, first_iteration=False) -> ExecutionPlan:
    #     # TODO for now there is no planning logic implemented, we just give back one hardcoded execution plan
    #     #  the caching and reuse indices are made up to test the execution engine
    #
    #     data_set_class = DatasetClass.CUSTOM_IMAGE_FOLDER
    #
    #     execution_steps = []
    #     num_workers = self.config.num_workers
    #     batch_size = self.config.batch_size
    #     snapshot = model_snapshots[0]
    #     layer_range = [0, 5]
    #
    #     test_feature_prefix = f'{snapshot.id}-{TEST}-l-{layer_range[-1]}'
    #     step = MosixExtractFeaturesStep(
    #         _id=f'{snapshot.id}-extract-test',
    #         model_snapshot=snapshot,
    #         data_info=DatasetInformation(
    #             data_set_class, dataset_paths[TEST], num_workers, batch_size, inference_transform),
    #         data_range=None,  # use all data
    #         layer_range=layer_range,
    #         cache_config=CachingConfig(id_prefix=test_feature_prefix, location=CacheLocation.SSD)
    #     )
    #     execution_steps.append(step)
    #
    #     data_range = [0, 50]
    #     train_feature_prefix = f'{snapshot.id}-{TRAIN}-l-{layer_range[-1]}-ds-{data_range[0]}-{data_range[1]}'
    #     step = MosixExtractFeaturesStep(
    #         _id=f'{snapshot.id}-extract-train',
    #         model_snapshot=snapshot,
    #         data_info=DatasetInformation(
    #             data_set_class, dataset_paths[TRAIN], num_workers, batch_size, inference_transform),
    #         data_range=data_range,
    #         layer_range=layer_range,
    #         cache_config=CachingConfig(id_prefix=train_feature_prefix, location=CacheLocation.SSD)
    #     )
    #     execution_steps.append(step)
    #     ###################################################
    #     ###################################################
    #
    #     # from here on load cached intermediates so use different batch size and num workers
    #     num_workers = 2
    #     batch_size = 2
    #
    #     layer_range = [5, 7]
    #     test_feature_prefix = f'{snapshot.id}-{TEST}-l-{layer_range[-1]}'
    #     step = MosixExtractFeaturesStep(
    #         _id=f'{snapshot.id}-extract-test',
    #         model_snapshot=snapshot,
    #         data_info=CachedDatasetInformation(
    #             f'{model_snapshots[0].id}-{TEST}-l-{layer_range[0]}', num_workers, batch_size),
    #         data_range=None,  # use all data
    #         layer_range=layer_range,
    #         cache_config=CachingConfig(id_prefix=test_feature_prefix, location=CacheLocation.SSD)
    #     )
    #     execution_steps.append(step)
    #     train_feature_prefix = f'{snapshot.id}-{TRAIN}-l-{layer_range[-1]}-ds-{data_range[0]}-{data_range[1]}'
    #     step = MosixExtractFeaturesStep(
    #         _id=f'{snapshot.id}-extract-train',
    #         model_snapshot=snapshot,
    #         data_info=CachedDatasetInformation(
    #             f'{model_snapshots[0].id}-{TRAIN}-l-{layer_range[0]}-ds-{data_range[0]}-{data_range[1]}', num_workers,
    #             batch_size),
    #         data_range=data_range,
    #         layer_range=layer_range,
    #         cache_config=CachingConfig(id_prefix=train_feature_prefix, location=CacheLocation.SSD)
    #     )
    #     execution_steps.append(step)
    #     ###################################################
    #     ###################################################
    #
    #     layer_range = [7, 10]
    #     test_feature_prefix = f'{snapshot.id}-{TEST}-l-{layer_range[-1]}'
    #     step = MosixExtractFeaturesStep(
    #         _id=f'{snapshot.id}-extract-test',
    #         model_snapshot=snapshot,
    #         data_info=CachedDatasetInformation(
    #             f'{model_snapshots[0].id}-{TEST}-l-{layer_range[0]}', num_workers, batch_size),
    #         data_range=None,  # use all data
    #         layer_range=layer_range,
    #         cache_config=CachingConfig(id_prefix=test_feature_prefix, location=CacheLocation.SSD)
    #     )
    #     execution_steps.append(step)
    #     train_feature_prefix = f'{snapshot.id}-{TRAIN}-l-{layer_range[-1]}-ds-{data_range[0]}-{data_range[1]}'
    #     step = MosixExtractFeaturesStep(
    #         _id=f'{snapshot.id}-extract-train',
    #         model_snapshot=snapshot,
    #         data_info=CachedDatasetInformation(
    #             f'{model_snapshots[0].id}-{TRAIN}-l-{layer_range[0]}-ds-{data_range[0]}-{data_range[1]}', num_workers,
    #             batch_size),
    #         data_range=data_range,
    #         layer_range=layer_range,
    #         cache_config=CachingConfig(id_prefix=train_feature_prefix, location=CacheLocation.SSD)
    #     )
    #     execution_steps.append(step)
    #     ###################################################
    #     ###################################################
    #
    #     layer_range = [10]
    #     test_feature_prefix = f'{snapshot.id}-{TEST}-l-{END}'
    #     step = MosixExtractFeaturesStep(
    #         _id=f'{snapshot.id}-extract-test',
    #         model_snapshot=snapshot,
    #         data_info=CachedDatasetInformation(
    #             f'{model_snapshots[0].id}-{TEST}-l-{layer_range[0]}', num_workers, batch_size),
    #         data_range=None,  # use all data
    #         layer_range=layer_range,
    #         cache_config=CachingConfig(id_prefix=test_feature_prefix, location=CacheLocation.SSD)
    #     )
    #     execution_steps.append(step)
    #     train_feature_prefix = f'{snapshot.id}-{TRAIN}-l-{END}-ds-{data_range[0]}-{data_range[1]}'
    #     step = MosixExtractFeaturesStep(
    #         _id=f'{snapshot.id}-extract-train',
    #         model_snapshot=snapshot,
    #         data_info=CachedDatasetInformation(
    #             f'{model_snapshots[0].id}-{TRAIN}-l-{layer_range[0]}-ds-{data_range[0]}-{data_range[1]}', num_workers,
    #             batch_size),
    #         data_range=data_range,
    #         layer_range=layer_range,
    #         cache_config=CachingConfig(id_prefix=train_feature_prefix, location=CacheLocation.SSD)
    #     )
    #     execution_steps.append(step)
    #     ###################################################
    #     ###################################################
    #
    #     step = ScoreModelStep(
    #         _id=f'{snapshot.id}-{SCORE}',
    #         scoring_method=ScoringMethod.FC,
    #         test_feature_cache_prefixes=[test_feature_prefix],
    #         train_feature_cache_prefixes=[train_feature_prefix],
    #         test_label_feature_cache_prefixes=
    #         train_label_feature_cache_prefixes =
    #         num_classes=100
    #     )
    #     execution_steps.append(step)
    #     ###################################################
    #     ###################################################
    #
    #     snapshot = model_snapshots[1]
    #     layer_range = [10]
    #     test_feature_prefix = f'{snapshot.id}-{TEST}-l-{END}'
    #     step = MosixExtractFeaturesStep(
    #         _id=f'{snapshot.id}-extract-test',
    #         model_snapshot=snapshot,
    #         data_info=CachedDatasetInformation(  # reference previous model snapshot intermediates as input
    #             f'{model_snapshots[0].id}-{TEST}-l-{layer_range[0]}', num_workers, batch_size),
    #         data_range=None,  # use all data
    #         layer_range=layer_range,
    #         cache_config=CachingConfig(id_prefix=test_feature_prefix, location=CacheLocation.SSD)
    #     )
    #     execution_steps.append(step)
    #     train_feature_prefix = f'{snapshot.id}-{TRAIN}-l-{END}-ds-{data_range[0]}-{data_range[1]}'
    #     step = MosixExtractFeaturesStep(
    #         _id=f'{snapshot.id}-extract-train',
    #         model_snapshot=snapshot,
    #         data_info=CachedDatasetInformation(
    #             f'{model_snapshots[0].id}-{TRAIN}-l-{layer_range[0]}-ds-{data_range[0]}-{data_range[1]}', num_workers,
    #             batch_size),
    #         data_range=data_range,
    #         layer_range=layer_range,
    #         cache_config=CachingConfig(id_prefix=train_feature_prefix, location=CacheLocation.SSD)
    #     )
    #     execution_steps.append(step)
    #     print(f'step_id: {snapshot.id}-extract-train')
    #     print(f'expected_prefix: {train_feature_prefix}')
    #     ###################################################
    #     ###################################################
    #
    #     step = ScoreModelStep(
    #         _id=f'{snapshot.id}-{SCORE}',
    #         scoring_method=ScoringMethod.FC,
    #         test_feature_cache_prefixes=[test_feature_prefix],
    #         train_feature_cache_prefixes=[train_feature_prefix],
    #         test_label_feature_cache_prefixes=
    #         train_label_feature_cache_prefixes =
    #         num_classes=100
    #     )
    #     execution_steps.append(step)
        ###################################################
        ###################################################

        #
        # snapshot = model_snapshots[2]
        # layer_range = [7]
        # test_feature_prefix = f'{snapshot._id}-{TEST}-l-{END}'
        # step = MosixExtractFeaturesStep(
        #     _id=f'{snapshot._id}-extract-test',
        #     model_snapshot=snapshot,
        #     data_info=CachedDatasetInformation(  # reference previous model snapshot intermediates as input
        #         f'{model_snapshots[0]._id}-{TEST}-l-{layer_range[0]}', num_workers, batch_size),
        #     data_range=None,  # use all data
        #     layer_range=layer_range,
        #     cache_config=CachingConfig(id_prefix=test_feature_prefix, location=CacheLocation.SSD)
        # )
        # execution_steps.append(step)
        #
        # snapshot = model_snapshots[3]
        # layer_range = [5]
        # test_feature_prefix = f'{snapshot._id}-{TEST}-l-{END}'
        # step = MosixExtractFeaturesStep(
        #     _id=f'{snapshot._id}-extract-test',
        #     model_snapshot=snapshot,
        #     data_info=CachedDatasetInformation(  # reference previous model snapshot intermediates as input
        #         f'{model_snapshots[0]._id}-{TEST}-l-{layer_range[0]}', num_workers, batch_size),
        #     data_range=None,  # use all data
        #     layer_range=layer_range,
        #     cache_config=CachingConfig(id_prefix=test_feature_prefix, location=CacheLocation.SSD)
        # )
        # execution_steps.append(step)

        # TODO train phase and model scoring is missing, also automated generation of plan and best order is missing

        # return ExecutionPlan(execution_steps)

    def get_sorted_model_scores(self, plan: ExecutionPlan):
        scores = []
        for step in plan.execution_steps:
            if isinstance(step, ScoreModelStep):
                scores.append([step.execution_result[SCORE], step._id.replace(f'-{SCORE}', '')])

        return sorted(scores)

    def _register_train_feature_prefix(self, snapshot, train_dataset_range):
        train_feature_prefix = f'{snapshot.id}-{TRAIN}-{train_dataset_range[0]}-{train_dataset_range[1]}'
        if not snapshot.id in self._train_feature_prefixes:
            self._train_feature_prefixes[snapshot.id] = []
        self._train_feature_prefixes[snapshot.id].append(train_feature_prefix)
        return train_feature_prefix
