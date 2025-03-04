from experiments.main_experiments.model_search.benchmark_level import BenchmarkLevel
from experiments.main_experiments.snapshots.synthetic.generate import RetrainDistribution
from model_search.execution.planning.execution_plan import CacheLocation


def _str_to_distribution(dist_str) -> RetrainDistribution:
    if dist_str == "TOP_LAYERS":
        return RetrainDistribution.TOP_LAYERS
    elif dist_str == "HARD_CODED":
        return RetrainDistribution.HARD_CODED
    elif dist_str == "RANDOM":
        return RetrainDistribution.RANDOM
    elif dist_str == "TWENTY_FIVE_PERCENT":
        return RetrainDistribution.TWENTY_FIVE_PERCENT
    elif dist_str == "FIFTY_PERCENT":
        return RetrainDistribution.FIFTY_PERCENT
    elif dist_str == "LAST_ONE_LAYER":
        return RetrainDistribution.LAST_ONE_LAYER
    else:
        raise ValueError(f"Unknown distribution string: {dist_str}")


def _str_to_benchmark_level(dist_str) -> BenchmarkLevel:
    if dist_str == "END_TO_END":
        return BenchmarkLevel.END_TO_END
    elif dist_str == "SH_PHASES":
        return BenchmarkLevel.SH_PHASES
    elif dist_str == "EXECUTION_STEPS":
        return BenchmarkLevel.EXECUTION_STEPS
    elif dist_str == "STEPS_DETAILS":
        return BenchmarkLevel.STEPS_DETAILS
    else:
        raise ValueError(f"Unknown distribution string: {dist_str}")


def _str_to_cache_location(location_str) -> CacheLocation:
    if location_str == "SSD":
        return CacheLocation.SSD
    elif location_str == "CPU":
        return CacheLocation.CPU
    elif location_str == "GPU":
        return CacheLocation.GPU


class ExpArgs:
    def __init__(self, args, section):
        self.train_data = args[section]['train_data']
        self.test_data = args[section]['test_data']
        self.num_train_items = args.getint(section, 'num_train_items')
        self.num_test_items = args.getint(section, 'num_test_items')
        self.num_workers = args.getint(section, 'num_workers')
        self.batch_size = args.getint(section, 'batch_size')
        self.num_target_classes = args.getint(section, 'num_target_classes')
        self.persistent_caching_path = args[section]['persistent_caching_path']
        self.base_snapshot_save_path = args[section]['base_snapshot_save_path']
        self.num_models = args.getint(section, 'num_models')
        self.distribution = _str_to_distribution(args[section]['distribution'])
        self.snapshot_set_string = args[section]['snapshot_set_string']
        self.approach = args[section]['approach']
        self.result_dir = args[section]['result_dir']
        self.benchmark_level = _str_to_benchmark_level(args[section]['benchmark_level'])
        self.default_cache_location = _str_to_cache_location(args[section]['default_cache_location'])
        self.limit_fs_io = args.getboolean(section, 'limit_fs_io')
        self.ssd_caching_active = args.getboolean(section, 'ssd_caching_active')
        self.cache_size = args.getint(section, 'cache_size')
        self.trained_snapshots = args.getboolean(section, 'trained_snapshots')
        # self.model_name = args[section].get('model_name', None)
        self.hf_snapshots = self._get_optional_field('hf_snapshots', args, section)
        self.load_full = self._get_optional_field('load_full', args, section)
        self.hf_caching_path =args[section]['hf_caching_path']

    def __str__(self):
        return str(self.__dict__)

    def get_dict(self):
        return self.__dict__

    def _get_optional_field(self, field_name, args, section):
        if field_name in args[section]:
            return args.getboolean(section, field_name)
        return False
