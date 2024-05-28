from experiments.model_search.benchmark_level import BenchmarkLevel
from experiments.snapshots.generate import RetrainDistribution
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

    def __str__(self):
        return str(self.__dict__)

    def get_dict(self):
        return self.__dict__
