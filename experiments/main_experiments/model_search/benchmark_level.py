from enum import Enum


class BenchmarkLevel(Enum):
    END_TO_END = 1
    SH_PHASES = 2
    EXECUTION_STEPS = 3
    STEPS_DETAILS = 4
