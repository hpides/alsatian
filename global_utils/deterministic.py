import os
import random

import numpy as np
import torch

DETERMINISTIC_EXECUTION = "DETERMINISTIC_EXECUTION"
TRUE = "True"


def check_deterministic_env_var_set():
    if DETERMINISTIC_EXECUTION in os.environ:
        return os.environ[DETERMINISTIC_EXECUTION] == TRUE
    else:
        return False


def set_deterministic():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
