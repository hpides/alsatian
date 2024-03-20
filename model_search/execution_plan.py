from model_search.model_snapshot import RichModelSnapshot


class SingleModelExecutionStep:
    def __init__(self, model_snapshot: RichModelSnapshot, cache_indices: [int], cache_ids: [str],
                 cache_evict_ids: [str]):
        self.model_snapshot: RichModelSnapshot = model_snapshot
        self.cache_indices: [int] = cache_indices
        self.cache_ids: [str] = cache_ids
        self.cache_evict_ids: [str] = cache_evict_ids


class ExecutionPlan:

    def __init__(self, execution_steps: [SingleModelExecutionStep]):
        self.execution_steps: [SingleModelExecutionStep] = execution_steps


class ExecutionPlanner:

    def generate_execution_plan(self, model_snapshots: [RichModelSnapshot]) -> ExecutionPlan:
        # here we need to insert all the logic to generate the execution plan
        pass
