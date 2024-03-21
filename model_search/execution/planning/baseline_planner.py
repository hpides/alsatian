from model_search.execution.planning.execution_plan import ExecutionPlanner, ExecutionPlan
from model_search.model_snapshot import ModelSnapshot


class BaselineExecutionPlanner(ExecutionPlanner):
    def generate_execution_plan(self, model_snapshots: [ModelSnapshot]) -> ExecutionPlan:
        # here we need to insert all the logic to generate the execution plan
        pass
