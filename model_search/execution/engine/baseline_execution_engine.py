from abc import ABC, abstractmethod

from model_search.execution.planning.execution_plan import ExecutionPlan, SingleModelExecutionStep


class ExecutionEngine(ABC):

    def execute_plan(self, execution_plan: ExecutionPlan):
        for exex_step in execution_plan.execution_steps:
            self.execute_step(exex_step)

    @abstractmethod
    def execute_step(self, exex_step: SingleModelExecutionStep):
        pass
