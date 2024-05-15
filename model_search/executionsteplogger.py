from model_search.execution.planning.execution_plan import ExecutionStep


class ExecutionStepLogger:

    def __init__(self):
        self.log_dict = {}
        self.log_id = None

    def set_exec_step(self, execution_step: ExecutionStep):
        self.log_id = execution_step._id

    def log_value(self, _key, _value):
        self.log_dict[_key] = _value

    def append_value(self, _key, _value):
        if _key not in self.log_dict:
            self.log_dict[_key] = []

        self.log_dict[_key].append(_value)
