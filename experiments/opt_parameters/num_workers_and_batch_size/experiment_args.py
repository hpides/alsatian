class ExpArgs:
    def __init__(self, args, section):
        self.dataset_path = args[section]['dataset_path']
        self.dummy_input_dir = args[section]['dummy_input_dir']
        self.device = args[section]['device']
        self.result_dir = args[section]['result_dir']

    def __str__(self):
        return str(self.__dict__)

    def get_dict(self):
        return self.__dict__
