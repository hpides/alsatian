class ExpArgs:
    def __init__(self, args, section):
        self.result_dir = args[section]['result_dir']
        self.device = args[section]['device']
        self.dummy_input_dir = args[section]['dummy_input_dir']
        self.dataset_size = int(args[section]['dataset_size'])

    def __str__(self):
        return str(self.__dict__)

    def get_dict(self):
        return self.__dict__
