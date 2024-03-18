class ExpArgs:
    def __init__(self, args, section):
        self.result_dir = args[section]['result_dir']
        self.dataset_path = args[section]['dataset_path']
        self.batch_size = args.getint(section, 'batch_size')
        self.dummy_input_dir = args[section]['dummy_input_dir']

    def __str__(self):
        return str(self.__dict__)

    def get_dict(self):
        return self.__dict__
