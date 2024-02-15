class ExpArgs:
    def __init__(self, args, section):
        self.model_name = args[section]['model_name']
        self.result_dir = args[section]['result_dir']
        self.dataset_path = args[section]['dataset_path']
        self.extract_batch_size = args.getint(section, 'extract_batch_size')
        self.classify_batch_size = args.getint(section, 'classify_batch_size')
        self.num_items = args.getint(section, 'num_items')
        self.device = args[section]['device']
        self.split_level = args[section]['split_level']
        self.dataset_type = args[section]['dataset_type']
        self.data_workers = args.getint(section, 'data_workers')
        self.dummy_input_dir = args[section]['dummy_input_dir']

    def __str__(self):
        return str(self.__dict__)

    def get_dict(self):
        return self.__dict__
