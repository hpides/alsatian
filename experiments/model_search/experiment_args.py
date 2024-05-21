class ExpArgs:
    def __init__(self, args, section):
        self.train_data = args[section]['train_data']
        self.test_data = args[section]['test_data']
        self.num_workers = args.getint(section, 'num_workers')
        self.batch_size = args.getint(section, 'batch_size')
        self.num_target_classes = args.getint(section, 'num_target_classes')
        self.persistent_caching_path = args[section]['persistent_caching_path']
        self.base_snapshot_save_path = args[section]['base_snapshot_save_path']
        self.snapshot_set_string = args[section]['snapshot_set_string']
        self.approach = args[section]['approach']
        self.result_dir = args[section]['result_dir']


    def __str__(self):
        return str(self.__dict__)

    def get_dict(self):
        return self.__dict__
