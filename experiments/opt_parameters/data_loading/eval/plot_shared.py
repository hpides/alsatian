def get_file_id(batch_size, dataset_type, num_workers, sleep):
    file_id = f"data-loading-exp-params-des-gpu-{num_workers}-batch_size-{batch_size}-sleep-{sleep}-data-{dataset_type}"
    return file_id