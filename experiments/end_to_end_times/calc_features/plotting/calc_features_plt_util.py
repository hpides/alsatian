def get_times_after_epochs(epoch, root_dir, model_name, classes, batch_size, train_size, val_size):
    data = get_raw_data(root_dir, model_name, classes, batch_size, train_size, val_size)
    extracted = {}
    extracted['init_model'] = data[MEASUREMENTS][NULL][INIT]
    train_details = data[MEASUREMENTS][NULL][TRAIN_DETAIL]
    # we always ignore epoch 1, to have consistent results
    train_times = [train_details[f'epoch-{i}-phase-train'] for i in range(1, epoch + 1)]
    val_times = [train_details[f'epoch-{i}-phase-val'] for i in range(1, epoch + 1)]
    extracted['train'] = sum(train_times)
    extracted['val'] = sum(val_times)
    return extracted
