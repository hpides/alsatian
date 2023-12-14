import os

from experiments.end_to_end_times.calc_features.plotting.calc_features_plt_util import bar_plot_calc_features
from global_utils.model_names import RESNET_18, MOBILE_V2, RESNET_50, RESNET_152, VIT_B_16, VIT_L_16


def gen_plots():
    root_dir = os.path.abspath("../results")
    root_out_dir = os.path.abspath("./plots")
    model_names = [MOBILE_V2, RESNET_18, RESNET_50, RESNET_152, VIT_B_16, VIT_L_16]
    env_names = ["DES-GPU-SERVER"]
    data_tuples = [[80, 20], [800, 200], [3200, 800]]
    factor = 10 ** -9  # transform ns in s

    for env_name in env_names:
        for train_size, val_size in data_tuples:
            plot_kwargs = {"save_path": root_out_dir,
                           "file_name": f"extract-features-{env_name}-{train_size}-{val_size}",
                           "factor": factor}

            bar_plot_calc_features(model_names, root_dir, env_name, train_size, val_size, plot_kwargs=plot_kwargs)


if __name__ == '__main__':
    gen_plots()
