import os

from experiments.side_experiments.end_to_end_times.fine_tuning.plotting.fine_tuning_plt_util import bar_plot_fine_tuning
from global_utils.model_names import RESNET_18, MOBILE_V2, RESNET_50, RESNET_152, VIT_B_16, VIT_L_16


def gen_plots():
    root_dir = os.path.abspath("../results")
    root_out_dir = os.path.abspath("plots")
    model_names = [MOBILE_V2, RESNET_18, RESNET_50, RESNET_152, VIT_B_16, VIT_L_16]
    env_names = ["DES-GPU-SERVER"]
    data_tuples = [[80, 20], [800, 200], [3200, 800]]
    factor = 10 ** -9  # transform ns in s
    fine_tuning_variants = ["full_fine_tuning", "feature_extraction"]

    for env_name in env_names:
        for train_size, val_size in data_tuples:
            for fine_tuning_variant in fine_tuning_variants:
                plot_kwargs = {"save_path": os.path.join(root_out_dir, fine_tuning_variant),
                               "file_name": f"{fine_tuning_variant}-{env_name}-{train_size}-{val_size}",
                               "factor": factor}

                bar_plot_fine_tuning(model_names, root_dir, env_name, train_size, val_size, fine_tuning_variant,
                                     plot_kwargs=plot_kwargs)


if __name__ == '__main__':
    gen_plots()
