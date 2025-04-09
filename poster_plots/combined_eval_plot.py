import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from experiments.main_experiments.bottlenecks.model_rank.eval.plotting import get_bottleneck_data
from experiments.main_experiments.model_search.eval.hf_snapshots.plot_hf_combined_snapshot import \
    plot_end_to_end_times_given_axis
from experiments.main_experiments.model_search.eval.limited_memory import plot_limited_memory
from experiments.main_experiments.model_search.eval.synthetic_snapshots.plot_time_breakdown import \
    plot_approaches_across_memory_config_given_axis, plot_fixed_approach_changed_config_given_axis
from experiments.plot_util import HPI_LIGHT_ORANGE, HPI_ORANGE, HPI_RED, PURPLE
from experiments.side_experiments.plot_shared.horizontal_normalized_bars import \
    plot_horizontal_normalized_bar_chart_with_given_axis
from global_utils.model_names import RESNET_18, RESNET_152, EFF_NET_V2_L, VIT_L_32


def plot_combined_horizontal_bar_chart_with_side_plot(save_path=None, file_name=None):
    FONT_SIZE = 50
    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'axes.titlesize': FONT_SIZE,
        'axes.labelsize': FONT_SIZE,
        'xtick.labelsize': FONT_SIZE,
        'ytick.labelsize': FONT_SIZE,
        'legend.fontsize': FONT_SIZE,
    })

    colors = [HPI_LIGHT_ORANGE, HPI_ORANGE, HPI_RED, PURPLE]

    fig = plt.figure(figsize=(90, 15.5))
    spacing = 0.05
    gs = gridspec.GridSpec(2, 11, width_ratios=[3, spacing, 1, 1, 1, spacing, 1, 1, 1, spacing, 1])

    # Legend
    # Manually calculate horizontal center of the 6th column
    # The 6th column index in the layout is 6 (0-based)
    # Total width = sum of width_ratios
    total_width = sum([3, spacing, 1, 1, 1, spacing, 1, 1, 1, spacing, 1])
    center_of_col6 = sum([3, spacing, 1, 1, 1, spacing, 1 / 2]) / total_width  # halfway into column 6

    tasks = ["prepare model", "prepare data", "inference", "proxy scoring"]
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(len(tasks))]
    legend = fig.legend(handles, tasks, loc='upper center', bbox_to_anchor=(center_of_col6, 1.02), ncol=4)
    bbox = legend.get_window_extent()  # Returns in display (pixel) coordinates
    # Convert to figure coordinates (0–1)
    bbox_fig = bbox.transformed(fig.transFigure.inverted())
    # You can then access width and height like this:
    legend_width = bbox_fig.width
    legend.remove()
    center_of_col6 = center_of_col6 - legend_width / 2
    fig.legend(handles, tasks, loc='upper center', bbox_to_anchor=(center_of_col6, 1.02), ncol=4)

    legend_elements = [
        Patch(label='B - Baseline'),
        Patch(label='S - SHiFT'),
        Patch(label='A - Alsatian')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3,
               handlelength=0, handletextpad=0.5)

    legend_elements = [
        Patch(label='50 - Last 50%'),
        Patch(label='25 - Last 25%'),
        Patch(label='TOP - Top few')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3,
               handlelength=0, handletextpad=0.5)

    # first and second bottleneck plot
    bottleneck_axis_top = fig.add_subplot(gs[0, 0])
    bottleneck_axis_bottom = fig.add_subplot(gs[1, 0])
    for items, axis in zip([3 * 32, 9 * 1024], [bottleneck_axis_top, bottleneck_axis_bottom]):
        root_dir = os.path.abspath(
            '/Users/nils/uni/programming/model-search-paper/experiments/main_experiments/bottlenecks/model_rank/results/bottleneck-analysis')
        file_template = 'bottleneck_analysis-model-{}-items-{}-split-{}-dataset_type-{}'
        model_names = [RESNET_18, RESNET_152, EFF_NET_V2_L, VIT_L_32]
        data, _, ignore = get_bottleneck_data('imagenette.', 200, model_names, items, None, file_template, root_dir)
        plot_horizontal_normalized_bar_chart_with_given_axis(data, axis, ignore=ignore)

    # breakdown plots
    breakdown_axis_left = fig.add_subplot(gs[:, 2])
    breakdown_axis_middle = fig.add_subplot(gs[:, 3])
    breakdown_axis_right = fig.add_subplot(gs[:, 4])

    root_dir = os.path.abspath(
        '/Users/nils/uni/programming/model-search-paper/experiments/main_experiments/model_search/eval/synthetic_snapshots/results/des-gpu-imagenette-synthetic-snapshots')
    model = RESNET_152
    device = "CPU"
    approaches = ['baseline', 'shift', 'mosix']

    plot_approaches_across_memory_config_given_axis(root_dir, model, 2000, device, "TOP_LAYERS", approaches,
                                                    breakdown_axis_left, y_label=True)
    plot_approaches_across_memory_config_given_axis(root_dir, model, 8000, device, "TOP_LAYERS", approaches,
                                                    breakdown_axis_middle)
    plot_fixed_approach_changed_config_given_axis(root_dir, model, 8000, device,
                                                  ["FIFTY_PERCENT", "TWENTY_FIVE_PERCENT", "TOP_LAYERS"], 'mosix',
                                                  breakdown_axis_right)

    # limited memory plots
    output_path = './plots'
    model_distribution = "FIFTY_PERCENT"
    root_dir = os.path.abspath(f'/Users/nils/uni/programming/model-search-paper/experiments/main_experiments/model_search/eval/limited_memory/results/limited-memory-exps')
    model_dist = "FIFTY_PERCENT"

    lim_mem_axis_left = fig.add_subplot(gs[:, 6])
    approach = 'baseline'
    plot_limited_memory.plot_approaches_across_memory_config_given_axis(root_dir, model_distribution, model_dist, output_path, approach,
                                                    lim_mem_axis_left)

    lim_mem_axis_middle = fig.add_subplot(gs[:, 7])
    approach = 'shift'
    plot_limited_memory.plot_approaches_across_memory_config_given_axis(root_dir, model_distribution, model_dist,
                                                                        output_path, approach,
                                                                        lim_mem_axis_middle, x_label=True)

    lim_mem_axis_right = fig.add_subplot(gs[:, 8])
    approach = 'mosix'
    plot_limited_memory.plot_approaches_across_memory_config_given_axis(root_dir, model_distribution, model_dist,
                                                                        output_path, approach,
                                                                        lim_mem_axis_right)
    # hugging face model plots
    hf_axis_top = fig.add_subplot(gs[0, 10])
    root_dir = os.path.abspath(
        '/Users/nils/uni/programming/model-search-paper/experiments/main_experiments/model_search/eval/hf_snapshots/results')
    file_template = 'des-gpu-imagenette-huggingface-all-hf-architecture-search#approach#{}#cache#CPU#snapshot#{}#models#-1#items#{}#level#{}'
    models = ['combined']
    approaches = ['baseline', 'shift', 'mosix']
    measure_type = 'EXECUTION_STEPS'
    plot_end_to_end_times_given_axis(root_dir, file_template, models, approaches, 2000, measure_type, hf_axis_top)

    hf_axis_bottom = fig.add_subplot(gs[1, 10])
    plot_end_to_end_times_given_axis(root_dir, file_template, models, approaches, 8000, measure_type, hf_axis_bottom)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path and file_name:
        fig.savefig(os.path.join(save_path, f'{file_name}.pdf'), format="pdf", bbox_inches='tight')
        fig.savefig(os.path.join(save_path, f'{file_name}.png'), format="png", bbox_inches='tight')

    plt.close(fig)


if __name__ == '__main__':
    save_path = '/Users/nils/uni/programming/model-search-paper/poster_plots/plots'
    plot_combined_horizontal_bar_chart_with_side_plot(
        save_path=save_path,
        file_name="combined-dummy-with-side"
    )
