import os

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

from experiments.plot_util import HPI_LIGHT_ORANGE, HPI_ORANGE, HPI_RED, PURPLE
from experiments.side_experiments.plot_shared.horizontal_normalized_bars import normalize_to_percent


import matplotlib.gridspec as gridspec

def plot_combined_horizontal_bar_chart_with_side_plot(data_list, titles, side_data1, side_data2, side_title,
                                                      save_path=None, file_name=None, ignore=[], legend=True):
    FONT_SIZE = 50
    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'axes.titlesize': FONT_SIZE,
        'axes.labelsize': FONT_SIZE,
        'xtick.labelsize': FONT_SIZE,
        'ytick.labelsize': FONT_SIZE,
        'legend.fontsize': FONT_SIZE,
    })

    colors = ['#bae4bc', '#7bccc4', '#43a2ca', '#0868ac']
    num_main_plots = len(data_list)

    fig = plt.figure(figsize=(80, 15.5))
    spacing = 0.2
    gs = gridspec.GridSpec(num_main_plots, 11, width_ratios=[3, spacing, 1, 1, 1, spacing, 1, 1, 1, spacing, 1])

    axes = [fig.add_subplot(gs[i, 0]) for i in range(num_main_plots)]
    side_ax1 = fig.add_subplot(gs[:, 2])  # spans all rows
    side_ax2 = fig.add_subplot(gs[:, 3])  # spans all rows
    side_ax3 = fig.add_subplot(gs[:, 4])  # spans all rows
    side_ax4 = fig.add_subplot(gs[:, 6])  # spans all rows
    side_ax5 = fig.add_subplot(gs[:, 7])  # spans all rows
    side_ax6 = fig.add_subplot(gs[:, 8])  # spans all rows
    side_ax7 = fig.add_subplot(gs[0, 10])  # spans all rows
    side_ax8 = fig.add_subplot(gs[1, 10])  # spans all rows

    # Main plots
    for idx, (data, ax) in enumerate(zip(data_list, axes)):
        data = normalize_to_percent(data)
        models = list(data.keys())
        first_key = models[0]
        tasks = list(data[first_key].keys())
        for ign in ignore:
            tasks.remove(ign)
        num_models = len(models)

        for i, (model, task_times) in enumerate(data.items()):
            for ign in ignore:
                del task_times[ign]
            left = 0
            for j, (task, time) in enumerate(task_times.items()):
                ax.barh(i, time, color=colors[j], label=task if i == 0 and idx == 0 else None, left=left)
                left += time

        ax.set_yticks(range(num_models))
        ax.set_yticklabels(models)
        ax.set_xlim(0, 100)
        ax.set_title(titles[idx])
        ax.grid(axis='x')

        if idx == num_main_plots - 1:
            ax.set_xlabel('Time distribution')
        else:
            ax.set_xticklabels([])

    # Legend
    if legend:
        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(len(tasks))]
        fig.legend(handles, tasks, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=4)

    # Side plot (data_3)
    categories = list(side_data1.keys())
    values = list(side_data1.values())
    bars = side_ax1.bar(categories, values, color='#a6bddb')
    side_ax1.set_title(side_title)
    side_ax1.set_ylabel("Some Value")
    side_ax1.set_ylim(0, max(values) * 1.2)
    side_ax1.grid(axis='y')

    side_plot_2(side_ax2, side_data2)
    side_plot_2(side_ax3, side_data2)
    side_plot_2(side_ax4, side_data2)
    side_plot_2(side_ax5, side_data2)
    side_plot_2(side_ax6, side_data2)
    side_plot_2(side_ax7, side_data2)
    side_plot_2(side_ax8, side_data2)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path and file_name:
        fig.savefig(os.path.join(save_path, f'{file_name}.pdf'), format="pdf", bbox_inches='tight')
        fig.savefig(os.path.join(save_path, f'{file_name}.png'), format="png", bbox_inches='tight')

    plt.close(fig)


def side_plot_2(side_ax2, side_data2, y_axis_label=False):
    # Side plot 2 (data_4)
    categories = list(side_data2.keys())
    values = list(side_data2.values())
    bars = side_ax2.bar(categories, values, color='#a6bddb')
    side_ax2.set_title("side title 2")
    if y_axis_label:
        side_ax2.set_ylabel("Some Value")
    side_ax2.set_ylim(0, max(values) * 1.2)
    # side_ax2.grid(axis='y')


if __name__ == '__main__':
    data1 = {
        "Model A": {"Task 1": 25, "Task 2": 25, "Task 3": 25, "Task 4": 25},
        "Model B": {"Task 1": 10, "Task 2": 10, "Task 3": 10, "Task 4": 10},
        "Model C": {"Task 1": 10, "Task 2": 10, "Task 3": 10, "Task 4": 10},
    }

    data2 = {
        "Model D": {"Task 1": 10, "Task 2": 10, "Task 3": 10, "Task 4": 10},
        "Model E": {"Task 1": 10, "Task 2": 10, "Task 3": 10, "Task 4": 10},
        "Model F": {"Task 1": 12, "Task 2": 18, "Task 3": 22, "Task 4": 10}
    }

    data_3 = {
        "Cat A": 10,
        "Cat B": 20,
        "Cat C": 30,
    }

    data_4 = {
        "Cat A": 10,
        "Cat B": 20,
        "Cat C": 30,
    }

    save_path = '/Users/nils/uni/programming/model-search-paper/poster_plots/plots'
    plot_combined_horizontal_bar_chart_with_side_plot(
        [data1, data2],
        titles=["Dummy 1", "Dummy 2"],
        side_data1=data_3,
        side_data2=data_4,
        side_title="Summary Chart",
        save_path=save_path,
        file_name="combined-dummy-with-side"
    )
