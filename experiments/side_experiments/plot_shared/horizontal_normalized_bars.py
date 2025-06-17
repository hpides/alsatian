import os

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

from experiments.plot_util import HPI_LIGHT_ORANGE, HPI_ORANGE, HPI_RED, PURPLE


def plot_horizontal_normalized_bar_chart_with_given_axis(data, ax, ignore=[], title="", items=None):
    colors = [HPI_LIGHT_ORANGE, HPI_ORANGE, HPI_RED, PURPLE]

    data = normalize_to_percent(data)

    models = list(data.keys())
    first_key = list(data.keys())[0]
    tasks = list(data[first_key].keys())
    for ign in ignore:
        tasks.remove(ign)
    num_models = len(models)

    # Plotting stacked horizontal bars for each model
    for i, (model, task_times) in enumerate(data.items()):

        for ign in ignore:
            del task_times[ign]

        left = 0
        for j, (task, time) in enumerate(task_times.items()):
            ax.barh(i, time, color=colors[j], label=task if i == 0 else None, left=left)
            left += time

    ax.set_yticks(range(num_models))
    ax.set_yticklabels(models)
    ax.xaxis.set_major_locator(FixedLocator(range(0, 101, 20)))
    ax.set_xticklabels([f"{i}%" for i in range(0, 101, 20)])
    ax.set_xlim(0, 100)  # Set x-axis limit to ensure it ends at 100%
    x_label = 'Time distribution'
    # if items is not None:
    #     if items < 100:
    #         x_label += ", ~100 items"
    #     elif 9000 < items < 10000:
    #         x_label += ", ~10,000 items"
    ax.set_xlabel(x_label, labelpad=20)  # Adjust labelpad as needed
    # ax.set_ylabel('Model', fontsize='large')
    ax.set_title(title)

    plt.tight_layout()



def plot_horizontal_normalized_bar_chart(data, ignore=[], title="", save_path=None, file_name=None, legend=True):
    FONT_SIZE = 18
    plt.rcParams.update({
        'font.size': FONT_SIZE,  # General font size
        'axes.titlesize': FONT_SIZE,  # Title font size
        'axes.labelsize': FONT_SIZE,  # X and Y label font size
        'xtick.labelsize': FONT_SIZE,  # X tick labels font size
        'ytick.labelsize': FONT_SIZE,  # Y tick labels font size
        'legend.fontsize': FONT_SIZE,  # Legend font size
    })

    data = normalize_to_percent(data)

    models = list(data.keys())
    first_key = list(data.keys())[0]
    tasks = list(data[first_key].keys())
    for ign in ignore:
        tasks.remove(ign)
    num_models = len(models)
    num_tasks = len(tasks)

    fig, ax = plt.subplots(figsize=(8, num_models / 2))

    colors = [HPI_LIGHT_ORANGE, HPI_ORANGE, HPI_RED, PURPLE]
    colors = ['#bae4bc', '#7bccc4', '#43a2ca', '#0868ac']

    # Plotting stacked horizontal bars for each model
    for i, (model, task_times) in enumerate(data.items()):

        for ign in ignore:
            del task_times[ign]

        left = 0
        for j, (task, time) in enumerate(task_times.items()):
            ax.barh(i, time, color=colors[j], label=task if i == 0 else None, left=left)
            left += time

    ax.set_yticks(range(num_models))
    ax.set_yticklabels(models)
    ax.xaxis.set_major_locator(FixedLocator(range(0, 101, 20)))
    ax.set_xticklabels([f"{i}%" for i in range(0, 101, 20)])
    ax.set_xlim(0, 100)  # Set x-axis limit to ensure it ends at 100%
    ax.set_xlabel('Time distribution', labelpad=10)  # Adjust labelpad as needed
    # ax.set_ylabel('Model', fontsize='large')
    ax.set_title(title)

    # Create a custom legend
    if legend:
        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(len(tasks))]
        plt.legend(handles, tasks, loc='upper center', bbox_to_anchor=(0.38, 1.5), ncol=4)
    else:
        # Create and save the legend as a separate image
        fig_legend, ax_legend = plt.subplots(figsize=(8, 2))
        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(len(tasks))]
        # tasks = ["prep. model", "prep. data", "infer.", "p. scoring"]
        tasks = ["prepare model", "prepare data", "inference", "proxy scoring"]
        ax_legend.legend(handles, tasks, loc='center', ncol=len(tasks), frameon=False)
        ax_legend.axis('off')  # Hide the axes

        if save_path and file_name:
            legend_path = os.path.join(save_path, f'legend.png')
            fig_legend.savefig(legend_path, format="png", bbox_inches='tight')
            legend_path = os.path.join(save_path, f'legend.svg')
            fig_legend.savefig(legend_path, bbox_inches='tight')
            legend_path = os.path.join(save_path, f'legend.pdf')
            fig_legend.savefig(legend_path, bbox_inches='tight')

        plt.close(fig_legend)

    plt.grid(axis='x')

    if save_path:
        # Use bbox_inches='tight' to ensure the legend is not cut off
        path = os.path.join(save_path, f'{file_name}.pdf')
        plt.savefig(path, format="pdf", bbox_inches='tight')
        path = os.path.join(save_path, f'{file_name}.png')
        plt.savefig(path, format="png", bbox_inches='tight')

    # plt.show()
    plt.close()


def normalize_to_percent(data):
    normalized_data = {}

    for model, tasks in data.items():
        total = sum(tasks.values())
        normalized_tasks = {task: (value / total) * 100 for task, value in tasks.items()}
        normalized_data[model] = normalized_tasks

    return normalized_data


if __name__ == '__main__':
    data = {
        "Model A": {"Task 1": 25, "Task 2": 25, "Task 3": 25, "Task 4": 25},
        "Model B": {"Task 1": 10, "Task 2": 10, "Task 3": 10, "Task 4": 10},
        "Model C": {"Task 1": 10, "Task 2": 10, "Task 3": 10, "Task 4": 10},
    }
    plot_horizontal_normalized_bar_chart(data,
                                         save_path='/Users/nils/uni/programming/model-search-paper/experiments/main_experiments/bottlenecks/model_rank/eval/',
                                         file_name='dummy-1')

    data = {
        "Model D": {"Task 1": 10, "Task 2": 10, "Task 3": 10, "Task 4": 10},
        "Model E": {"Task 1": 10, "Task 2": 10, "Task 3": 10, "Task 4": 10},
        "Model F": {"Task 1": 12, "Task 2": 18, "Task 3": 22, "Task 4": 10}
    }
    plot_horizontal_normalized_bar_chart(data,
                                         save_path='/Users/nils/uni/programming/model-search-paper/experiments/main_experiments/bottlenecks/model_rank/eval/',
                                         file_name='dummy-2')
