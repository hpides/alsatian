import os

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator


def plot_horizontal_normalized_bar_chart(data, ignore=[], title="", save_path=None, file_name=None):
    data = normalize_to_percent(data)

    models = list(data.keys())
    first_key = list(data.keys())[0]
    tasks = list(data[first_key].keys())
    for ign in ignore:
        tasks.remove(ign)
    num_models = len(models)
    num_tasks = len(tasks)

    fig, ax = plt.subplots(figsize=(8, num_models / 2))

    # Specify non-transparent colors
    colors = [
        (0.12156863, 0.46666667, 0.70588235),  # Blue
        (1.0, 0.49803922, 0.05490196),  # Orange
        (0.17254902, 0.62745098, 0.17254902),  # Green
        (0.83921569, 0.15294118, 0.15686275),  # Red
        (0.58039216, 0.40392157, 0.74117647),  # Purple
        (0.54901961, 0.3372549, 0.29411765),  # Brown
        (0.89019608, 0.46666667, 0.76078431),  # Pink
        (0.49803922, 0.49803922, 0.49803922),  # Gray
        (0.7372549, 0.74117647, 0.13333333),  # Yellow
        (0.09019608, 0.74509804, 0.81176471)  # Cyan
    ]

    # Plotting stacked horizontal bars for each model
    for i, (model, task_times) in enumerate(data.items()):

        for ign in ignore:
            del task_times[ign]

        left = 0
        for j, (task, time) in enumerate(task_times.items()):
            ax.barh(i, time, color=colors[j], label=task if i == 0 else None, left=left)
            left += time

    ax.set_yticks(range(num_models))
    ax.set_yticklabels(models, fontsize='large')
    ax.xaxis.set_major_locator(FixedLocator(range(0, 101, 20)))
    ax.set_xticklabels([f"{i}%" for i in range(0, 101, 20)], fontsize='large')
    ax.set_xlim(0, 100)  # Set x-axis limit to ensure it ends at 100%
    ax.set_xlabel('Percentage', fontsize='large')
    ax.set_ylabel('Model', fontsize='large')
    ax.set_title(title, fontsize='large')

    # Create a custom legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(len(tasks))]
    if num_models == 12:
        magic = 1.15
    elif num_models == 4:
        magic = 1.5
    else:
        magic = 1.2
    plt.legend(handles, tasks, loc='upper center', bbox_to_anchor=(0.5, magic), ncol=4, fontsize='large')

    plt.grid(axis='x')

    if save_path:
        # Use bbox_inches='tight' to ensure the legend is not cut off
        path = os.path.join(save_path, f'{file_name}.svg')
        plt.savefig(path, format="svg", bbox_inches='tight')
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
        "Model D": {"Task 1": 10, "Task 2": 10, "Task 3": 10, "Task 4": 10},
        "Model E": {"Task 1": 10, "Task 2": 10, "Task 3": 10, "Task 4": 10},
        "Model F": {"Task 1": 12, "Task 2": 18, "Task 3": 22, "Task 4": 10}
    }
    plot_horizontal_normalized_bar_chart(data, save_path='./', file_name='dummy')
