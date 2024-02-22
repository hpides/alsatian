import os

import matplotlib.pyplot as plt


def plot_stacked_bar_chart(data, ignore=[], factor=1, title="", save_path=None, file_name=None, distance_in_data=None):
    models = list(data.keys())
    tasks = list(data[models[0]].keys())
    for i in ignore:
        if i in tasks:
            tasks.remove(i)

    task_values = {task: [data[model][task] * factor for model in models] for task in tasks}

    model_sums = [sum(x) * factor for x in [[data[model][task] for task in tasks] for model in models]]

    fig, ax = plt.subplots(figsize=(8, 6))  # Adjust the figure size as needed
    bottom = [0] * len(models)

    for task in tasks:
        bars = ax.bar(models, task_values[task], label=task, bottom=bottom)
        bottom = [bottom[i] + task_values[task][i] for i in range(len(models))]

    # Plot the sum on top of each stacked bar
    for model, total in zip(models, model_sums):
        y_min, y_max = ax.get_ylim()
        distance_in_cm = 0.15
        distance_in_data = distance_in_cm / (y_max - y_min) if not distance_in_data else distance_in_data
        y_position = total + distance_in_data
        ax.text(models.index(model), y_position, '{:.4f}'.format(total),
                ha='center', va='center', fontweight='bold', color='black')

    ax.set_ylabel('Time')
    ax.set_title(title)
    plt.xticks(rotation=45)

    # Adjust the legend position and add a title
    legend = ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Legend")
    legend.get_title().set_fontweight('bold')

    if save_path:
        # Use bbox_inches='tight' to ensure the legend is not cut off
        path = os.path.join(save_path, f'{file_name}.svg')
        plt.savefig(path, format="svg", bbox_inches='tight')
        path = os.path.join(save_path, f'{file_name}.png')
        plt.savefig(path, format="png", bbox_inches='tight')


    print(model_sums)
    plt.show()


if __name__ == '__main__':
    # Creating dummy data
    data = {
        "Model A": {"Task 1": 10, "Task 2": 15, "Task 3": 20},
        "Model B": {"Task 1": 8, "Task 2": 12, "Task 3": 18},
        "Model C": {"Task 1": 12, "Task 2": 18, "Task 3": 22}
    }

    # Plotting the dummy data
    plot_stacked_bar_chart(data, title="Dummy Data Stacked Bar Chart", save_path='./', file_name='dummy')
