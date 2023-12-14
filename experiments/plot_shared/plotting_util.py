import os

import matplotlib.pyplot as plt
import numpy as np


def plot_stacked_bar_chart(data, ignore=[], factor=1, title="", save_path=None, file_name=None, distance_in_data=None):
    categories = list(data.keys())
    sub_categories = list(data[categories[0]].keys())
    for i in ignore:
        if i in sub_categories:
            sub_categories.remove(i)

    sub_category_values = {sub_category: [data[category][sub_category] * factor for category in categories] for
                           sub_category in sub_categories}

    category_sums = [sum(x) * factor for x in
                     [[data[category][sub_category] for sub_category in sub_categories] for category in categories]]

    fig, ax = plt.subplots(figsize=(5, 6))  # Adjust the figure size as needed
    bottom = [0] * len(categories)

    for sub_category in sub_categories:
        bars = ax.bar(categories, sub_category_values[sub_category], bottom=bottom, label=sub_category)
        bottom = [bottom[i] + sub_category_values[sub_category][i] for i in range(len(categories))]

    # Plot the sum on top of each stacked bar
    for category, total in zip(categories, category_sums):
        y_min, y_max = ax.get_ylim()
        distance_in_cm = 0.15
        distance_in_data = distance_in_cm / (y_max - y_min) if not distance_in_data else distance_in_data
        y_position = total + distance_in_data
        ax.text(categories.index(category), y_position, '{:.4f}'.format(total),
                ha='center', va='center', fontweight='bold', color='black')

    ax.set_ylabel('Time')
    ax.set_title(title)
    plt.xticks(rotation=45)

    # Adjust the legend position and add a title
    legend = ax.legend(reversed(ax.get_legend_handles_labels()[0]), reversed(ax.get_legend_handles_labels()[1]),
                       loc='upper left', bbox_to_anchor=(1.05, 1), title="Legend")
    legend.get_title().set_fontweight('bold')

    if save_path:
        path = os.path.join(save_path, file_name)
        plt.savefig(f'{path}.svg', format="svg",
                    bbox_inches='tight')  # Use bbox_inches='tight' to ensure the legend is not cut off
        plt.savefig(f'{path}.png', format="png",
                    bbox_inches='tight')  # Use bbox_inches='tight' to ensure the legend is not cut off

    print(category_sums)


if __name__ == '__main__':
    # Generating dummy data
    categories = ['Category1', 'Category2', 'Category3']
    sub_categories = ['SubCategory1', 'SubCategory2', 'SubCategory3']

    data = {}
    for category in categories:
        data[category] = {sub_category: np.random.randint(1, 10) for sub_category in sub_categories}

    data['Category1']['SubCategory1'] = 0

    plot_stacked_bar_chart(data, file_name='test', save_path='/Users/nils/uni/programming/model-search-paper/tmp_dir')
