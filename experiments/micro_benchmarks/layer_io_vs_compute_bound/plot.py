import matplotlib.pyplot as plt


import matplotlib.pyplot as plt

def plot(data, output_path=None):
    # Extract keys and values from the dictionary
    keys = list(data.keys())
    values = list(data.values())

    # Plot the data as connected dots
    plt.plot(keys, values, marker='o', linestyle='-')
    plt.xlabel('Layers')
    plt.ylabel('number of layers')
    plt.grid(True)

    # Ensure the y-axis starts at 0
    plt.ylim(bottom=0)

    if output_path:
        # Use bbox_inches='tight' to ensure the legend is not cut off
        path = f'{output_path}.svg'
        plt.savefig(path, format="svg", bbox_inches='tight')
        path = f'{output_path}.png'
        plt.savefig(path, format="png", bbox_inches='tight')

    plt.close()



if __name__ == '__main__':
    data = {"2": 2, "3": 3, "4": 4}
    plot(data, './test')
