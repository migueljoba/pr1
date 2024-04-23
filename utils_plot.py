import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np


def plot_binary_array(array, step=None, b=None, file_prefix=None, grid_data=False):
    color_defector = 'lightcoral'
    color_cooperator = 'steelblue'
    color_map = colors.ListedColormap([color_defector, color_cooperator])

    fig, ax = plt.subplots(dpi=700)
    ax.imshow(array, cmap=color_map)

    total_rows, total_cols = array.shape
    ax.set_xticks(np.arange(total_cols))
    ax.set_yticks(np.arange(total_rows))

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    plt.title(f"Step: {step} - b: {b}")

    if grid_data:
        for i, j in np.ndindex(array.shape):
            ax.text(j, i, array[i, j],
                    horizontalalignment='center',
                    verticalalignment='center',
                    color='white')

    # TODO directorio de salida debe ser parametrizable
    if file_prefix is not None:
        filename = f"images/{file_prefix}-{b}-{step}.png"
    else:
        filename = f"images/{b}-{step}.png"

    plt.savefig(filename)
    plt.close()


def plot_frecuency(collection: list, strategy: str = 'c', title: str = None):
    data = []

    plt.figure(1, dpi=800)

    for col in collection:
        values, counter = np.unique(col, return_counts=True)
        if strategy == 'c':
            group = counter[1] / 400  # TODO total de indiv. debe ser dinamico
        else:
            group = counter[1] / 400  # TODO total de indiv. debe ser dinamico

        data.append(group)
        print(group)

    x_min = 0
    x_max = len(collection)
    # x_max = 100
    y_min = 0
    y_max = 1

    plt.axis((x_min, x_max, y_min, y_max))
    plt.plot(data, linewidth=1)

    if title is not None:
        plt.title(title)

    return plt
