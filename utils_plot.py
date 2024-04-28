import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np


def plot_binary_array(array, step=None, b=None, title: str = None, file_prefix=None, grid_data=False,
                      format: str = 'png'):
    color_defector = 'lightcoral'
    color_cooperator = 'steelblue'
    color_map = colors.ListedColormap([color_defector, color_cooperator])

    fig, ax = plt.subplots(dpi=700)
    ax.imshow(array, cmap=color_map)

    total_rows, total_cols = array.shape
    ax.set_xticks(np.arange(total_cols))
    ax.set_yticks(np.arange(total_rows))

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    if title is not None:
        plt.title(title)
    else:
        plt.title(f"Step: {step} - b: {b}")

    if grid_data:
        for i, j in np.ndindex(array.shape):
            ax.text(j, i, array[i, j],
                    horizontalalignment='center',
                    verticalalignment='center',
                    color='white')

    # TODO directorio de salida debe ser parametrizable
    if file_prefix is not None:
        filename = f"images/{file_prefix}-{b}-{step}.{format}"
    else:
        filename = f"images/{b}-{step}.{format}"

    plt.savefig(filename, format=format)
    plt.close()


def plot_4s_array(array, step=None, b=None, title: str = None, file_prefix=None, grid_data=False, format="png",
                  ticks: bool = False):
    color_defector = 'lightcoral'
    color_new_defector = 'gold'
    color_new_cooperator = 'yellowgreen'
    color_cooperator = 'steelblue'

    color_map = colors.ListedColormap([color_defector, color_new_defector, color_new_cooperator, color_cooperator])

    fig, ax = plt.subplots(dpi=700)
    ax.imshow(array, cmap=color_map)

    total_rows, total_cols = array.shape

    if ticks:
        ax.set_xticks(np.arange(total_cols))
        ax.set_yticks(np.arange(total_rows))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    if title is not None:
        plt.title(title)
    else:
        plt.title(f"Step: {step} - b: {b}")

    if grid_data:
        for i, j in np.ndindex(array.shape):
            ax.text(j, i, array[i, j],
                    horizontalalignment='center',
                    verticalalignment='center',
                    color='white')

    # TODO directorio de salida debe ser parametrizable
    if file_prefix is not None:
        filename = f"images/{file_prefix}-{b}-{step}.{format}"
    else:
        filename = f"images/{b}-{step}.{format}"

    plt.savefig(filename, format=format)
    plt.close()


def plot_frequency(data: list, title: str = None):
    plt.figure(1, dpi=800)

    x_min = 0
    x_max = len(data)
    # x_max = 100
    y_min = 0
    y_max = 1

    plt.axis((x_min, x_max, y_min, y_max))
    plt.plot(data, linewidth=1)

    if title is not None:
        plt.title(title)

    return plt
