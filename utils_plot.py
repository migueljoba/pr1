import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np


def plot_binary_array(array, step=None, b=None, file_prefix=None):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # Major ticks every 20, minor ticks every 5
    major_ticks = np.arange(0.5, 101, 5)
    minor_ticks = np.arange(0.5, 101, 1)

    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)

    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)

    ax.set_yticklabels([])
    ax.set_xticklabels([])

    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=1, color='black')
    ax.grid(which='major', alpha=1)

    cmap = matplotlib.colors.ListedColormap(['red', 'blue'])
    plt.imshow(array, cmap=cmap)
    plt.grid(True, color='#000', linestyle='-', linewidth=1)

    plt.title(f"Step: {step} - b: {b}")
    # plt.show()

    # TODO directorio de salida debe ser parametrizable
    if file_prefix is not None:
        filename = f"images/{file_prefix}-{b}-{step}.png"
    else:
        filename = f"images/{b}-{step}.png"
    plt.savefig(filename)
