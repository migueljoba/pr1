import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.random import RandomState

import utils
from rule import Rule

# Available color maps
# https://matplotlib.org/stable/users/explain/colors/colormaps.html

rule = Rule()
rule.b = 1.3
rule.use_binary_rule()

rand_np = RandomState(123456789)

size = (20, 20)
print_map_data = True
initial_population = rand_np.choice([0, 1], p=[0.1, 0.9], size=size)  # rand_np.randint(2, size=size, dtype=int)

weight_array = utils.generate_weight_array(initial_population, rule)

fig, ax = plt.subplots(dpi=700)
plt.title(f"b: {rule.b}")
# create an Axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

img = ax.imshow(weight_array, cmap='inferno')
plt.colorbar(img, cax=cax)

if print_map_data:
    for i, j in np.ndindex(weight_array.shape):
        ax.text(j, i, weight_array[i, j],
                fontsize=5,
                horizontalalignment='center',
                verticalalignment='center',
                color='white')

plt.savefig(f"images/payoff-{rule.b}.svg", format="svg")

plt.show()
