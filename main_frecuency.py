import numpy as np
import utils
import utils_plot

from numpy.random import RandomState


def resume_frecuency_data(collection: list, strategy: str = "c"):
    frecuency_data = []

    for col in collection:
        values, counter = np.unique(col, return_counts=True)
        group = counter[1] / 400 if strategy == "c" else counter[0] / 400
        frecuency_data.append(group)
    return frecuency_data


rule = utils.Rule()
rule.b = 1.13
rule.matrix = [
    [0, rule.b],
    [0, 1]
]

rand_np = RandomState(123456789)
initial_population = rand_np.choice([0, 1], p=[0.1, 0.9], size=(20, 20))

matrix_list = [initial_population]

for step in range(200):
    current_step = np.zeros(initial_population.shape, dtype=np.int8)
    previous_step = matrix_list[-1]
    payoff_array = utils.generate_weight_array(previous_step, rule)

    for idx_i, idx_j in np.ndindex(initial_population.shape):
        neighbours_payoff = utils.get_neighbours(payoff_array, idx_i, idx_j)
        winner_idx = utils.get_highest_element_idx(neighbours_payoff)
        neighbours = utils.get_neighbours(previous_step, idx_i, idx_j)
        current_step[idx_i, idx_j] = neighbours[winner_idx[0]][winner_idx[1]]

    matrix_list.append(current_step)

plot_data = resume_frecuency_data(matrix_list)
[print(f"{idx},{val}") for idx, val in enumerate(plot_data)]

plot_title = f"b: {rule.b}"
plot = utils_plot.plot_frecuency(plot_data, title=plot_title)

filename = f"images/freq-{rule.b}.png"
plot.savefig(filename)
plot.close()
