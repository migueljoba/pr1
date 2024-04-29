import time

import numpy as np

import utils
import utils_file
import utils_plot

file_dir = "./data_source/"
filename = "kaleido"
rule = utils.Rule()
rule.b = 1.85
rule.matrix = [
    [0, rule.b, 0, rule.b],
    [0, 1, 0, 1],
    [0, rule.b, 0, rule.b],
    [0, 1, 0, 1]
]
# rotator con b = 1.67 o mayor

rule.transition = [
    [0, 3, 0, 3],
    [2, 1, 2, 1],
    [0, 3, 0, 3],
    [2, 1, 2, 1],
]

filepath = file_dir + filename + ".csv"

print(f"Opening {filepath}")

population_array = utils_file.import_csv(filepath)
initial_population = np.array(population_array, dtype=int)

matrix_list = [initial_population]

for step in range(150):
    print(f"Step {step}")

    current_step = np.empty(initial_population.shape, dtype=int)
    previous_step = matrix_list[-1]

    payoff_array = utils.generate_weight_array(previous_step, rule)

    for idx_i, idx_j in np.ndindex(payoff_array.shape):
        neighbours_payoff = utils.get_neighbours(payoff_array, idx_i, idx_j)
        winner_idx = utils.get_highest_element_idx(neighbours_payoff)
        neighbours = utils.get_neighbours(previous_step, idx_i, idx_j)

        invader = neighbours[winner_idx[0]][winner_idx[1]]
        current = previous_step[idx_i][idx_j]

        result = rule.transition[current][invader]

        current_step[idx_i, idx_j] = result

    matrix_list.append(current_step)

for idx, m in enumerate(matrix_list):
    print(f"Plotting: {idx}")
    time.sleep(1)
    utils_plot.plot_4s_array(m, step=idx, b=rule.b, file_prefix=filename, format="png", grid_data=True)
