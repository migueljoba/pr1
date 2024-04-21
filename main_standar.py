import time

import utils
import utils_plot
import utils_file
import numpy as np

file_dir = "./data_source/"
filename = "grower"
rule = utils.Rule()
rule.b = 1.9
rule.matrix = [
    [0, rule.b],
    [0, 1]
]
# rotator con b = 1.67 o mayor

filepath = file_dir + filename + ".csv"

print(f"Opening {filepath}")

population_array = utils_file.import_csv(filepath)
initial_population = np.array(population_array, dtype=int)

matrix_list = [initial_population]

for step in range(10):
    print(f"Step {step}")

    current_step = np.zeros(initial_population.shape, dtype=int)
    previous_step = matrix_list[-1]

    payoff_array = utils.generate_weight_array(previous_step, rule)

    for idx_i, row in enumerate(payoff_array):
        for idx_j, col in enumerate(row):
            neighbours_payoff = utils.get_neighbours(payoff_array, idx_i, idx_j)
            winner_idx = utils.get_highest_element_idx(neighbours_payoff)
            neighbours = utils.get_neighbours(previous_step, idx_i, idx_j)
            current_step[idx_i, idx_j] = neighbours[winner_idx[0]][winner_idx[1]]

    matrix_list.append(current_step)

for idx, m in enumerate(matrix_list):
    print(f"Plotting: {idx}")
    time.sleep(1)
    utils_plot.plot_binary_array(m, step=idx, b=rule.b, file_prefix=filename)
