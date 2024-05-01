import time

import numpy as np
from numpy.random import RandomState

import utils
import utils_plot
from rule import Rule

rule = Rule()
rule.b = 1.8
rule.use_binary_rule()
rule.use_binary_transition()

rand_np = RandomState(1234567)
rand_initial_population = rand_np.randint(2, size=(100, 100), dtype=int)

matrix_list = [rand_initial_population]

for step in range(10):
    print(f"Step {step}")

    current_step = np.zeros(rand_initial_population.shape, dtype=int)
    previous_step = matrix_list[-1]

    payoff_array = utils.generate_weight_array(previous_step, rule)

    for idx_i, idx_j in np.ndindex(rand_initial_population.shape):
        neighbours_payoff = utils.get_neighbours(payoff_array, idx_i, idx_j)
        winner_idx = utils.get_highest_element_idx(neighbours_payoff)
        neighbours = utils.get_neighbours(previous_step, idx_i, idx_j)
        current_step[idx_i, idx_j] = neighbours[winner_idx[0]][winner_idx[1]]

    matrix_list.append(current_step)

for idx, m in enumerate(matrix_list):
    print(f"Plotting: {idx}")
    time.sleep(1)
    utils_plot.plot_binary_array(m, step=idx, b=rule.b, file_prefix="rand")
