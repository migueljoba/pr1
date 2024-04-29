import time

import numpy as np

import utils
import utils_file
import utils_plot

rule = utils.Rule()
rule.b = 1.9
rule.matrix = [
    [0, rule.b],
    [0, 1]
]
# rotator con b = 1.67 o mayor
filename = "rotator"
print(f"Opening {filename}")

population_array = utils_file.import_csv(filename)
initial_population = np.array(population_array, dtype=int)

matrix_list = [initial_population]

matrix_list = utils.run(initial_population, rule, 10)

for idx, m in enumerate(matrix_list):
    print(f"Plotting: {idx}")
    time.sleep(0.6)
    utils_plot.plot_binary_array(m, step=idx, b=rule.b, file_prefix=filename)
