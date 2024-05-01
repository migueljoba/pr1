import time

import numpy as np

import utils
import utils_file
import utils_plot
from rule import Rule

rule = Rule()
rule.b = 1.9
rule.use_binary_rule()
rule.use_binary_transition()

# rotator con b = 1.67 o mayor
filename = "rotator"
print(f"Opening {filename}")

population_array = utils_file.import_csv(filename)
initial_population = np.array(population_array, dtype=int)

matrix_list = utils.run(initial_population, rule, 10)

for idx, m in enumerate(matrix_list):
    print(f"Plotting: {idx}")
    time.sleep(0.6)
    utils_plot.plot_binary_array(m, step=idx, b=rule.b, file_prefix=filename)
