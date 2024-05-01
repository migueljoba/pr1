import time

import utils
import utils_plot
from rule import Rule

rule = Rule()
rule.b = 1.8
rule.use_binary_rule()
rule.use_binary_transition()

rand_np = utils.default_random()
initial_population = rand_np.randint(2, size=(100, 100), dtype=int)
generations = 20

matrix_list = utils.run(initial_population, rule, generations)

for idx, m in enumerate(matrix_list):
    print(f"Plotting: {idx}")
    time.sleep(1)
    utils_plot.plot_binary_array(m, step=idx, b=rule.b, file_prefix="rand")
