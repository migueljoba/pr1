import numpy as np
import utils
import utils_plot

from numpy.random import RandomState

rule = utils.Rule()
rule.b = 1.13
rule.matrix = [
    [0, rule.b],
    [0, 1]
]

total_steps = 200
seed = 123456789

rand_np = RandomState(seed)
initial_population = rand_np.choice([0, 1], p=[0.1, 0.9], size=(20, 20))

matrix_list = utils.run(initial_population, rule, total_steps)

plot_data = utils.resume_frequency_data(matrix_list)
[print(f"{idx},{val}") for idx, val in enumerate(plot_data)]

plot_title = f"b: {rule.b}"
plot = utils_plot.plot_frequency(plot_data, title=plot_title)

filename = f"images/freq-{rule.b}.png"
plot.savefig(filename)
plot.close()
