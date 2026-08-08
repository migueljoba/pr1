import numpy as np

import utils
import utils_plotly
import utils_population
from rule import Rule

# definir la regla de juego
rule = Rule()
rule.b = 1.85
rule.use_binary_rule()
# rule.use_4s_rule()
rule.use_binary_transition()
# rule.use_4s_transition()
# dimensiones de cada lado para la matriz
sides = 50

# numero de generaciones
generations = 100

# initial_population = utils_population.single_defector(sides=sides)
# initial_population = utils_population.custom()
initial_population = utils.random_population([0, 1], [0.3, 0.7],
                                             (sides, sides), seed=0)


matrix_list = utils.run(initial_population, rule, generations)

utils_plotly.imshow_animate(np.array(matrix_list))

plot_data = utils.resume_frequency_data(matrix_list)

plot_title = f"b: {rule.b}, dim: {initial_population.shape}, generations: {generations}"
# plot = utils_plotly.plot_frequency(data=plot_data, title=plot_title)
# plot.show()