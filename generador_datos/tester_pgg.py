import numpy as np

import utils
import utils_file
import utils_plotly
import utils_file
import utils_population
from rule import RulePgg

# definir la regla de juego
rule = RulePgg()
rule.pay = 1
rule.factor = 3.5
rule.tolerance = 0

# seed = 40

# dimensiones de cada lado para la matriz
sides = 45

# numero de generaciones
generations = 100

initial_population = utils.random_population([0, 1], [0.2, 0.8], (sides, sides))
# initial_population = utils_population.isla()
# file_data = utils_file.import_csv("isla", directory="../data_source")
# initial_population = np.array(file_data)

matrix_list = utils.run_pgg(initial_population, rule, generations, stop_when_all=0)

utils_plotly.imshow_animate(np.array(matrix_list['matrix_list']))

plot_data = utils.resume_frequency_data(matrix_list['matrix_list'])

plot_title = f"r: {rule.factor}, t:{rule.tolerance} dim: {initial_population.shape}, generations: {generations}"
plot = utils_plotly.plot_frequency(data=plot_data, title=plot_title)
plot.show()