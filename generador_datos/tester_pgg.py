import numpy as np

import utils
import utils_file
import utils_plotly
import utils_file
import utils_population
from rule import RulePgg

# definir la regla de juego
rule = RulePgg()
rule.use_3s_transition()
rule.pay = 1
rule.factor = 3
rule.tolerance = 0
rule.sides = 51
rule.radio = 1

# numero de generaciones
generations = 80

# initial_population = utils.random_population([0, 1], [0.2, 0.8], (rule.sides, rule.sides))
# initial_population = utils_population.cluster(sides=rule.sides)
# initial_population = utils_population.frente(sides=rule.sides, i=10, j=10)
initial_population = utils_population.cruz(sides=rule.sides, i=10, j=10)
# initial_population = utils_population.ele_original(sides=rule.sides, i=10, j=10, len_i=9, len_j=9)
# initial_population = utils_population.ele()
# initial_population = utils_population.custom()
# initial_population = utils_population.single_defector(sides=rule.sides)

# initial_population = utils_population.diagonal(sides=rule.sides)
# initial_population = utils_population.equis(sides=rule.sides)
# file_data = utils_file.import_csv("isla", directory="../data_source")
# initial_population = np.array(file_data)

matrix_list = utils.run_pgg(initial_population, rule, generations, stop_when_all=10)

# matrices de evolucion
utils_plotly.imshow_animate(np.array(matrix_list['matrix_list']), title=f'Evolución - {rule}')

# matrices de pago
utils_plotly.imshow_animate(np.array(matrix_list['payoff_list']), title=f'Pagos - {rule}')

plot_data = utils.resume_frequency_data(matrix_list['matrix_list'])

plot_title = f"r: {rule.factor}, t:{rule.tolerance} dim: {initial_population.shape}, generations: {generations}"
plot = utils_plotly.plot_frequency(data=plot_data, title=plot_title)
plot.show()