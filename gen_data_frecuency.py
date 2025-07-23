"""
Script para generar masivamente datos para
graficos de frecuencia por generacion
"""

import utils
import utils_file
from rule import RulePgg

for f in utils.custom_range(1, 10, step=1):
    print(f"Factor: {f}")
    rule = RulePgg()
    rule.pay = 1
    rule.factor = f
    rule.tolerance = 13

    # dimensiones de cada lado para la matriz
    sides = 20

    # numero de generaciones
    generations = 30

    # % generar poblacion inicial
    initial_population = utils.random_population([0, 1], [0.2, 0.8], (sides, sides))

    # % ejecutar juego(poblacion inicial, regla, generaciones)
    matrix_list = utils.run_pgg(initial_population, rule, generations, stop_when_all=0)

    # utils_plotly.imshow_animate(np.array(matrix_list), height=700)

    rule_identifier = f"dim{sides}-p{rule.pay}-r{rule.factor}-tol{rule.tolerance}"
    directory = f"./resultado/{rule_identifier}/"
    # utils_file.export_evolution_csv(matrix_list, directory, fileprefix=rule_prefix)

    plot_freq = utils.resume_frequency_data(matrix_list)
    plot_title = f"pago: {rule.pay}, factor: {rule.factor}, tolerancia: {rule.tolerance}"
    # plot = utils_plotly.plot_frequency(data=plot_freq, title=plot_title)
    # plot.show()

    csv_frecuency_file = f"./resultado/data_frecuency/{rule_identifier}.csv"
    utils_file.export_csv(plot_freq, csv_frecuency_file, header=["indice", "valor"])
