"""
Script para repetir masivamente simulaciones con mismas reglas,
pero con poblaciones iniciales distintas
"""

import os

import numpy as np

import utils
from rule import RulePgg


def run(rule: RulePgg):
    # numero maximo de generaciones
    generations = 50

    csv_info_list = []

    for n in range(0, 2000):
        rule.info_seed = n

        initial_population = utils.random_population(
            [0, 1], [rule.prob_defector, rule.prob_cooperator],
            (rule.sides, rule.sides),
            seed=n)

        # file_data = utils_file.import_csv("isla", directory="../data_source")
        # initial_population = np.array(file_data)

        matrix_list = utils.run_pgg(initial_population, rule, generations, stop_when_all=utils.VARIANTS_DEFECTOR)

        # dibujar matrices de evolucion
        # utils_plotly.imshow_animate(np.array(matrix_list['matrix_list']), title=f'Evolución - {rule}')

        # dibujar matrices de pago
        # utils_plotly.imshow_animate(np.array(matrix_list['payoff_list']), title=f'Pagos - {rule}')

        # dibujar gráfico de frecuencias
        # frecuency_data = utils.resume_frequency_data(matrix_list['matrix_list'])
        # plot_title = f"r: {rule.factor}, t:{rule.tolerance} dim: {initial_population.shape}, generations: {generations}"
        # plot = utils_plotly.plot_frequency(data=plot_data, title=plot_title)
        # plot.show()

        rule.info_generations = len(matrix_list['matrix_list']) - 1
        print(rule.csv_row())
        csv_info_list.append(rule.csv_row())

    # Verificar si el archivo existe y tiene contenido
    file_name = f"simulations/{rule.params_str()}.csv"
    is_new = not os.path.exists(file_name) or os.stat(file_name).st_size == 0

    with open(file_name, 'a') as f:
        if is_new:
            f.write(",".join(rule.csv_headers()) + "\n")
        np.savetxt(f, csv_info_list, delimiter=",", fmt="%s")


if __name__ == '__main__':
    factors_all = [1, 1.5]
    dimensiones = [20]
    factores = factors_all
    tolerancias = [0]

    # definir la regla de juego

    for dim in dimensiones:
        for fact in factores:
            for tol in tolerancias:
                rule = RulePgg()
                # rule.use_3s_transition()
                rule.sides = dim
                rule.population_prob(c=0.8, d=0.2)
                rule.radio = 1
                rule.pay = 1
                rule.factor = fact
                rule.tolerance = tol

                run(rule)
