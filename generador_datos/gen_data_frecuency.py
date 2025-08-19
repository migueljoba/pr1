"""
PGG: Script para generar masivamente datos para
graficos de:
    - frecuencia por generacion
    - matrices de evolución
    - matrices de pago
"""

import utils
import utils_file
from rule import RulePgg


def simulate(rule: RulePgg, evolution: bool = True, frequency: bool = True, payment: bool = True):
    print(f"Factor: {rule.factor}, Tolerancia: {rule.tolerance}")

    # % generar poblacion inicial
    initial_population = utils.random_population([0, 1], [0.2, 0.8], (rule.sides, rule.sides))

    # % ejecutar juego(poblacion inicial, regla, generaciones)
    matrix_result = utils.run_pgg(initial_population, rule, rule.generations, stop_when_all=0)
    matrix_list = matrix_result.get("matrix_list")
    payoff_list = matrix_result.get("payoff_list")

    # utils_plotly.imshow_animate(np.array(matrix_list), height=700)

    rule_identifier = f"dim{rule.sides}-p{rule.pay}-r{rule.factor}-tol{rule.tolerance}"

    directory = f"./generador_datos/{rule_identifier}"  # ruta relativa al utilitario

    # Exportar mapa de evolución
    if evolution:
        utils_file.export_evolution_csv(matrix_list, directory, fileprefix=f"evo")

    # Exportar mapa de pagos
    if payment:
        utils_file.export_evolution_csv(payoff_list, directory, fileprefix=f"pago")

    plot_freq = utils.resume_frequency_data(matrix_list)
    plot_title = f"pago: {rule.pay}, factor: {rule.factor}, tolerancia: {rule.tolerance}"
    # plot = utils_plotly.plot_frequency(data=plot_freq, title=plot_title)
    # plot.show()

    csv_frecuency_file = f"{directory}/freq-{rule_identifier}.csv"

    if frequency:
        utils_file.export_csv(plot_freq, csv_frecuency_file, header=["indice", "valor"])


for factor in utils.custom_range(1, 2, step=1):
    for tol in utils.custom_range(0, 2, step=1):
        print(f"Factor: {factor}, Tolerancia: {tol}")
        rule = RulePgg()
        rule.pay = 1
        rule.factor = factor
        rule.tolerance = tol

        rule.sides = 25
        rule.generations = 5

        simulate(rule,
                 evolution=True,
                 frequency=True,
                 payment=True)
