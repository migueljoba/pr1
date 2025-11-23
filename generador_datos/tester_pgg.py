import numpy as np
from pathlib import Path
import utils
import utils_file
import utils_plotly
import utils_population
from rule import RulePgg

# definir la regla de juego
rule = RulePgg()
# rule.use_3s_transition()
rule.sides = 20
rule.population_prob(c=0.8, d=0.2)
rule.radio = 1
rule.pay = 1
rule.factor = 1
rule.tolerance = 0
rule.border = False
rule.stop_when_undetermined = True
# numero de generaciones
generations = 50
seed = 0

# banderas de datos o graficos
# exportar matrices de evolucion en formato CSV
export_evolution = False

# matrices de evolucion
render_evolution_array = True

# matrices de pago
render_payment_array = False

# grafico de frecuencia
render_frequency_array = False

initial_population = utils.random_population([0, 1], [rule.prob_defector, rule.prob_cooperator],
                                             (rule.sides, rule.sides), seed=seed)
# initial_population = utils_population.cluster(sides=rule.sides, rows=20, cols=3)
# initial_population = utils_population.frente(sides=rule.sides, i=10, j=10)
# initial_population = utils_population.cruz()
# initial_population = utils_population.ele_original(sides=rule.sides, i=10, j=10, len_i=9, len_j=9)
# initial_population = utils_population.ele()
# initial_population = utils_population.custom()
# initial_population = utils_population.single_defector(sides=rule.sides)
# initial_population = utils_population.single_cooperator(sides=rule.sides)
# initial_population = utils_population.diagonal(sides=rule.sides)
# initial_population = utils_population.equis(sides=rule.sides)
# file_data = utils_file.import_csv("isla", directory="../data_source")
# initial_population = np.array(file_data)


if render_evolution_array:
    pgg_result = utils.run_pgg(initial_population, rule, generations, stop_when_all=utils.VARIANTS_DEFECTOR)

# matrices de evolucion
utils_plotly.imshow_animate(np.array(pgg_result['matrix_list']), title=f'Evolución - {rule} nuevo')

if render_payment_array:
    utils_plotly.imshow_animate(np.array(pgg_result['payoff_list']), title=f'Pagos - {rule} nuevo')

if render_frequency_array:
    plot_data = utils.resume_frequency_data(pgg_result['matrix_list'])

    plot_title = f"r: {rule.factor}, t:{rule.tolerance} dim: {initial_population.shape}, generations: {generations}"
    plot = utils_plotly.plot_frequency(data=plot_data, title=plot_title)
    plot.show()

if export_evolution:
    # guardar CSV de evolucion
    # Directorio de entrada: por defecto, el mismo donde está este script
    INPUT_DIR = Path(__file__).resolve().parent
    # Directorio de salida (se creará si no existe)

    dir_name = f'{rule.sides}-{seed}-{rule.factor}-{rule.tolerance}'
    OUTPUT_DIR = INPUT_DIR / "evolution" / dir_name
    print(f"Exporting evolution results to {OUTPUT_DIR}")
    # Prefijo para los archivos generados (evita re-procesarlos)
    OUTPUT_PREFIX = "summary-"

    # fileprefix = "evo-"
    utils_file.export_evolution_csv(pgg_result["matrix_list"], OUTPUT_DIR)
