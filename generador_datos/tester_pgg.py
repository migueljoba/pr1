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
rule.sides = 10
rule.population_prob(c=0.7, d=0.3)
rule.radio = 1
rule.pay = 1
rule.factor = 1.6
rule.tolerance = 0
rule.border = False
rule.stop_when_undetermined = True
# numero de generaciones
generations = 50


ESTOY TRABAJANDO EN DETENER LA SIMULACION CUANDO LA POBLACION YA NO EVOLUCIONA. PARECE FUNCIONAR BIEN. HICE REFACTOR DE NOMBRE
DE VARIABLE Y FIX DE BUG QUE CONFUNDE COLAPSO CON EQUILIBRIO ESTATICO.
TAMBIÉN ADAPTE SCRIPT DE SIMULADOR MASIVO A CSV PARA AGREGAR COLUMNA DE BANDERA DE COLAPSO INDETERMINADO O NO.
DEBO CONTINUAR VERIFICANDO VISUALMENTE ESO.
LA IDEA ES QUE UN SET DE SIMULACIONES SE EJECUTE MAS RAPIDO

initial_population = utils.random_population([0, 1], [rule.prob_defector, rule.prob_cooperator],
                                             (rule.sides, rule.sides), seed=139)
# initial_population = utils_population.cluster(sides=rule.sides, rows=3, cols=3)
# initial_population = utils_population.frente(sides=rule.sides, i=10, j=10)
# initial_population = utils_population.cruz(sides=rule.sides, i=15, j=15)
# initial_population = utils_population.ele_original(sides=rule.sides, i=10, j=10, len_i=9, len_j=9)
# initial_population = utils_population.ele()
# initial_population = utils_population.custom()
# initial_population = utils_population.single_defector(sides=rule.sides)
# initial_population = utils_population.single_cooperator(sides=rule.sides)
# initial_population = utils_population.diagonal(sides=rule.sides)
# initial_population = utils_population.equis(sides=rule.sides)
# file_data = utils_file.import_csv("isla", directory="../data_source")
# initial_population = np.array(file_data)

pgg_result = utils.run_pgg(initial_population, rule, generations, stop_when_all=utils.VARIANTS_DEFECTOR)

# matrices de evolucion
utils_plotly.imshow_animate(np.array(pgg_result['matrix_list']), title=f'Evolución - {rule} nuevo')

# matrices de pago
render_payment_array = False
if render_payment_array:
    utils_plotly.imshow_animate(np.array(pgg_result['payoff_list']), title=f'Pagos - {rule} nuevo')

render_frequency_array = True
if render_frequency_array:
    plot_data = utils.resume_frequency_data(pgg_result['matrix_list'])

    plot_title = f"r: {rule.factor}, t:{rule.tolerance} dim: {initial_population.shape}, generations: {generations}"
    plot = utils_plotly.plot_frequency(data=plot_data, title=plot_title)
    plot.show()

export_evolution = False
if export_evolution:
    # guardar CSV de evolucion
    # Directorio de entrada: por defecto, el mismo donde está este script
    INPUT_DIR = Path(__file__).resolve().parent
    # Directorio de salida (se creará si no existe)
    OUTPUT_DIR = INPUT_DIR / "evolution" / "prueba"

    # Prefijo para los archivos generados (evita re-procesarlos)
    OUTPUT_PREFIX = "summary-"

    fileprefix = "evo-"
    utils_file.export_evolution_csv(pgg_result["matrix_list"], OUTPUT_DIR, fileprefix)
