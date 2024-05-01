import utils
import utils_plotly
import utils_population
from rule import Rule

rule = Rule()
rule.b = 1.85
rule.use_4s_rule()
rule.use_4s_transition()

map_rows = 30
map_cols = 30
generations = 10

initial_population = utils_population.single_defector(map_rows, map_cols)

matrix_list = utils.run(initial_population, rule, generations)

for idx, m in enumerate(matrix_list):
    print(f"Plotting: {idx}")
    # time.sleep(1)
    # utils_plot.plot_4s_array(m, step=idx, b=rule.b, file_prefix=filename, format="png", grid_data=True)
    utils_plotly.plot_map(m, step=idx, b=rule.b, file_prefix="kaleido", format="png", grid_data=True,
                          title=f"b: {rule.b} - step: {idx}")
