import utils
import utils_plotly
import utils_population

rule = utils.Rule()
rule.b = 1.85
rule.matrix = [
    [0, rule.b, 0, rule.b],
    [0, 1, 0, 1],
    [0, rule.b, 0, rule.b],
    [0, 1, 0, 1]
]
# rotator con b = 1.67 o mayor

rule.transition = [
    [0, 3, 0, 3],
    [2, 1, 2, 1],
    [0, 3, 0, 3],
    [2, 1, 2, 1],
]

map_rows = 50
map_cols = 50
generations = 85

initial_population = utils_population.single_defector(map_rows, map_cols)

matrix_list = utils.run(initial_population, rule, generations)

for idx, m in enumerate(matrix_list):
    print(f"Plotting: {idx}")
    # time.sleep(1)
    # utils_plot.plot_4s_array(m, step=idx, b=rule.b, file_prefix=filename, format="png", grid_data=True)
    utils_plotly.plot_map(m, step=idx, b=rule.b, file_prefix="kaleido", format="png", grid_data=True,
                          title=f"b: {rule.b} - step: {idx}")
