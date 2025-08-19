import utils
import utils_plotly
from rule import Rule

rule = Rule()
rule.b = 1.85
rule.use_4s_rule()
rule.use_4s_transition()

sides = 21
size = (sides, sides)
generations = 3

initial_population = utils.random_population([0, 1], [0.2, 0.8], size=size)

matrix_result = utils.run_pgg(initial_population, rule, generations, verbose=True)
matrix_list = matrix_result.get("matrix_list")

plot_data = utils.resume_frequency_data(matrix_list, strategy=[1])
plot_title = f"b: {rule.b}"
# plot = utils_plotly.plot_frequency(data=plot_data, title=plot_title)
# plot.show()
