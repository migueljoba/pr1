import utils
import utils_plotly
from rule import Rule

rule = Rule()
rule.b = 1.26
rule.use_binary_rule()
rule.use_binary_transition()

total_steps = 200
population_elements = [0, 1]
individual_probability = [0.1, 0.9]
size = (20, 20)

initial_population = utils.random_population(
    elements=population_elements,
    probability=individual_probability,
    size=size
)

matrix_list = utils.run(initial_population, rule, total_steps)

plot_data = utils.resume_frequency_data(matrix_list)

plot_title = f"b: {rule.b}"
plot = utils_plotly.plot_frequency(data=plot_data, title=plot_title)
plot.show()
