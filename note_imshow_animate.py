import numpy as np

import utils
import utils_plotly
import utils_population
from rule import Rule

rule = Rule()
rule.b = 1.85
rule.use_4s_rule()
rule.use_4s_transition()

run_set = [
    # (sides, generations)
    (25, 19)
]

for sides, generations in run_set:
    initial_population = utils_population.single_defector(sides=sides)
    matrix_list = utils.run(initial_population, rule, generations, verbose=True)
    utils_plotly.imshow_animate(np.array(matrix_list))
