import os

import utils_plotly
from rule import RulePgg


def plot_data(rule: RulePgg, filename: str = None, out_summary: str = None):
    if filename is None:
        filename = f"{rule.params_str()}.csv"

    utils_plotly.plot_histogram_from_csv(filename, "info_generations",
                                         out_summary_csv=out_summary)


if __name__ == "__main__":
    # definir la regla de juego
    rule = RulePgg()
    # rule.use_3s_transition()
    rule.sides = 10
    rule.population_prob(c=0.8, d=0.2)
    rule.radio = 1
    rule.pay = 1
    rule.factor = 5
    rule.tolerance = 0

    # rutas para archivos CSV de entrada y salida
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_source_dir = os.path.join(script_dir, "simulations")
    data_target_dir = os.path.join(script_dir, "histograma")

    filename_in = f"{rule.params_str()}.csv"
    filename_out = f"summary-{rule.params_str()}.csv"

    file_in = os.path.join(data_source_dir, filename_in)
    file_out = os.path.join(data_target_dir, filename_out)

    plot_data(rule, filename=file_in, out_summary=file_out)
