import time

import plotly.express as px
import numpy as np

COLOR_DEFECTOR = "#ff7f50"
COLOR_COOPERATOR = "#4682b4"
COLOR_NEW_DEFECTOR = "#ffd700"
COLOR_NEW_COOPERATOR = "#9acd32"

colors_scale_4s = [
    (0, COLOR_DEFECTOR), (0.25, COLOR_DEFECTOR),
    (0.25, COLOR_COOPERATOR), (0.5, COLOR_COOPERATOR),
    (0.5, COLOR_NEW_DEFECTOR), (0.75, COLOR_NEW_DEFECTOR),
    (0.75, COLOR_NEW_COOPERATOR), (1, COLOR_NEW_COOPERATOR)
]


def plot_frequency(data: list, title: str = None):
    return px.line(y=data, range_y=[0, 1], title=title)


def plot_map(array, step=None, b=None, title: str = None, file_prefix=None, grid_data=False, format="png"):
    fig = px.imshow(array, text_auto=grid_data, color_continuous_scale=colors_scale_4s, range_color=[0, 3], title=title)

    # esconder barra de colores
    fig.update_layout(coloraxis_showscale=False)

    # TODO directorio de salida debe ser parametrizable
    if file_prefix is not None:
        filename = f"images/{file_prefix}-{b}-{step}.{format}"
    else:
        filename = f"images/{b}-{step}.{format}"

    dpi = 300
    fig.write_image(filename, width=5 * dpi, height=2.5 * dpi)


def imshow_animate(evolution_list):
    fig = px.imshow(
        evolution_list,
        text_auto=True,
        color_continuous_scale=colors_scale_4s, range_color=[0, 3],
        animation_frame=0
    )

    # esconder barra de colores
    fig.update_layout(coloraxis_showscale=False)

    time.sleep(0.01)
    fig.show()
