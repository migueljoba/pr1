import time
from pathlib import Path

import pandas as pd
import plotly.express as px

COLOR_DEFECTOR = "#ff7f50"  # rojo
COLOR_COOPERATOR = "#4682b4"  # azul
COLOR_NEW_DEFECTOR = "#ffd700"  # amarillo
COLOR_NEW_COOPERATOR = "#9acd32"  # verder

colors_scale_4s = [
    (0, COLOR_DEFECTOR), (0.25, COLOR_DEFECTOR),
    (0.25, COLOR_COOPERATOR), (0.5, COLOR_COOPERATOR),
    (0.5, COLOR_NEW_DEFECTOR), (0.75, COLOR_NEW_DEFECTOR),
    (0.75, COLOR_NEW_COOPERATOR), (1, COLOR_NEW_COOPERATOR)
]


def plot_frequency(data: list, title: str = None):
    labels = {
        "x": "generaciones",
        "y": "cooperadores / total_poblacion"
    }
    return px.line(y=data, range_y=[0, 1], title=title, labels=labels, template="plotly_white")


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


def imshow_animate(evolution_list, **kwargs):
    fig = px.imshow(
        evolution_list,
        text_auto=True,
        color_continuous_scale=colors_scale_4s, range_color=[0, 3],
        animation_frame=0,
        template="plotly_white",
        **kwargs
    )

    # esconder barra de colores
    fig.update_layout(coloraxis_showscale=False)

    time.sleep(0.01)
    fig.show()


def plot_histogram_from_csv(
        path_csv: str | Path,
        column_name: str,
        save_html: str | Path | None = None,
        separator: str = ",",
        encoding: str = "utf-8"
):
    """
    Crea un histograma de frecuencias para la columna especificada de un CSV.

    Args:
        path_csv: Ruta al archivo CSV.
        column_name: Nombre de la columna a graficar (obligatorio).
        save_html: Si se indica, guarda el gráfico como HTML interactivo.
        separator: Separador del CSV (default: ',').
        encoding: Encoding del archivo (default: 'utf-8').
    """
    path = Path(path_csv)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    df = pd.read_csv(path, sep=separator, encoding=encoding)

    if column_name not in df.columns:
        raise ValueError(f"La columna '{column_name}' no existe en el CSV. Columnas: {list(df.columns)}")

    # Convertir a numérico
    x = pd.to_numeric(df[column_name], errors="coerce").dropna()
    if x.empty:
        raise ValueError(f"La columna '{column_name}' no contiene valores numéricos válidos.")

    fig = px.histogram(
        df,
        x=column_name,
        text_auto=True,
        title=f"{path_csv}: {column_name}",
        labels={column_name: column_name, "count": "Ocurrencias"}
    )

    fig.update_layout(
        bargap=0.1,
        template="plotly_white"
    )

    if save_html:
        out = Path(save_html)
        if out.suffix.lower() != ".html":
            out = out.with_suffix(".html")
        fig.write_html(out, include_plotlyjs="cdn")

    fig.show()
