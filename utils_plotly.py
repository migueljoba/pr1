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
        "y": "cooperadores / total_poblacion",
    }

    # Usamos índices enteros explícitos para el eje X
    x_values = list(range(len(data)))

    fig = px.line(
        x=x_values,
        y=data,
        range_y=[0, 1],
        title=title,
        labels=labels,
        template="plotly_white",
    )

    # Forzar ticks enteros en el eje X
    fig.update_xaxes(
        tickmode="linear",  # escala lineal
        tick0=2,  # primer tick en 0
        dtick=1  # separación de 1 en 1
    )

    return fig


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
        encoding: str = "utf-8",
        out_summary_csv: str | Path | None = None,
        assume_integers: bool = True
):
    """
    Crea un histograma de frecuencias para la columna especificada de un CSV.
    Además, puede generar un CSV resumido con columnas: value,count.

    Args:
        path_csv: Ruta al archivo CSV.
        column_name: Nombre de la columna a graficar (obligatorio).
        save_html: Si se indica, guarda el gráfico como HTML interactivo.
        separator: Separador del CSV (default: ',').
        encoding: Encoding del archivo (default: 'utf-8').
        out_summary_csv: Ruta para guardar el CSV resumido (opcional).
        assume_integers: Convierte a int antes de contar (default: True).
    """
    path = Path(path_csv)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    df = pd.read_csv(path, sep=separator, encoding=encoding)

    if column_name not in df.columns:
        raise ValueError(
            f"La columna '{column_name}' no existe en el CSV. Columnas: {list(df.columns)}"
        )

    s = pd.to_numeric(df[column_name], errors="coerce").dropna()
    if s.empty:
        raise ValueError(
            f"La columna '{column_name}' no contiene valores numéricos válidos."
        )

    # resumen value,count
    x = s.to_numpy()
    if assume_integers:
        x = x.astype(int)
    vc = pd.Series(x).value_counts().sort_index()
    summary_df = pd.DataFrame({"value": vc.index, "count": vc.values})

    if out_summary_csv:
        summary_df.to_csv(out_summary_csv, index=False)

    # gráfico desde el resumen
    fig = px.bar(
        summary_df,
        x="value",
        y="count",
        text="count",
        title=f"{path_csv}: {column_name} (discreto)",
        labels={"value": column_name, "count": "Ocurrencias"},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(bargap=0.1, template="plotly_white")

    if save_html:
        out = Path(save_html)
        if out.suffix.lower() != ".html":
            out = out.with_suffix(".html")
        fig.write_html(out, include_plotlyjs="cdn")

    fig.show()


def export_frequency_svg(
        data: list,
        output_dir: str,
        filename: str,
        title: str = None,
        trace_color: str = "#1f77b4",  # color de línea / barras
        background_color: str = "white",  # fondo del gráfico
        font_family: str = "LMRoman10",  # fuente
        font_size: int = 14
) -> Path:
    """
    Genera el gráfico de frecuencia y lo exporta como SVG.

    :param data: Datos para el eje Y.
    :param output_dir: Directorio donde se guardará el SVG.
    :param filename: Nombre de archivo (con o sin extensión .svg).
    :param title: Título del gráfico (opcional).
    :param trace_color: Color de la línea o barras.
    :param background_color: Color de fondo (paper y plot).
    :param font_family: Familia tipográfica a usar.
    :param font_size: Tamaño de fuente en puntos.
    :return: Ruta completa del archivo SVG generado.
    """

    # Crear la figura base
    fig = plot_frequency(data, title=title)

    # Estilo de la serie (línea o barras, según el tipo de fig)
    fig.update_traces(
        line=dict(color=trace_color),  # para line plots
        marker=dict(color=trace_color)  # por si es scatter/bar con marker
    )

    # Estilo global (fondo, fuente)
    fig.update_layout(
        paper_bgcolor=background_color,
        plot_bgcolor=background_color,
        font=dict(
            family=font_family,
            size=font_size
        )
    )

    # Normalizar directorio de salida
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Asegurar extensión .svg
    if not filename.lower().endswith(".svg"):
        filename = f"{filename}.svg"

    out_path = out_dir / filename

    # Exportar a SVG (requiere tener instalado "kaleido")
    fig.write_image(str(out_path), format="svg")

    return out_path
