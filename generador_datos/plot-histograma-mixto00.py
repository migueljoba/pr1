#!/usr/bin/env python3
import math
import os
import time
from pathlib import Path

import pandas as pd
import plotly.express as px

import utils_file as uf
import utils_plotly as up
import constants as cons

# ============================================================
# Configuración de directorios
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "simulations"  # CSV crudos de simulaciones
OUTPUT_DIR = Path(f"{cons.IMAGES_DIR}/sim/histograms")  # salida de los PDF y SVG

INPUT_PREFIX = "summary-"  # mismo prefijo que usabas antes
OUTPUT_PREFIX = ""  # mismo prefijo que usabas antes

COLOR_FALSE = up.COLOR_DEFECTOR
COLOR_TRUE = up.COLOR_COOPERATOR


# ============================================================
# Utilitarios estadísticos (idénticos a tu script original)
# ============================================================
def mean_weighted(values, counts):
    N = sum(counts)
    return sum(v * c for v, c in zip(values, counts)) / N if N else 0


def variance_weighted(values, counts):
    mu = mean_weighted(values, counts)
    N = sum(counts)
    return sum(c * (v - mu) ** 2 for v, c in zip(values, counts)) / N if N else 0


def std_weighted(values, counts):
    return math.sqrt(variance_weighted(values, counts))


def compute_global_x_max(csv_files):
    """
    Devuelve el máximo info_generations encontrado en todos los CSV.
    Si no hay datos válidos, devuelve 10 como fallback.
    """
    max_vals = []

    for file in csv_files:
        df = pd.read_csv(file)

        if "info_generations" not in df.columns:
            continue

        s = pd.to_numeric(df["info_generations"], errors="coerce").dropna()
        if not s.empty:
            max_vals.append(s.max())

    if max_vals:
        return int(max(max_vals))
    else:
        return 10  # fallback seguro


# ============================================================
# Resumen en memoria (dos series: False y True)
# ============================================================
def generate_two_summaries(df: pd.DataFrame):
    """
    Devuelve:
        - summary_false: DataFrame con columnas ['value','count']
        - summary_true:  DataFrame con columnas ['value','count']

    Ambos usando los valores reales de info_generations.
    """

    if "status_undetermined" not in df.columns:
        raise ValueError("Falta columna 'status_undetermined'")
    if "info_generations" not in df.columns:
        raise ValueError("Falta columna 'info_generations'")

    # Serie A: status_undetermined = False
    df_false = df[df["status_undetermined"] == False].copy()

    # Serie B: status_undetermined = True
    df_true = df[df["status_undetermined"] == True].copy()

    # Convertir info_generations a entero seguro
    df_false["info_generations"] = pd.to_numeric(df_false["info_generations"], errors="coerce").astype("Int64")
    df_true["info_generations"] = pd.to_numeric(df_true["info_generations"], errors="coerce").astype("Int64")

    # Agrupar (value -> count) para cada serie
    summary_false = (
        df_false["info_generations"]
        .value_counts()
        .sort_index()
        .rename_axis("value")
        .reset_index(name="count")
    )

    summary_true = (
        df_true["info_generations"]
        .value_counts()
        .sort_index()
        .rename_axis("value")
        .reset_index(name="count")
    )

    return summary_false, summary_true


# ============================================================
# Render del gráfico (puede tener 1 o 2 series)
# ============================================================
def render_plot(file_path: Path, summary_false: pd.DataFrame, summary_true: pd.DataFrame, x_max: int):
    """
    Crea y guarda un gráfico mixto de hasta dos series.
    """

    # --------------------------------------------------------
    # Crear figura en blanco
    # --------------------------------------------------------
    fig = px.bar()  # se irá completando serie por serie

    # --------------------------------------------------------
    # Serie A: invasor (False)
    # --------------------------------------------------------
    if not summary_false.empty:
        fig_false = px.bar(
            summary_false,
            x="value",
            y="count",
            labels={"value": "Generaciones", "count": "Frecuencia"},
            color_discrete_sequence=[COLOR_FALSE],
        )
        fig.add_traces(fig_false.data)
        # fig.add_traces([t.update(name="Invasor (False)") for t in fig_false.data])

        # Estadísticas de serie False
        values = summary_false["value"].tolist()
        counts = summary_false["count"].tolist()
        mean_val = mean_weighted(values, counts)
        fig.add_vline(
            x=mean_val,
            line_dash="dash",
            line_color="black",
            annotation_text=f"Invasor μ={mean_val:.2f}",
            annotation_bgcolor="rgba(255, 255, 255, 0.8)",
            annotation_position="top right"
        )

    # --------------------------------------------------------
    # Serie B: mixto (True)
    # --------------------------------------------------------
    if not summary_true.empty:
        fig_true = px.bar(
            summary_true,
            x="value",
            y="count",
            labels={"value": "Generaciones", "count": "Frecuencia"},
            color_discrete_sequence=[COLOR_TRUE],
        )

        fig.add_traces(fig_true.data)
        # fig.add_traces([t.update(name="Mixto (True)") for t in fig_true.data])

        # Estadísticas de serie True
        values = summary_true["value"].tolist()
        counts = summary_true["count"].tolist()
        mean_val = mean_weighted(values, counts)
        fig.add_vline(
            x=mean_val,
            line_dash="dot",
            line_color="red",
            annotation_text=f"Mixto μ={mean_val:.2f}",
            annotation_bgcolor="rgba(255, 255, 255, 0.8)",
            annotation_position="bottom right"
        )

    # --------------------------------------------------------
    # Estética final
    # --------------------------------------------------------
    fig.update_xaxes(range=[0, x_max + 1])
    fig.update_yaxes(range=[0, 2000])
    fig.update_yaxes(nticks=5)
    # fig.update_yaxes(tick0=0, dtick=200)

    # Tipografía idéntica a tu script previo
    fig.update_layout(
        barmode="group",
        # barmode="overlay",
        font=dict(
            family="LMRoman10",
            size=32,
            color="black"
        )
    )

    # Título (parseado a partir del filename original)
    title = f"{uf.parse_filename(file_path.name).get('str')}"
    fig.update_layout(title=title)

    # --------------------------------------------------------
    # Guardar PDF y SVG
    # --------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_stem = OUTPUT_PREFIX + file_path.stem

    fig.write_image(str(OUTPUT_DIR / f"{output_stem}.pdf"))
    # fig.write_image(str(OUTPUT_DIR / f"{output_stem}.svg")) TODO mbaez descomentar para guardar SVG

    print(f"Gráfico generado: {output_stem}.pdf")


def render_directory():
    pass


# ============================================================
# Main
# ============================================================
def main():
    # --------------------------------------------------------
    # Preparar el gráfico vacío
    # (include workaround de MathJax como tu script original)
    # --------------------------------------------------------
    print("Generando gráfico basura ...")
    tmp_pdf = "delete_this.pdf"
    fig_tmp = px.scatter(x=[0, 1], y=[0, 1])
    fig_tmp.write_image(tmp_pdf, format="pdf")
    time.sleep(2)
    print("Gráfico basura generado ...")

    subdir_list = ["dim10", "dim20", "dim30", "dim40", "dim50", "dim60", "dim70", "dim80", "dim90", "dim100"]
    subdir_list = ["dim10"]
    for subdir in subdir_list:

        csv_files = uf.get_all_csv_files(INPUT_DIR / subdir, recursive=False)

        """
        csv_files = sorted(
            p for p in INPUT_DIR.iterdir()
            if p.is_file() and p.suffix.lower() == ".csv"
            and not p.name.startswith(INPUT_PREFIX)  # evita procesar "summary-..."
        )
        """

        if not csv_files:
            raise FileNotFoundError(f"No se encontraron CSV en {INPUT_DIR}")

        # Máximo global de info_generations entre todos los CSV
        global_x_max = compute_global_x_max(csv_files)
        print(f"Max X en {subdir}: {global_x_max}")

        for file in csv_files:
            df = pd.read_csv(file)
            summary_false, summary_true = generate_two_summaries(df)
            render_plot(file, summary_false, summary_true, x_max=global_x_max)


if __name__ == "__main__":
    main()
