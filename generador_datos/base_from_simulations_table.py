#!/usr/bin/env python3
import math
import os
import time
from pathlib import Path
import re  # <-- agregado

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go  # <-- agregado

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
    df_false["info_generations"] = pd.to_numeric(
        df_false["info_generations"], errors="coerce"
    ).astype("Int64")
    df_true["info_generations"] = pd.to_numeric(
        df_true["info_generations"], errors="coerce"
    ).astype("Int64")

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
# Media combinando False y True
# ============================================================
def mean_from_summaries(summary_false: pd.DataFrame, summary_true: pd.DataFrame) -> float:
    """
    Combina ambas series (False y True) y calcula la media ponderada
    de info_generations usando las columnas ['value','count'].
    """
    combined = pd.concat([summary_false, summary_true], ignore_index=True)

    if combined.empty:
        return float("nan")

    combined = combined.groupby("value", as_index=False)["count"].sum()
    values = combined["value"].astype(float).tolist()
    counts = combined["count"].astype(int).tolist()

    return mean_weighted(values, counts)


# ============================================================
# Media usando SOLO status_undetermined = False
# ============================================================
def mean_from_summary_false(summary_false: pd.DataFrame) -> float:
    """
    Calcula la media ponderada de info_generations usando solo
    las filas con status_undetermined = False (summary_false).
    """
    if summary_false.empty:
        return float("nan")

    values = summary_false["value"].astype(float).tolist()
    counts = summary_false["count"].astype(int).tolist()

    return mean_weighted(values, counts)


# ============================================================
# Creación de la tabla y render en browser
# ============================================================
def generate_table(subdir: str, rows: list[dict]):
    """
    rows: lista de dicts con claves:
        - 'factor' (float)
        - 'tolerance' (str, por ejemplo 'tol00', 'tol12', ...)
        - 'mean' (float)
    """
    if not rows:
        return

    df = pd.DataFrame(rows)

    table_df = (
        df.pivot(index="factor", columns="tolerance", values="mean")
        .sort_index()
        .sort_index(axis=1)
    )

    print(f"\n=== Tabla de medias para {subdir} ===")
    print(table_df)

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["factor"] + list(table_df.columns),
                ),
                cells=dict(
                    values=[table_df.index.tolist()]
                           + [table_df[col].round(3).tolist() for col in table_df.columns]
                ),
            )
        ]
    )
    fig.update_layout(title=f"Medias ponderadas de info_generations – {subdir}")
    fig.show()


def main():
    subdir_list = [
        "dim10",
        "dim20",
        "dim30",
        "dim40",
        "dim50",
        "dim60",
        "dim70",
        "dim80",
        "dim90",
        "dim100",
    ]
    # subdir_list = ["dim20"]
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
            raise FileNotFoundError(f"No se encontraron CSV en {INPUT_DIR / subdir}")

        rows = []

        for file in csv_files:
            df = pd.read_csv(file)
            summary_false, summary_true = generate_two_summaries(df)

            mean_val = mean_from_summary_false(summary_false)

            # factor desde la columna (constante en cada archivo)
            factor_value = float(df["factor"].iloc[0])

            # tolerance desde el nombre del archivo (tol00, tol12, tol32, ...)
            m = re.search(r"tol(\d+)", file.stem)
            if not m:
                raise ValueError(f"No se encontró patrón 'tolNN' en el nombre: {file.name}")
            tol_code = m.group(1)
            tol_label = f"tol{tol_code}"

            rows.append(
                {
                    "factor": factor_value,
                    "tolerance": tol_label,
                    "mean": mean_val,
                }
            )

        generate_table(subdir, rows)


if __name__ == "__main__":
    main()
