#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import utils_file as uf
import plotly.graph_objects as go

# --- Configuración ---
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "simulations"
OUTPUT_DIR = BASE_DIR / "borrar"
OUTPUT_PREFIX = "summary-"  # por si luego quieres filtrar/evitar reprocesar salidas


def weighted_mean_from_series(values: pd.Series) -> float:
    """Media ponderada a partir de una serie (se pondera por frecuencia de cada valor)."""
    # Compactar por frecuencia
    counts = values.value_counts().sort_index()
    idx = counts.index.to_numpy(dtype=float)
    cnt = counts.to_numpy(dtype=float)
    return float((idx * cnt).sum() / cnt.sum()) if cnt.sum() else float("nan")


def process_csv(csv_path: Path) -> dict:
    """Procesa un CSV y retorna un dict con metadatos (de filename) + métricas."""
    # 1) Metadatos del nombre
    meta = uf.parse_filename(
        csv_path.name)  # {'dim', 'prob-d', 'prob-c', 'radio', 'pay', 'factor', 'tol', 'str', ...}

    # 2) Leer CSV
    df = pd.read_csv(csv_path)

    # 3) Validaciones mínimas de columnas
    required_cols = {"info_generations", "status_undetermined"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name}: faltan columnas requeridas: {sorted(missing)}")

    # 4) Reglas sobre status_undetermined: True -> info_generations = 49
    mask_und = df["status_undetermined"] == True
    if mask_und.any():
        df.loc[mask_und, "info_generations"] = 49

    # 5) Asegurar numérico y limpiar NaN
    s = pd.to_numeric(df["info_generations"], errors="coerce").dropna().astype(int)

    # 6) Métricas
    n_rows = int(len(s))  # cantidad de registros efectivos
    mean_info = weighted_mean_from_series(s)

    colapso = int((df["status_undetermined"] == False).sum())
    equilibrio = int((df["status_undetermined"] == True).sum())

    # 7) Mapear metadatos a cabeceras solicitadas
    out = {
        "sides": int(meta["dim"]),
        "defector": float(meta["prob-d"]),
        "cooperator": float(meta["prob-c"]),
        "radio": int(meta["radio"]),
        "pay": float(meta["pay"]),
        "factor": float(meta["factor"]),
        "tolerance": int(meta["tol"]),
        "n_rows": n_rows,
        "mean_info_generations": round(mean_info, 6),
        # opcionalmente mantener el nombre del archivo:
        # "filename": csv_path.name,
        # "relpath": str(csv_path.relative_to(INPUT_DIR)),
        "colapso": colapso,
        "equilibrio": equilibrio
    }
    return out


def build_summary() -> pd.DataFrame:
    """Itera recursivamente sobre INPUT_DIR y construye la tabla resumen."""
    rows = []
    for csv_path in sorted(p for p in INPUT_DIR.rglob("*.csv") if p.is_file()):
        # Si quieres saltar archivos que ya son “summary-”:
        # if csv_path.name.startswith(OUTPUT_PREFIX): continue
        rows.append(process_csv(csv_path))

    if not rows:
        return pd.DataFrame(columns=[
            "sides", "defector", "cooperator", "radio", "pay", "factor", "tolerance",
            "n_rows", "mean_info_generations"  # , "filename", "relpath"
        ])

    df_summary = pd.DataFrame(rows).sort_values(
        ["sides", "factor", "tolerance"], ignore_index=True
    )
    return df_summary


def show_table_plotly(df, out_path: Path):
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns)),
        cells=dict(values=[df[col] for col in df.columns])
    )])
    fig.write_html(out_path, include_plotlyjs="cdn", auto_open=True)


def show_summary_table(df):
    # --- Colores y estilo ---
    headerColor = 'grey'
    rowEvenColor = 'lightgrey'
    rowOddColor = 'white'

    # Header: títulos de columnas en negrita
    header = dict(
        values=[f"<b>{c}</b>" for c in df.columns],
        line_color='darkslategray',
        fill_color=headerColor,
        align=['left', 'center'],
        font=dict(color='white', size=12)
    )

    # Alternancia de colores por fila
    n_rows = len(df)
    fill_matrix = [[rowOddColor if i % 2 == 0 else rowEvenColor for i in range(n_rows)]] * len(df.columns)

    # Celdas: valores por columna
    cells = dict(
        values=[df[c].tolist() for c in df.columns],
        line_color='darkslategray',
        fill_color=fill_matrix,
        align=['left', 'center'],
        font=dict(color='darkslategray', size=11),
        height=24
    )

    fig = go.Figure(data=[go.Table(header=header, cells=cells)])

    # Opcional: tamaño y márgenes (ajusta a gusto)
    fig.update_layout(
        title="Resumen de simulaciones",
        # width=1500,
        # height=min(800, 120 + 26 * max(10, n_rows)),  # autoalto simple
        margin=dict(l=10, r=10, t=40, b=10)
    )

    # Para tablas muy altas, puedes activar scroll vertical embebido:
    # fig.update_layout(height=800)  # y usar el scroll del navegador/renderer

    fig.show()


# --- Ejemplo de uso ---
# summary_df = build_summary()   # <- de tu script anterior
# show_summary_table(summary_df)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_df = build_summary()
    show_summary_table(summary_df)

    out_csv = OUTPUT_DIR / f"{OUTPUT_PREFIX}overview.csv"
    summary_df.to_csv(out_csv, index=False)
    # print(f"Resumen generado: {out_csv} ({len(summary_df)} filas)")
    # # Muestra unas primeras filas en terminal
    # with pd.option_context('display.max_columns', None, 'display.width', 160):
    #     print(summary_df.head(12))

    # uso:
    # out_file = OUTPUT_DIR / "summary_table_plotly.html"
    # show_table_plotly(summary_df, out_file)
