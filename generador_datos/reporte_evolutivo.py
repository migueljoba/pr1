# csv_heatmaps_to_pdf.py
# Lee todos los CSV de un directorio, crea "heatmaps" (0/1) con plotly.express.imshow
# y los guarda en un único PDF paginado, con una grilla de rows x cols por página.
# Requiere: pip install plotly kaleido PyPDF2

from __future__ import annotations
import os
import csv
import math
from typing import List
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
from PyPDF2 import PdfMerger
import tempfile


def _read_binary_csv(path: str, delimiter: str = ",") -> np.ndarray:
    """Lee CSV a matriz 2D de 0/1 (int). Acepta '0', '1' o numéricos; binariza por umbral 0.5."""
    rows: List[List[int]] = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        for line in reader:
            if not line:
                continue
            vals = []
            for v in line:
                v = v.strip()
                if v == "":
                    continue
                try:
                    x = float(v)
                except ValueError:
                    raise ValueError(f"Valor no numérico en {path}: {v!r}")
                vals.append(1 if x > 0.5 else 0)
            if vals:
                rows.append(vals)
    if not rows:
        raise ValueError(f"CSV vacío: {path}")
    # Verifica rectangularidad
    w = len(rows[0])
    if any(len(r) != w for r in rows):
        raise ValueError(f"CSV con filas de longitudes distintas: {path}")
    return np.asarray(rows, dtype=int)


def _list_csvs(input_dir: str) -> List[str]:
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
             if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(".csv")]
    files.sort(key=lambda s: os.path.basename(s).lower())
    if not files:
        raise FileNotFoundError(f"Sin CSV en: {input_dir}")
    return files


def _make_page_figure(mats: List[np.ndarray],
                      titles: List[str],
                      rows: int,
                      cols: int,
                      cell_px: int) -> "plotly.graph_objects.Figure":
    fig = make_subplots(
        rows=rows,
        cols=cols,
        horizontal_spacing=0.03,
        vertical_spacing=0.08,
        specs=[[{"type": "heatmap"} for _ in range(cols)] for _ in range(rows)],
        subplot_titles=titles + [""] * (rows * cols - len(titles)),
    )
    for k, mat in enumerate(mats):
        r = k // cols + 1
        c = k % cols + 1
        subfig = px.imshow(
            mat,
            color_continuous_scale=[(0.0, "#f0f0f0"), (1.0, "#1f77b4")],
            origin="upper",
            zmin=0, zmax=1,
            aspect="equal",
        )
        trace = subfig.data[0]
        trace.update(showscale=False, coloraxis=None, zmin=0, zmax=1)
        fig.add_trace(trace, row=r, col=c)
        fig.update_xaxes(visible=False, row=r, col=c)
        fig.update_yaxes(visible=False, row=r, col=c)

    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        width=cols * cell_px,
        height=rows * cell_px,
        # Texto de títulos más pequeño para muchas páginas
        font=dict(size=12),
    )
    return fig


def save_csv_heatmaps_to_pdf(input_dir: str,
                             output_pdf: str = "heatmaps.pdf",
                             rows_per_page: int = 2,
                             cols_per_row: int = 4,
                             cell_px: int = 240,
                             delimiter: str = ",") -> None:
    """
    Genera un único PDF con N páginas (según cantidad de CSV) y grilla rows x cols por página.
    - input_dir: carpeta con .csv
    - output_pdf: ruta del PDF de salida
    - rows_per_page, cols_per_row: grilla por página
    - cell_px: tamaño (en px) de cada subgráfico (afecta tamaño del PDF)
    - delimiter: separador del CSV (por defecto ',')
    """
    csv_files = _list_csvs(input_dir)
    per_page = rows_per_page * cols_per_row
    pages = math.ceil(len(csv_files) / per_page)

    tmp_pdf_paths: List[str] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for p in range(pages):
            start = p * per_page
            chunk = csv_files[start:start + per_page]
            mats = [_read_binary_csv(fp, delimiter=delimiter) for fp in chunk]
            titles = [os.path.basename(fp) for fp in chunk]
            fig = _make_page_figure(mats, titles, rows_per_page, cols_per_row, cell_px)
            page_pdf = os.path.join(tmpdir, f"page_{p + 1}.pdf")
            fig.write_image(page_pdf, format="pdf")  # requiere kaleido
            tmp_pdf_paths.append(page_pdf)

        # Unir todas las páginas en un único PDF
        merger = PdfMerger()
        for path in tmp_pdf_paths:
            merger.append(path)
        with open(output_pdf, "wb") as fout:
            merger.write(fout)
        merger.close()


if __name__ == "__main__":
    # Ejemplo de uso:
    # - Directorio con CSVs (cada CSV -> un heatmap binario)
    # - Salida: PDF con 2 filas x 4 columnas por página
    data_directory = "dim25-p1-r3-tol5"

    save_csv_heatmaps_to_pdf(
        input_dir=f"./generador_datos/{data_directory}",  # <-- cambia por tu carpeta
        output_pdf=f"{data_directory}.pdf",
        rows_per_page=5,
        cols_per_row=4,
        cell_px=220,
        delimiter=",",  # ajusta si usas ';' u otro
    )
    print(f"PDF generado: {data_directory}.pdf")
