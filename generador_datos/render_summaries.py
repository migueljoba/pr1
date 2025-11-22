import utils_file as uf
import utils_math as um
import utils_plotly as up
import os
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from math import ceil
from plotly.subplots import make_subplots
import re
from typing import Iterable, Optional

# Prefijo parametrizable
PREFIX = "summary-"

# Patrón base
PATTERN = re.compile(
    rf"^{PREFIX}"
    r"dim(?P<dim>\d+)-prob-d0\.2c0\.8-"
    r"radio\d-pay1-"
    r"factor(?P<factor>\d\.\d)-"
    r"tol(?P<tol>\d{2})\.csv$"
)


def filter_files(
        files: Iterable[Path],
        dim: Optional[int | str] = None,
        factor: Optional[float | str] = None,
        tol: Optional[int | str] = None,
) -> list[Path]:
    """
    Filtra nombres de archivo según los parámetros dim, factor y tol.

    - dim: entero o string, ej. 10 o "10"
    - factor: float o string, ej. 2.5 o "2.5"
    - tol: entero o string, ej. 5, "05", "5"
    - Si no se especifica filtros, se retorna la lista de entrada
    Retorna lista con los nombres que cumplen los filtros.
    """

    # if dim is None and factor is None and tol is None:
    #     raise ValueError("Debe especificarse al menos un filtro (dim, factor o tol).")

    # Normalizar filtros
    dim_str = f"{int(dim)}" if dim is not None else None
    factor_str = f"{float(factor):.1f}" if factor is not None else None
    tol_str = f"{int(tol):02d}" if tol is not None else None

    result = []
    for p in files:
        name = p.name
        m = PATTERN.match(name)
        if not m:
            raise ValueError(f"Nombre de archivo inválido: {name}")

        if dim_str is not None and m.group("dim") != dim_str:
            continue
        if factor_str is not None and m.group("factor") != factor_str:
            continue
        if tol_str is not None and m.group("tol") != tol_str:
            continue

        result.append(p)

    return result


if __name__ == '__main__':
    BASE_DIR = Path(__file__).resolve().parent
    SOURCE_DIR = BASE_DIR / 'summaries'

    files = uf.get_all_csv_files(SOURCE_DIR)
    dim_result = 100
    files = filter_files(files,
                         dim=dim_result,
                         # tol=0,
                         # factor=Nones
                         )

    total = len(files)
    cols = 3
    rows = ceil(total / cols)

    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=[uf.parse_filename(f.name)['str'] for f in files])

    for idx, f in enumerate(files):
        df = pd.read_csv(f)

        # Agregar barras al subplot (fila=1, col=1)
        row = idx // cols + 1
        col = idx % cols + 1
        fig.add_trace(go.Bar(x=df["value"], y=df["count"], text=df["count"], marker_color=up.COLOR_COOPERATOR),
                      row=row, col=col)

        y_max = df["count"].max()
        w = um.mean_weighted(values=df["value"], counts=df["count"])

        # Línea vertical
        fig.add_shape(
            col=col,
            row=row,
            type="line",
            x0=w, x1=w,
            y0=0, y1=y_max + 0.2 * y_max,
            xref="x",
            # yref="paper",
            line=dict(color=up.COLOR_DEFECTOR, width=2)
        )


        # Texto con el valor

        fig.add_annotation(
            row=row, col=col,
            x=w, y=y_max, xref="x",
            text=f"m: {w}", showarrow=False,
            yanchor="bottom", font=dict(color="black")
        )

    fig.update_xaxes(range=[0, 50])
    # fig.update_yaxes(range=[0, 2000])

    fig.update_layout(title_text="Histogramas",
                      # font=dict(family="Latin Modern Roman", size=14, color="black"),
                      # plot_bgcolor="green",
                      # paper_bgcolor="red",
                      bargap=0.2,
                      showlegend=False,
                      height=rows * 250,
                      width=cols * 500,
                      )

    # fig.show()


    # Guardar como SVG
    # ": f"{dim=}, {factor=}, {tol=}",
    OUTPUT_PATH = BASE_DIR / f"dim-{dim_result}-histogramas.svg"
    fig.write_image(str(OUTPUT_PATH), format="svg")
    print(f"SVG guardado en: {OUTPUT_PATH}")