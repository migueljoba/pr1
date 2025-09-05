import utils_file as uf
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


def _get_all_csv_files(dir: Path):
    csv_files = sorted(p for p in dir.iterdir() if p.is_file() and p.suffix == '.csv')
    if not csv_files:
        raise FileNotFoundError(f"No se encontraron CSV en {dir}")
    return csv_files


# Patrón base
PATTERN = re.compile(
    rf"^{PREFIX}"
    r"dim(?P<dim>\d+)-prob-d0\.2c0\.8-"
    r"radio1-pay1-"
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

    - Al menos uno de los filtros debe ser distinto de None.
    - dim: entero o string, ej. 10 o "10"
    - factor: float o string, ej. 2.5 o "2.5"
    - tol: entero o string, ej. 5, "05", "5"
    Retorna lista con los nombres que cumplen los filtros.
    """

    if dim is None and factor is None and tol is None:
        raise ValueError("Debe especificarse al menos un filtro (dim, factor o tol).")

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
    SOURCE_DIR = BASE_DIR / 'simulations' / 'summaries'
    # files = _get_all_csv_files(SOURCE_DIR)
    files = uf.get_all_csv_files(SOURCE_DIR)
    # files = ["summary-dim10-prob-d0.2c0.8-radio1-pay1-factor2.5-tol05.csv"]
    files = filter_files(files, dim=40)

    total = len(files)
    cols = 5
    rows = ceil(total / cols)

    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=[f"Gráfico {i + 1}" for i in range(total)])

    for idx, f in enumerate(files):
        df = pd.read_csv(f)

        # Agregar barras al subplot (fila=1, col=1)
        row = idx // cols + 1
        col = idx % cols + 1
        fig.add_trace(go.Bar(x=df["value"], y=df["count"], name="A", text=df["count"]),
                      row=row, col=col)

    fig.update_xaxes(range=[0, 50])
    # fig.update_yaxes(range=[0, 2000])

    fig.update_layout(title_text="Ejemplo de Subplots con Plotly",
                      # font=dict(family="Latin Modern Roman", size=14, color="black"),
                      # plot_bgcolor="white",
                      # paper_bgcolor="white",
                      bargap=0.2,
                      showlegend=False,
                      height=rows * 250,
                      )

    fig.show()
