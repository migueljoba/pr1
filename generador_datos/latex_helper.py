import math
import os
from pathlib import Path

import pandas as pd
import plotly.express as px

import utils_file as uf
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
histograms_dir = Path(os.path.join(script_dir, "histograms"))
output_dir = Path(os.path.join(script_dir, "histograms"))

pdf_files = sorted(p for p in histograms_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")

files_dict = {}


def template(fn: dict[str, str]) -> str:
    """
        {%
          \SimSetup{images/sim/histogram/summary-dim30-prob-d0.2c0.8-radio1-pay1-factor3.0-}
          \begin{figure}[p]
            \centering
            \begin{tabular}{@{}ccc@{}}
              \SimPlot{$\text{tol}=00$}{tol00} &
              \SimPlot{$\text{tol}=34$}{tol34} &
              \SimPlot{$\text{tol}=67$}{tol67}
            \end{tabular}
            \caption{Dimensiones 30x30, factor $r=3.0$}
          \end{figure}
        }%
    """

    s = "{\n"
    s = s + f'summary-dim{fn["dim"]}-prob-d{fn["prob-d"]}c{fn["prob-c"]}-radio1-pay1-factor{fn["factor"]}-' + "\n"
    s = s + "\\begin{figure}[p]\n"
    s = s + "\\begin{tabular}{@{}ccc@{}}\n"

    for i in range(fn["dim"]):
        i

    return s


for file in pdf_files:
    # filename
    fn = uf.parse_filename(file.name)

    # fileset_kw = f'{fn["dim"]}-{fn["prob-d"]}-{fn["prob-c"]}-{fn["factor"]}.'
    # if fileset_kw not in files_dict:
    #    files_dict[fileset_kw] = []
    # files_dict[fileset_kw].append(fn["tol"])

    print(template(fn))
