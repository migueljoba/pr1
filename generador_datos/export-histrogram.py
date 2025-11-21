import math
import os
from pathlib import Path

import pandas as pd
import plotly.express as px

"""
Generar gráficos en formato PDF masivamente para los CSV de resumen de frecuencias
dentro del directorio pr1/generador_datos/simulations/sumaries
"""


# === Calcular media y desviación estándar ponderadas ===
def mean_weighted(values, counts):
    N = sum(counts)
    return sum(v * c for v, c in zip(values, counts)) / N


def variance_weighted(values, counts):
    mu = mean_weighted(values, counts)
    N = sum(counts)
    return sum(c * (v - mu) ** 2 for v, c in zip(values, counts)) / N


def std_weighted(values, counts):
    return math.sqrt(variance_weighted(values, counts))


script_dir = os.path.dirname(os.path.abspath(__file__))
summaries_dir = Path(os.path.join(script_dir, "summaries"))
output_dir = Path(os.path.join(script_dir, "histograms"))

csv_files = sorted(p for p in summaries_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv")

if not csv_files:
    raise FileNotFoundError(f"No se encontraron archivos .csv en {summaries_dir}")

for file in csv_files:
    df = pd.read_csv(file, sep=",", encoding="utf-8")

    values = df["value"].tolist()
    counts = df["count"].tolist()

    mean_val = mean_weighted(values, counts)
    std_val = std_weighted(values, counts)

    # === Crear gráfico con Plotly ===
    fig = px.bar(df, x="value", y="count",
                 labels={"value": "Valor (generaciones)", "count": "Frecuencia"},
                 title="Distribución de info_generations")

    # Agregar líneas de referencia
    fig.add_vline(x=mean_val, line_dash="dash", line_color="red",
                  annotation_text=f"Media={mean_val:.2f}", annotation_position="top left", fillcolor="red")

    fig.add_vline(x=mean_val - std_val, line_dash="dot", line_color="green", annotation_text="-1σ",
                  annotation_position="bottom left")

    fig.add_vline(x=mean_val + std_val, line_dash="dot", line_color="green", annotation_text="+1σ",
                  annotation_position="bottom right")

    fig.update_xaxes(range=[0, 50])

    # === Guardar como SVG o PDF ===
    # Requiere instalar kaleido una sola vez:  pip install -U kaleido
    output_dir.mkdir(parents=True, exist_ok=True)

    fig.write_image(os.path.join(output_dir, f"{file.stem}.pdf"))
    # fig.write_image("histograma2.pdf")
    # fig.show()
    print(f"Archivo '{file.stem}.pdf' generado.")
