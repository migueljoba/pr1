import pandas as pd
import plotly.graph_objs as go
import glob
import os

"""
Generador de gráfico interactivo de frecuencias, a partir de datos generados previamente
"""


# Ruta y carga de archivos
script_dir = os.path.dirname(os.path.abspath(__file__))
files = sorted(glob.glob(os.path.join(script_dir, 'frequency/*.csv')))

if not files:
    raise FileNotFoundError(f"No se encontraron archivos dentro de {script_dir}")

# Obtener solo el nombre del archivo, no la ruta completa
nombres = [os.path.basename(f) for f in files]

curvas = [pd.read_csv(f) for f in files]

traces = []
for i, df in enumerate(curvas):
    traces.append(
        go.Scatter(
            x=df['indice'], y=df['valor'],
            mode='lines+markers',
            name=nombres[i],  # la leyenda será el nombre de archivo
            visible=(i==0)
        )
    )

steps = []
for i in range(len(traces)):
    step = dict(
        method="update",
        args=[
            {"visible": [j == i for j in range(len(traces))]},
            {"title": f"Curva: {nombres[i]}"}
        ],
        label=f"{i+1}"
    )
    steps.append(step)

sliders = [dict(
    active=0,
    currentvalue={"prefix": "Curva: "},
    pad={"t": 50},
    steps=steps
)]

# El título inicial es el nombre del primer archivo
fig = go.Figure(data=traces)
fig.update_layout(
    sliders=sliders,
    xaxis=dict(range=[0, 50], title='Índice'),
    yaxis=dict(range=[0, 1], title='Valor'),
    title=f"Curva: {nombres[0]}"
)
fig.show()
