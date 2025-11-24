#!/usr/bin/env python3
import csv
from pathlib import Path
import sys

"""
Genera un solo archivo SVG, a partir de un solo archivo CSV
"""

# ==============================
# CONFIGURACIÓN EDITABLE
# ==============================
INPUT_CSV = "../data_source/base-dim3.csv"  # archivo CSV de entrada
OUTPUT_SVG = None  # si es None, usa mismo nombre que el CSV con extensión .svg

CELL_SIZE = 20  # tamaño de cada celda en píxeles
COLOR0 = "#e0e0e0"  # color para celdas con valor 0
COLOR1 = "#ff4444"  # color para celdas con valor 1

STROKE_COLOR = "#000"  # color del borde de las celdas
STROKE_WIDTH = 1  # ancho del borde (0.0 = sin borde)

DELIMITER = ","  # delimitador del CSV ("," o ";" por ejemplo)


def leer_csv(path_csv, delimiter=","):
    with open(path_csv, newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        datos = []
        for fila in reader:
            # Ignorar filas completamente vacías
            if not fila or all(c.strip() == "" for c in fila):
                continue
            try:
                datos.append([int(c.strip()) for c in fila])
            except ValueError as e:
                raise ValueError(f"Error al convertir a entero en la fila {fila}: {e}")

    if not datos:
        raise ValueError("El archivo CSV está vacío o sólo tiene filas vacías.")

    # Verificar que todas las filas tengan la misma longitud
    ancho = len(datos[0])
    for i, fila in enumerate(datos):
        if len(fila) != ancho:
            raise ValueError(
                f"La fila {i} tiene {len(fila)} columnas, "
                f"pero se esperaban {ancho} columnas."
            )
    return datos


def generar_svg(datos, cell_size, color0, color1, stroke_color, stroke_width):
    filas = len(datos)
    cols = len(datos[0])

    width = cols * cell_size
    height = filas * cell_size

    svg_lineas = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg"',
        f'     width="{width}" height="{height}"',
        f'     viewBox="0 0 {width} {height}">',
        "",
        f'  <!-- {filas} filas x {cols} columnas, tamaño de celda = {cell_size} -->',
    ]

    # Estilo base de las celdas
    base_style = []
    if stroke_width > 0:
        base_style.append(f"stroke:{stroke_color}")
        base_style.append(f"stroke-width:{stroke_width}")
    else:
        base_style.append("stroke:none")
    base_style_str = ";".join(base_style)

    for y, fila in enumerate(datos):
        for x, val in enumerate(fila):
            color = color1 if val == 1 else color0
            px = x * cell_size
            py = y * cell_size
            svg_lineas.append(
                f'  <rect x="{px}" y="{py}" '
                f'width="{cell_size}" height="{cell_size}" '
                f'style="fill:{color};{base_style_str}" />'
            )

    svg_lineas.append("</svg>")
    return "\n".join(svg_lineas)


def main():
    input_path = Path(INPUT_CSV)
    if not input_path.exists():
        print(f"Error: no se encontró el archivo {input_path}", file=sys.stderr)
        sys.exit(1)

    if OUTPUT_SVG is not None:
        output_path = Path(OUTPUT_SVG)
    else:
        output_path = input_path.with_suffix(".svg")

    try:
        datos = leer_csv(input_path, delimiter=DELIMITER)
    except Exception as e:
        print(f"Error al leer el CSV: {e}", file=sys.stderr)
        sys.exit(1)

    svg_contenido = generar_svg(
        datos=datos,
        cell_size=CELL_SIZE,
        color0=COLOR0,
        color1=COLOR1,
        stroke_color=STROKE_COLOR,
        stroke_width=STROKE_WIDTH,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg_contenido)

    print(f"SVG generado en: {output_path}")


if __name__ == "__main__":
    main()
