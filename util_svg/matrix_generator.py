#!/usr/bin/env python3
import sys
from pathlib import Path
import constants as cons
"""
Genera un solo archivo SVG, a partir de dimensiones (rows x cols),
sin usar un archivo CSV como entrada.

Cada celda es un rectángulo, y dentro de cada celda se escribe texto
según el modo seleccionado:
- mode_absolute: numera las celdas desde 0 en adelante.
- mode_index: escribe el par de índices (i,j) de cada celda.
"""

# ==============================
# CONFIGURACIÓN EDITABLE
# ==============================

# Dimensiones de la cuadrícula
ROWS = 10  # número de filas
COLS = 10  # número de columnas

# Modos de contenido de texto
MODE_ABSOLUTE = True  # True: escribe 0,1,2,3,... en cada celda
MODE_INDEX = not MODE_ABSOLUTE  # True: escribe (i,j) en cada celda

# Ruta de salida del SVG (obligatoria)
OUTPUT_DIR = f"{cons.IMAGES_DIR}/algoritmos/pgg"

if MODE_ABSOLUTE:
    OUTPUT_SVG = f"{OUTPUT_DIR}/matrix-abs-{ROWS}-{COLS}.svg"

if MODE_INDEX:
    OUTPUT_SVG = f"{OUTPUT_DIR}/matrix-idx-{ROWS}-{COLS}.svg"

CELL_SIZE = 40  # tamaño de cada celda en píxeles

# Colores de las celdas
CELL_FILL_COLOR = "#ffffff"  # color de fondo por defecto (blanco)
STROKE_COLOR = "#000000"  # color del borde de las celdas (negro)
STROKE_WIDTH = 1  # ancho del borde (0.0 = sin borde)
FONT_FAMILY = "LMRoman10"  # "Latin Modern Roman", # fuente del texto

def generar_svg_grid(
        rows,
        cols,
        cell_size,
        fill_color,
        stroke_color,
        stroke_width,
        mode_absolute,
        mode_index,
):
    """
    Genera el contenido SVG para una cuadrícula de rows x cols.
    """
    width = cols * cell_size
    height = rows * cell_size

    svg_lineas = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg"',
        f'     width="{width}" height="{height}"',
        f'     viewBox="0 0 {width} {height}">',
        "",
        f'  <!-- {rows} filas x {cols} columnas, tamaño de celda = {cell_size} -->',
    ]

    # Estilo base de las celdas
    base_style = []
    if stroke_width > 0:
        base_style.append(f"stroke:{stroke_color}")
        base_style.append(f"stroke-width:{stroke_width}")
    else:
        base_style.append("stroke:none")
    base_style_str = ";".join(base_style)

    # Texto: dejamos fuente/tamaño por defecto del SVG
    # Solo centramos el texto en la celda
    text_style = f'text-anchor="middle" dominant-baseline="middle" font-family="{FONT_FAMILY}"'

    # Contador para mode_absolute
    contador = 0

    for i in range(rows):
        for j in range(cols):
            px = j * cell_size
            py = i * cell_size

            # Rectángulo de la celda
            svg_lineas.append(
                f'  <rect x="{px}" y="{py}" '
                f'width="{cell_size}" height="{cell_size}" '
                f'style="fill:{fill_color};{base_style_str}" />'
            )

            # Contenido de texto según el modo
            texto = ""
            if mode_absolute:
                texto = str(contador)
            elif mode_index:
                texto = f"({i},{j})"

            # Solo agregamos <text> si hay algo que mostrar
            if texto:
                cx = px + cell_size / 2
                cy = py + cell_size / 2
                svg_lineas.append(
                    f'  <text x="{cx}" y="{cy}" {text_style}>{texto}</text>'
                )

            contador += 1

    svg_lineas.append("</svg>")
    return "\n".join(svg_lineas)


def main():
    # Validaciones básicas de configuración
    if ROWS <= 0 or COLS <= 0:
        print("Error: ROWS y COLS deben ser mayores que 0.", file=sys.stderr)
        sys.exit(1)

    if OUTPUT_SVG is None:
        print("Error: Debes especificar OUTPUT_SVG.", file=sys.stderr)
        sys.exit(1)

    if MODE_ABSOLUTE and MODE_INDEX:
        print(
            "Error: MODE_ABSOLUTE y MODE_INDEX no pueden ser True al mismo tiempo.",
            file=sys.stderr,
        )
        sys.exit(1)

    output_path = Path(OUTPUT_SVG)
    # Crear el directorio si no existe
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    svg_contenido = generar_svg_grid(
        rows=ROWS,
        cols=COLS,
        cell_size=CELL_SIZE,
        fill_color=CELL_FILL_COLOR,
        stroke_color=STROKE_COLOR,
        stroke_width=STROKE_WIDTH,
        mode_absolute=MODE_ABSOLUTE,
        mode_index=MODE_INDEX,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg_contenido)

    print(f"SVG generado en: {output_path}")


if __name__ == "__main__":
    main()
