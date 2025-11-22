#!/usr/bin/env python3
import csv
from pathlib import Path
import sys

# ==============================
# CONFIGURACIÓN EDITABLE
# ==============================
# Directorio que contiene los CSV de entrada
sides = 40
factor = 1.3
tolerance = 0
dir_name = f'{sides}-{factor}-{tolerance}'
INPUT_DIR = f"../generador_datos/evolution/{dir_name}"

# Directorio donde se guardarán los SVG generados.
# Si es None, se usan los mismos directorios donde están los CSV.
OUTPUT_DIR = INPUT_DIR + "/svg"

CELL_SIZE = 20  # tamaño de cada celda en píxeles
COLOR0 = "#ff7f50"  # color para celdas con valor 0
COLOR1 = "#4682b4"  # color para celdas con valor 1

STROKE_COLOR = "#000000"  # color del borde de las celdas
STROKE_WIDTH = 1  # ancho del borde (0.0 = sin borde)

DELIMITER = ","  # delimitador del CSV ("," o ";" por ejemplo)

# Área de texto inferior (etiqueta con el nombre del archivo CSV)
LABEL_FONT_FAMILY = "monospace"           # fuente del texto
LABEL_FONT_SIZE = 14                      # tamaño de fuente en px
LABEL_MARGIN_LEFT = 5                     # margen desde el borde izquierdo
LABEL_MARGIN_TOP = 5                      # separación entre la cuadrícula y el texto


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
        raise ValueError(f"El archivo CSV '{path_csv}' está vacío o sólo tiene filas vacías.")

    # Verificar que todas las filas tengan la misma longitud
    ancho = len(datos[0])
    for i, fila in enumerate(datos):
        if len(fila) != ancho:
            raise ValueError(
                f"En '{path_csv}': la fila {i} tiene {len(fila)} columnas, "
                f"pero se esperaban {ancho} columnas."
            )
    return datos


def generar_svg(
    datos,
    cell_size,
    color0,
    color1,
    stroke_color,
    stroke_width,
    label_text: str,
):
    filas = len(datos)
    cols = len(datos[0])

    grid_width = cols * cell_size
    grid_height = filas * cell_size

    # Altura adicional para el texto inferior
    label_area_height = LABEL_FONT_SIZE + LABEL_MARGIN_TOP + 5  # un pequeño margen extra

    width = grid_width
    height = grid_height + label_area_height

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

    # Rectángulos de la cuadrícula
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

    # Texto con el nombre del archivo CSV, debajo de la cuadrícula
    text_x = LABEL_MARGIN_LEFT
    text_y = grid_height + LABEL_MARGIN_TOP + LABEL_FONT_SIZE  # baseline del texto
    svg_lineas.append(
        f'  <text x="{text_x}" y="{text_y}" '
        f'font-family="{LABEL_FONT_FAMILY}" '
        f'font-size="{LABEL_FONT_SIZE}px" '
        f'fill="#000000">'
        f'{label_text}'
        f'</text>'
    )

    svg_lineas.append("</svg>")
    return "\n".join(svg_lineas)


def procesar_csv(csv_path: Path, output_dir: Path):
    try:
        datos = leer_csv(csv_path, delimiter=DELIMITER)

        # Usamos el nombre del archivo (incluyendo .csv) como etiqueta
        label_text = csv_path.stem

        svg_contenido = generar_svg(
            datos=datos,
            cell_size=CELL_SIZE,
            color0=COLOR0,
            color1=COLOR1,
            stroke_color=STROKE_COLOR,
            stroke_width=STROKE_WIDTH,
            label_text=label_text,
        )

        output_path = output_dir / (csv_path.stem + ".svg")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(svg_contenido)

        print(f"[OK] {csv_path.name} -> {output_path}")
    except Exception as e:
        print(f"[ERROR] Al procesar '{csv_path}': {e}", file=sys.stderr)


def main():
    input_dir_path = Path(INPUT_DIR)

    if not input_dir_path.exists():
        print(f"Error: no se encontró el directorio de entrada {input_dir_path}", file=sys.stderr)
        sys.exit(1)
    if not input_dir_path.is_dir():
        print(f"Error: INPUT_DIR no es un directorio: {input_dir_path}", file=sys.stderr)
        sys.exit(1)

    # Directorio de salida
    if OUTPUT_DIR is not None:
        output_dir_path = Path(OUTPUT_DIR)
    else:
        output_dir_path = input_dir_path

    # Crear el directorio de salida si no existe
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Buscar todos los CSV en el directorio (no recursivo)
    csv_files = sorted(input_dir_path.glob("*.csv"))

    if not csv_files:
        print(f"No se encontraron archivos .csv en {input_dir_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Procesando {len(csv_files)} archivo(s) CSV en {input_dir_path}...")
    for csv_path in csv_files:
        procesar_csv(csv_path, output_dir_path)


if __name__ == "__main__":
    main()
