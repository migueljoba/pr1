#!/usr/bin/env python3
from pathlib import Path
import xml.etree.ElementTree as ET
import sys
import re
from math import ceil
import constants as cons

# ==============================
# CONFIGURACIÓN EDITABLE
# ==============================

# Directorio que agrupa resultados por parametros de simulaciones
sides = cons.EXPORTABLE_SIDES
seed = cons.EXPORTABLE_SEED
factor = cons.EXPORTABLE_FACTOR
tolerance = cons.EXPORTABLE_TOLERANCE
dir_name = f'{sides}-{seed}-{factor}-{tolerance}'

INPUT_DIR = f"{cons.IMAGES_DIR}/evolution/{dir_name}"
OUTPUT_SVG = f"{INPUT_DIR}/setmap.svg"

N_COLS = 4          # cantidad de columnas en la grilla
H_SPACING = 20      # separación horizontal entre imágenes (px)
V_SPACING = 20      # separación vertical entre imágenes (px)

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)


def parse_length(value: str) -> float:
    """
    Convierte '800', '800px' o '800.0' en 800.0 (float).
    Asume unidades en píxeles.
    """
    if value is None:
        raise ValueError("Atributo de longitud vacío")
    match = re.match(r"([0-9.+-eE]+)", value)
    if not match:
        raise ValueError(f"No se pudo interpretar la longitud: {value!r}")
    return float(match.group(1))


def load_svg_size(svg_path: Path):
    tree = ET.parse(svg_path)
    root = tree.getroot()

    width_attr = root.get("width")
    height_attr = root.get("height")

    if width_attr is None or height_attr is None:
        raise ValueError(f"{svg_path} no tiene atributos width/height definidos")

    width = parse_length(width_attr)
    height = parse_length(height_attr)
    return width, height


def merge_svgs_grid(input_dir: Path, output_path: Path):

    pattern = re.compile(r"^\d{3}\.svg$")
    svg_files = sorted(
        f for f in input_dir.glob("*.svg")
        if pattern.match(f.name)
    )

    if not svg_files:
        raise RuntimeError(f"No se encontraron archivos .svg en {input_dir}")

    # Tomamos el tamaño del primer SVG como referencia
    tile_width, tile_height = load_svg_size(svg_files[0])
    n = len(svg_files)

    n_cols = max(1, N_COLS)
    n_rows = ceil(n / n_cols)

    # Tamaño total del SVG combinado
    total_width = n_cols * tile_width + (n_cols - 1) * H_SPACING
    total_height = n_rows * tile_height + (n_rows - 1) * V_SPACING

    # SVG raíz
    root = ET.Element(
        f"{{{SVG_NS}}}svg",
        {
            "width": str(total_width),
            "height": str(total_height),
            "viewBox": f"0 0 {total_width} {total_height}",
        },
    )

    idx = 0
    done = False

    for row in range(n_rows):
        for col in range(n_cols):
            if idx >= n:
                done = True
                break  # no hay más SVG que colocar

            svg_path = svg_files[idx]
            idx += 1

            x_offset = col * (tile_width + H_SPACING)
            y_offset = row * (tile_height + V_SPACING)

            tree = ET.parse(svg_path)
            src_root = tree.getroot()

            group = ET.SubElement(
                root,
                f"{{{SVG_NS}}}g",
                {"transform": f"translate({x_offset}, {y_offset})"},
            )

            # Mover todos los hijos del SVG original al grupo
            for child in list(src_root):
                src_root.remove(child)
                group.append(child)

            print(f"[OK] agregado {svg_path.name} en fila {row}, columna {col}")

        if done:
            break  # salimos también del bucle de filas

    # Crear directorio de salida si no existe
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tree_out = ET.ElementTree(root)
    tree_out.write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"\nSVG combinado generado en: {output_path}")


def main():
    input_dir_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_SVG)

    if not input_dir_path.exists() or not input_dir_path.is_dir():
        print(f"Error: INPUT_DIR no es un directorio válido: {input_dir_path}", file=sys.stderr)
        sys.exit(1)

    try:
        merge_svgs_grid(input_dir_path, output_path)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
