#!/usr/bin/env python3
from pathlib import Path
import xml.etree.ElementTree as ET
import sys
import re
import constants as cons

# ==============================
# CONFIGURACIÓN EDITABLE
# ==============================

# Directorio que contiene los CSV de entrada
sides = 40
seed = 0
factor = 1
tolerance = 0
dir_name = f'{sides}-{seed}-{factor}-{tolerance}'

INPUT_DIR = "/Users/mbaez/dev_personal/pr1/generador_datos/evolution/40-1.3-0/svg"
OUTPUT_SVG = "/Users/mbaez/dev_personal/pr1/generador_datos/evolution/40-1.3-0/svg/merged.svg"  # archivo SVG combinado

OUTPUT_DIR = f"{cons.IMAGES_DIR}/evolution/{dir_name}"

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


def merge_svgs_horizontally(input_dir: Path, output_path: Path):
    svg_files = sorted(input_dir.glob("*.svg"))
    if not svg_files:
        raise RuntimeError(f"No se encontraron archivos .svg en {input_dir}")

    # Tomamos el tamaño del primer SVG como referencia
    first_width, first_height = load_svg_size(svg_files[0])
    n = len(svg_files)

    total_width = first_width * n
    total_height = first_height

    # Crear SVG raíz combinado
    root = ET.Element(
        f"{{{SVG_NS}}}svg",
        {
            "width": str(total_width),
            "height": str(total_height),
            "viewBox": f"0 0 {total_width} {total_height}",
        },
    )

    # Para cada SVG, copiamos sus hijos dentro de un <g> trasladado
    for idx, svg_path in enumerate(svg_files):
        tree = ET.parse(svg_path)
        src_root = tree.getroot()

        x_offset = first_width * idx

        group = ET.SubElement(
            root,
            f"{{{SVG_NS}}}g",
            {"transform": f"translate({x_offset}, 0)"},
        )

        # Mover todos los hijos del SVG original al grupo
        for child in list(src_root):
            src_root.remove(child)
            group.append(child)

        print(f"[OK] agregado {svg_path.name} en posición {idx}")

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
        merge_svgs_horizontally(input_dir_path, output_path)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
