#!/usr/bin/env python3
import re
from pathlib import Path

# Nombre del archivo de salida en el mismo directorio del script
OUTPUT_FILENAME = "./resumen_consolidado.txt"

PATTERN = re.compile(
    r"""
    ^summary
    -dim(?P<sides>\d+)
    -prob-d(?P<prob_defector>\d+(?:\.\d+)?)
          c(?P<prob_cooperator>\d+(?:\.\d+)?)
    -radio(?P<radio>\d+)
    -pay(?P<pay>\d+)
    -factor(?P<factor>\d+(?:\.\d+)?)
    -tol(?P<tolerance>\d+)
    \.csv$
    """,
    re.VERBOSE | re.IGNORECASE,
)


def parse_params(filename: str) -> str:
    m = PATTERN.match(filename)
    if not m:
        raise ValueError(f"Nombre de archivo no cumple el patrón esperado: {filename}")
    d = m.groupdict()
    factor = f"{float(d['factor']):.1f}"  # #.#   (un decimal)
    tolerance = f"{int(d['tolerance']):02d}"  # dos dígitos
    return (
        f"sides={d['sides']},"
        f"prob_defector={d['prob_defector']},"
        f"prob_cooperator={d['prob_cooperator']},"
        f"radio={d['radio']},"
        f"pay={d['pay']},"
        f"factor={factor},"
        f"tolerance={tolerance}"
    )


def clean_trailing_blank_lines(text: str) -> str:
    # Normaliza saltos de línea y elimina solo líneas en blanco al final
    lines = text.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    while lines and lines[-1].strip() == "":
        lines.pop()
    return "\n".join(lines)


def main():
    base_dir = Path(__file__).resolve().parent
    out_path = base_dir / OUTPUT_FILENAME

    csv_files = sorted(p for p in base_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv")

    if not csv_files:
        raise FileNotFoundError(f"No se encontraron archivos .csv en {base_dir}")

    with out_path.open("w", encoding="utf-8", newline="\n") as out:
        for idx, csv_path in enumerate(csv_files):
            params_line = parse_params(csv_path.name)  # lanza excepción si no coincide
            out.write(f"{csv_path.name}\n")
            out.write(f"{params_line}\n")
            content = csv_path.read_text(encoding="utf-8")
            content = clean_trailing_blank_lines(content)
            out.write(content)
            # Separador entre archivos (solo si no es el último)
            if idx < len(csv_files) - 1:
                out.write("\n####################")
                out.write("\n")

    print(f"Archivos encontrados: {len(csv_files)}")


if __name__ == "__main__":
    main()
