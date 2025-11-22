import csv
import os
import re
from pathlib import Path

import numpy as np


def import_csv(filename, directory: str = "./data_source", format: str = "csv"):
    filepath = f"{directory}/{filename}.{format}"
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        return list(reader)


def export_csv(data, filename, directory: str = "./data_source", header=None):
    # Asegurarse que el directorio existe
    os.makedirs(directory, exist_ok=True)

    if header is None:
        header = []
    # Abrimos el archivo en modo escritura ('w').
    # newline='' evita que se creen filas en blanco entre los datos.

    filepath = os.path.join(directory, filename)

    with open(filepath, 'w', newline='') as archivo_csv:
        # LineComment: Creamos un objeto escritor de CSV.
        escritor = csv.writer(archivo_csv)

        # Escribimos la fila de encabezado (opcional pero recomendado).
        if header:
            escritor.writerow(header)

        # Usamos enumerate para obtener el índice (i) y el valor (v) de cada elemento.
        for i, v in enumerate(data):
            # Escribimos una nueva fila con el índice y el valor.
            escritor.writerow([i, v])

    print(f"Archivo '{filename}' generado.")


def export_evolution_csv(data, directory, fileprefix):
    """
    Exporta cada ndarray de la lista 'data' a un archivo CSV separado.
    El nombre de cada archivo es: fileprefixXXX.csv donde XXX es el índice con 3 dígitos.
    """
    # Asegurarse que el directorio existe
    os.makedirs(directory, exist_ok=True)

    for idx, arr in enumerate(data):
        # Formatea el índice a 3 dígitos con ceros a la izquierda
        filename = f"{fileprefix}-gen-{idx:03d}.csv"
        filepath = os.path.join(directory, filename)
        # Guarda el ndarray como CSV
        np.savetxt(filepath, arr, fmt='%d', delimiter=',')
        # Puedes usar fmt='%d' porque son solo 0 y 1


def get_all_csv_files(dir: Path):
    csv_files = sorted(p for p in dir.iterdir() if p.is_file() and p.suffix == '.csv')
    if not csv_files:
        raise FileNotFoundError(f"No se encontraron CSV en {dir}")
    return csv_files


PREFIX_SUMMARY = "summary"  # configurable

# Patrón completo con grupos nombrados
PATTERN_PARSER_FILENAME = re.compile(
    rf"^(?:{PREFIX_SUMMARY}-)?"  # prefijo opcional
    r"dim(?P<dim>\d+)-"
    r"prob-d(?P<prob_d>\d\.\d)c(?P<prob_c>\d\.\d)-"
    r"radio(?P<radio>\d+)-"
    r"pay(?P<pay>\d+)-"
    r"factor(?P<factor>\d\.\d)-"
    r"tol(?P<tol>\d{2})\.(csv|pdf)$"
)


def parse_filename(filename: str | Path) -> dict[str, str]:
    """
    Extrae los valores numéricos de un nombre de archivo con el patrón definido.

    Retorna un diccionario con claves:
    dim, prob-d, prob-c, radio, pay, factor, tol
    """
    name = filename.name if isinstance(filename, Path) else filename
    m = PATTERN_PARSER_FILENAME.match(name)
    if not m:
        raise ValueError(f"Nombre de archivo inválido: {name}")

    dim = int(m.group("dim"))
    prob_d = float(m.group("prob_d"))
    prob_c = float(m.group("prob_c"))
    radio = int(m.group("radio"))
    pay = float(m.group("pay"))
    factor = float(m.group("factor"))
    tol = int(m.group("tol"))
    return {
        "dim": dim,
        "prob-d": prob_d,
        "prob-c": prob_c,
        "radio": radio,
        "pay": pay,
        "factor": factor,
        "tol": tol,
        "str": f"{dim=}, {factor=}, {tol=}",
    }
