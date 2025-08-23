import csv
import os
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


if __name__ == "__main__":
    # Crear algunos datos de ejemplo
    data = [np.random.randint(0, 2, size=(5, 5)), np.random.randint(0, 2, size=(3, 7))]
    export_evolution_csv(data, "resultados", "poblacion_")
    print("¡Archivos exportados!")
