import csv
import numpy as np  # TODO eliminar dependencia numpy


def import_csv(filepath):
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
        data_array = np.array(data, dtype=int)

        return data_array
