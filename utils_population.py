import numpy as np

DEFECTOR: int = 0
COOPERATOR: int = 1


def single_defector(rows: int = 20, cols: int = 20) -> np.ndarray:
    """
    Genera poblacion con un unico defector ubicado en el centro de filas y columnas
    :param rows: total de filas para la poblacion
    :param cols: total de columnas para la poblacion
    :return: ndarray de poblacion
    """
    matrix = np.full(shape=(rows, cols), fill_value=COOPERATOR, dtype=np.int8)
    matrix[int(rows / 2), int(cols / 2)] = DEFECTOR
    return matrix
