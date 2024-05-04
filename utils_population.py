import numpy as np

DEFECTOR: int = 0
COOPERATOR: int = 1


def single_defector(rows: int = 20, cols: int = 20, sides: int = None) -> np.ndarray:
    """
    Genera poblacion con un unico defector ubicado en el centro de filas y columnas
    :param rows: total de filas para la poblacion
    :param cols: total de columnas para la poblacion
    :param sides: total de filas y total de columnas para la poblacion. Sobreescribe valores de rows y cols
    :return: ndarray de poblacion
    """

    dim = (sides, sides) if sides is not None else (rows, cols)

    matrix = np.full(shape=(dim[0], dim[1]), fill_value=COOPERATOR, dtype=np.int8)
    matrix[int(dim[0] / 2), int(dim[1] / 2)] = DEFECTOR
    return matrix
