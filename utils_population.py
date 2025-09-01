import numpy as np

DEFECTOR: int = 0
COOPERATOR: int = 1


def custom():
    p = [
        [0, 0, 0, 0],
        [0, 1, 1, 1],
        [0, 1, 1, 1],
        [0, 1, 1, 1],
    ]
    return np.array(p)

def single_cooperator(rows: int = 20, cols: int = 20, sides: int = None) -> np.ndarray:
    """
    Genera poblacion con un unico cooperador ubicado en el centro de filas y columnas
    :param rows: total de filas para la poblacion
    :param cols: total de columnas para la poblacion
    :param sides: total de filas y total de columnas para la poblacion. Sobreescribe valores de rows y cols
    :return: ndarray de poblacion
    """

    dim = (sides, sides) if sides is not None else (rows, cols)

    matrix = np.full(shape=(dim[0], dim[1]), fill_value=DEFECTOR, dtype=np.int8)
    matrix[int(dim[0] / 2), int(dim[1] / 2)] = COOPERATOR
    return matrix

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


def cluster(sides: int = 20, rows: int = 5, cols: int = 5) -> np.ndarray:
    arr = np.ones((sides, sides), dtype=int)
    arr[0:rows, 0:cols] = 0
    return arr


def frente(sides: int = 20, i: int = 5, j: int = 5) -> np.ndarray:
    arr = np.ones((sides, sides), dtype=int)
    arr[i:i + 4, 0:sides] = 0
    # arr[0:sides, j:j + 1] = 0
    return arr


def cruz(sides: int = 20, i: int = 5, j: int = 5) -> np.ndarray:
    arr = np.ones((sides, sides), dtype=int)
    arr[i:i + 1, 0:sides] = 0
    arr[0:sides, j:j + 1] = 0
    return arr


def ele_original(sides: int = 40, i: int = 5, j: int = 5, len_i: int = 5, len_j: int = 5) -> np.ndarray:
    arr = np.ones((sides, sides), dtype=int)
    arr[i, j] = 0
    arr[i:i + 1, 0:len_j + 1] = 0
    arr[0:len_i + 1, j:j + 1] = 0
    return arr


def ele(sides: int = 40, i: int = 5, j: int = 5, len_i: int = 5, len_j: int = 5) -> np.ndarray:
    arr = np.ones((sides, sides), dtype=int)

    t = 5  # grosor de la "L" en celdas

    # "L" clásica: vertical izquierda + horizontal inferior
    arr[:, :t] = 0  # barra vertical (columna izquierda)
    arr[-t:, :] = 0  # barra horizontal (fila inferior)
    return arr


def diagonal(sides: int = 20, i: int = 5, j: int = 5) -> np.ndarray:
    arr = np.ones((sides, sides), dtype=int)
    np.fill_diagonal(arr, 0)
    return arr


def equis(sides: int = 20, i: int = 5, j: int = 5) -> np.ndarray:
    arr = np.ones((sides, sides), dtype=int)
    np.fill_diagonal(arr, 0)
    np.fill_diagonal(np.fliplr(arr), 0)
    return arr
