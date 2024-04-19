import numpy as np


def generate_weight_array(population, b: float):
    weight_array = np.empty(population.shape, dtype=float)

    for idx_i, row in enumerate(population):
        for idx_j, col in enumerate(row):
            neighbours = get_neighbours(arrange=population, i=idx_i, j=idx_j)
            weight_array[idx_i, idx_j] = compute_payoff(neighbours, b)

    return weight_array


def get_neighbours_idx(arrange: list = [], i: int = None, j: int = None) -> list:
    cols = len(arrange[0])
    n0 = [[i - 1, j - 1], [i - 1, j], [i - 1, (j + 1) % cols]]
    n1 = [[i, j - 1], [i, j], [i, (j + 1) % cols]]
    n2 = [[(i + 1) % cols, j - 1], [(i + 1) % cols, j], [(i + 1) % cols, (j + 1) % cols]]

    return [n0, n1, n2]


def get_neighbours(arrange: list = [], i: int = None, j: int = None) -> list:
    cols = len(arrange[0])
    n0 = [arrange[i - 1][j - 1], arrange[i - 1][j], arrange[i - 1][(j + 1) % cols]]
    n1 = [arrange[i][j - 1], arrange[i][j], arrange[i][(j + 1) % cols]]
    n2 = [arrange[(i + 1) % cols][j - 1], arrange[(i + 1) % cols][j], arrange[(i + 1) % cols][(j + 1) % cols]]

    return [n0, n1, n2]


def compute_payoff(array, b: float):
    narray = np.array(array)

    if narray.shape != (3, 3):
        raise ValueError("array must be of shape (3,3)")

    # asumir siempre que el individuo esta en (1, 1)
    individual = narray[1, 1]
    return narray.sum() if individual == 1 else narray.sum() * b


def compute_payoff_with_rule(block: list, rule: list):
    nblock = np.array(block)

    if nblock.shape != (3, 3):
        raise ValueError("array must be of shape (3,3)")
    else:
        # asumir siempre que el individuo esta en (1, 1) para matriz de orden 3x3
        individual = nblock[1, 1]

    return sum([rule[individual][neighbour] for neighbour in nblock.ravel()])


def get_highest_element_idx(array):
    array = np.array(array)
    return np.unravel_index(array.argmax(), array.shape)


def get_highest_element(array):
    return array[get_highest_element_idx(array)]
