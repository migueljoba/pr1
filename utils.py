import numpy as np


def generate_weight_array(population, b: float):
    weight_array = np.empty(population.shape, dtype=float)

    for idx_i, row in enumerate(population):
        for idx_j, col in enumerate(row):
            neighbours = get_neighbours(arrange=population, i=idx_i, j=idx_j)
            weight_array[idx_i, idx_j] = compute_payoff(neighbours, b)

    return weight_array


def get_neighbours_idx_i(i: int, cols):
    return [i - 1, i - 1, i - 1, i, i, i, (i + 1) % cols, (i + 1) % cols, (i + 1) % cols]


def get_neighbours_idx_j(j: int, cols):
    return [j - 1, j, (j + 1) % cols, j - 1, j, (j + 1) % cols, j - 1, j, (j + 1) % cols]


def get_neighbours_idx(arrange: list = [], i: int = None, j: int = None) -> list:
    cols = len(arrange[0])
    idx_row = get_neighbours_idx_i(i, cols)
    idx_col = get_neighbours_idx_j(j, cols)
    return [idx_row, idx_col]


def get_neighbours(arrange: list = [], i: int = None, j: int = None) -> list:
    cols = len(arrange[0])
    idx_i = get_neighbours_idx_i(i, cols)
    idx_j = get_neighbours_idx_j(j, cols)

    n = np.array(arrange)[idx_i, idx_j].reshape(3, 3)
    return n.tolist()  # TODO retornar ndarray


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
