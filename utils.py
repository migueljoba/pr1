import numpy as np

from rule import Rule


def default_random():
    return np.random.RandomState(123456789)


def random_population(elements: list, probability: list, size: tuple):
    return default_random().choice(elements, p=probability, size=size)


def generate_weight_array(population, rule: Rule):
    weight_array = np.empty(population.shape, dtype=float)

    for idx_i, idx_j in np.ndindex(population.shape):
        neighbours = get_neighbours(arrange=population, i=idx_i, j=idx_j)
        weight_array[idx_i, idx_j] = compute_payoff_with_rule(neighbours, rule)

    return weight_array


def get_neighbours_idx_i(i: int, rows):
    if rows < 3:
        raise ValueError("Cols cannot be less than 3!")
    return [i - 1, i - 1, i - 1, i, i, i, (i + 1) % rows, (i + 1) % rows, (i + 1) % rows]


def get_neighbours_idx_j(j: int, cols):
    if cols < 3:
        raise ValueError("Rows cannot be less than 3!")
    return [j - 1, j, (j + 1) % cols, j - 1, j, (j + 1) % cols, j - 1, j, (j + 1) % cols]


def get_neighbours_idx(arrange: list = [], i: int = None, j: int = None) -> list:
    rows = len(arrange)
    cols = len(arrange[0])
    idx_row = get_neighbours_idx_i(i, rows)
    idx_col = get_neighbours_idx_j(j, cols)
    return [idx_row, idx_col]


def get_neighbours(arrange: list = [], i: int = None, j: int = None) -> list:
    idx_i, idx_j = get_neighbours_idx(arrange, i, j)
    n = np.array(arrange)[idx_i, idx_j].reshape(3, 3)
    return n.tolist()  # TODO retornar ndarray


def compute_payoff(array, b: float):
    narray = np.array(array)

    if narray.shape != (3, 3):
        raise ValueError("array must be of shape (3,3)")

    # asumir siempre que el individuo esta en (1, 1)
    individual = narray[1, 1]
    return narray.sum() if individual == 1 else narray.sum() * b


def compute_payoff_with_rule(block: list, rule: Rule):
    if rule.b is None:
        raise ValueError("Rule must have b. None given.")
    if rule.matrix is None:
        raise ValueError("Rule must have matrix. None given.")

    nblock = np.array(block)

    if nblock.shape != (3, 3):
        raise ValueError("array must be of shape (3,3)")
    else:
        # asumir siempre que el individuo esta en (1, 1) para matriz de orden 3x3
        individual = nblock[1, 1]

    return sum([rule.matrix[individual][neighbour] for neighbour in nblock.ravel()])


def get_highest_element_idx(array):
    array = np.array(array)
    return np.unravel_index(array.argmax(), array.shape)


def get_highest_element(array):
    return array[get_highest_element_idx(array)]


def resume_frequency_data(collection: list, strategy: int = 1):
    if strategy not in [0, 1]:
        raise ValueError("Strategy must be 0 or 1.")

    frequency_data = []
    rows, cols = collection[0].shape
    population = rows * cols
    for col in collection:
        values, counter = np.unique(col, return_counts=True)
        strategy_counter = counter[strategy] / population
        frequency_data.append(strategy_counter)
    return frequency_data


def run(initial_population: np.ndarray, rule: Rule, generations: int):
    matrix_list = [initial_population]

    for gen in range(generations):
        current_step = np.zeros(initial_population.shape, dtype=np.int8)
        previous_step = matrix_list[-1]
        payoff_array = generate_weight_array(previous_step, rule)

        for idx_i, idx_j in np.ndindex(initial_population.shape):
            neighbours_payoff = get_neighbours(payoff_array, idx_i, idx_j)
            winner_idx = get_highest_element_idx(neighbours_payoff)
            neighbours = get_neighbours(previous_step, idx_i, idx_j)

            invader = neighbours[winner_idx[0]][winner_idx[1]]
            current = previous_step[idx_i][idx_j]
            result = rule.transition[current][invader]
            current_step[idx_i, idx_j] = result

        matrix_list.append(current_step)

    return matrix_list
