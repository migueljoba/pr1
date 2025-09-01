import numpy as np

from rule import Rule, RulePgg

VARIANTS_COOPERATOR = [1, 3]

VARIANTS_DEFECTOR = [0, 2]


class Run:
    def __init__(self, pay=None):
        self.pay = pay
        self.tolerance = None

        # factor multiplicador del fondo comun
        self.factor = 1


def default_random():
    return np.random.RandomState(123456789)


def random_population(elements: list, probability: list, size: tuple, seed: int = 123456789):
    return np.random.RandomState(seed).choice(elements, p=probability, size=size)


def generate_weight_array(population, rule: Rule):
    weight_array = np.empty(population.shape, dtype=float)

    for idx_i, idx_j in np.ndindex(population.shape):
        neighbours = get_neighbours(arrange=population, i=idx_i, j=idx_j)
        weight_array[idx_i, idx_j] = compute_payoff_with_rule(neighbours, rule)

    return weight_array


def generate_payoff_array_pgg(population, rule: RulePgg):
    payoff_array = np.empty(population.shape, dtype=float)

    # % iterar sobre poblacion actual
    for idx_i, idx_j in np.ndindex(population.shape):
        # obtener vecinos
        # neighbours = get_neighbours(arrange=population, i=idx_i, j=idx_j) # se mantiene para verificar validez de vecindad de Moore con radio = 1

        if rule.border:
            neighbours = get_moore_neighbours_clip(arrange=population, i=idx_i, j=idx_j, r=rule.radio)
        else:
            neighbours = get_moore_neighbours(arrange=population, i=idx_i, j=idx_j, r=rule.radio)

        # % calcular pago para el individuo
        individual = population[idx_i, idx_j]
        payoff_array[idx_i, idx_j] = compute_payoff_with_rule_pgg(neighbours, rule, individual)

    return payoff_array


def get_neighbours_idx_i(i: int, rows):
    if rows < 3:
        raise ValueError("Cols cannot be less than 3!")

    return [
        i - 1, i - 1, i - 1,
        i, i, i,
        (i + 1) % rows, (i + 1) % rows, (i + 1) % rows
    ]


def get_neighbours_idx_j(j: int, cols):
    if cols < 3:
        raise ValueError("Rows cannot be less than 3!")
    return [j - 1, j, (j + 1) % cols,
            j - 1, j, (j + 1) % cols,
            j - 1, j, (j + 1) % cols]


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


def get_von_neumann_neighbours(arrange: np.ndarray | list, i: int, j: int) -> list[list]:
    """
    Vecindad von Neumann (radio 1) con bordes periódicos.
    - arrange: np.ndarray 2D o list (se convierte a np.ndarray)
    - i: fila, j: columna
    Retorna: [[arr[i-1,j]],
              [arr[i,j-1], arr[i,j], arr[i,j+1]],
              [arr[i+1,j]]]
    """

    # Convertir a np.ndarray si vino como list
    if isinstance(arrange, list):
        arrange = np.asarray(arrange)

    if not isinstance(arrange, np.ndarray) or arrange.ndim != 2:
        raise ValueError("'arrange' debe ser un np.ndarray 2D o list convertible a 2D")

    m, n = arrange.shape

    # radio fijo = 1, con wrap
    up = arrange[(i - 1) % m, j % n]
    left = arrange[i % m, (j - 1) % n]
    mid = arrange[i % m, j % n]
    right = arrange[i % m, (j + 1) % n]
    down = arrange[(i + 1) % m, j % n]

    return [[up], [left, mid, right], [down]]


def get_moore_neighbours(arrange: np.ndarray | list, i: int, j: int, r: int = 1) -> list[list]:
    if not isinstance(arrange, np.ndarray) or arrange.ndim != 2:
        arrange = np.array(arrange)

    m, n = arrange.shape
    ii = (np.arange(i - r, i + r + 1) % m)
    jj = (np.arange(j - r, j + r + 1) % n)

    # producto cartesiando
    window = arrange[np.ix_(ii, jj)]
    return window.tolist()


def get_moore_neighbours_clip(arrange: np.ndarray | list, i: int, j: int, r: int = 1) -> list[list]:
    """
    Vecindad de Moore con bordes LIMITADOS alrededor de (i, j).
    - arrange: np.ndarray 2D o list (se convierte internamente a ndarray)
    - i, j: índices de fila y columna (deben estar dentro del arreglo)
    - r: radio >= 0
    Retorna: submatriz (lista de listas). Cerca de bordes, la ventana se recorta.
    """
    # Acepta list y lo convierte
    if isinstance(arrange, list):
        arrange = np.asarray(arrange)

    if not isinstance(arrange, np.ndarray) or arrange.ndim != 2:
        raise ValueError("'arrange' debe ser un np.ndarray 2D o list convertible a 2D")
    if r < 0:
        raise ValueError("r debe ser >= 0")

    m, n = arrange.shape
    if not (0 <= i < m and 0 <= j < n):
        raise ValueError("(i, j) fuera de los límites del arreglo")

    # Límites recortados (clip)
    i0 = max(0, i - r)
    i1 = min(m - 1, i + r)
    j0 = max(0, j - r)
    j1 = min(n - 1, j + r)

    window = arrange[i0:i1 + 1, j0:j1 + 1]
    return window.tolist()


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


def payoff_pgg(*, contribution: float, factor: float, total_coop: int, population: int) -> float:
    if population <= 0:
        raise ValueError("population debe ser > 0")
    if not (0 <= total_coop <= population):
        raise ValueError("total_coop debe estar entre 0 y population")
    if contribution < 0:
        raise ValueError("contribution debe ser ≥ 0")
    if factor < 0:
        raise ValueError("factor debe ser ≥ 0")

    return (factor / population) * total_coop * contribution


def compute_payoff_with_rule_pgg(block: list, rule: RulePgg, individual: int = None):
    if rule.pay is None:
        raise ValueError("Rule must have pay. None given.")
    if rule.tolerance is None:
        raise ValueError("Rule must have tolerance. None given.")

    nblock = np.array(block)

    if not rule.border:
        # control de tamano de vecindad, si no existen bordes
        required_shape = 2 * rule.radio + 1

        if nblock.shape != (required_shape, required_shape):
            raise ValueError(f"array must be of shape ({required_shape},{required_shape})")
        else:
            # elemento central en una vecindad de Moore. Sus coorenadas relativas siempre son (r, r)
            individual = nblock[rule.radio, rule.radio]

    else:
        # sí existe borde en la regla
        pass

    # total de individuos en el bloque
    t = nblock.size

    # total de cooperadores en el bloque. Cooperadores son valores 1 y 3
    n = np.sum(nblock == 1) + np.sum(nblock == 3)

    # pago que recibe cada individuo, independiente a estrategia
    # common_pay = rule.factor * rule.pay * (n / t)
    common_pay = payoff_pgg(contribution=rule.pay, factor=rule.factor, total_coop=n, population=t)

    # pago del individuo de interes; indice [r,r] del bloque
    if individual in VARIANTS_COOPERATOR:
        # pago para cooperador
        individual_payoff = common_pay - rule.pay

    else:
        # pago para free rider
        individual_payoff = common_pay

    return individual_payoff


def get_highest_element_idx(array):
    array = np.array(array)
    return np.unravel_index(array.argmax(), array.shape)


def get_highest_element(array):
    return array[get_highest_element_idx(array)]


def resume_frequency_data(collection: list, strategy: list = [1, 3], break_loop: bool = True):
    if not isinstance(strategy, list):
        raise ValueError(f"Specified `strategy` param must be {type([])}. Got `{type(strategy)}` instead.")

    frequency_data = []
    rows, cols = collection[0].shape
    population = rows * cols

    for col in collection:
        values, counter = np.unique(col, return_counts=True)
        count = 0
        for v, c in zip(values, counter):
            if v in strategy:
                count = count + c

        ratio = count / population

        frequency_data.append(ratio)

        if ratio == 0 and break_loop:
            break

    return frequency_data


def resume_frequency_factor_data():
    pass


def run(initial_population: np.ndarray, rule: Rule, generations: int, verbose: bool = False) -> list:
    matrix_list = [initial_population]

    for gen in range(generations):

        if verbose:
            print(f"Generation {gen + 1}/{generations}")

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


def run_pgg(initial_population: np.ndarray, rule: RulePgg, generations: int, verbose: bool = False,
            stop_when_all: list | int = None, stop_when_repeat: int = None) -> dict:
    matrix_list = [initial_population]
    payoff_list = []
    # generations = rule.generations

    # % iterar generaciones
    for gen in range(generations - 1):
        if verbose:
            print(f"Generation {gen + 1}/{generations}")

        current_step = np.zeros(initial_population.shape, dtype=np.int8)
        previous_step = matrix_list[-1]

        # % generar matriz de pagos
        payoff_array = generate_payoff_array_pgg(previous_step, rule)
        payoff_list.append(payoff_array)

        # % iterar poblacion actual
        for idx_i, idx_j in np.ndindex(initial_population.shape):

            # % obtener pago para indice i,j
            payoff = payoff_array[idx_i, idx_j]

            estado_previo = previous_step[idx_i][idx_j]

            # % tolerancia: evaluar estrategia segun el pago obtenido
            if payoff >= rule.pay * (1 - (rule.tolerance / 100)):
                # % pago tolerable. Conservar estrategia
                current_step[idx_i, idx_j] = rule.transition[estado_previo][estado_previo]

            else:
                # % pago no tolerable. Cambiar a estrategia desertora
                if estado_previo in VARIANTS_COOPERATOR:
                    current_step[idx_i, idx_j] = rule.transition[estado_previo][0]

                # else:
                # comentar D -> C, o
                # agregar probabilidad para convertir
                # current_step[idx_i, idx_j] = 1

        # si paso actual y anterior son iguales, asumir que ya habrá evolución y terminar simulación
        if np.all(previous_step == current_step):
            # break
            pass  # TODO detener simulacion cuando las matrices ya no evolucionen

        matrix_list.append(current_step)

        if stop_when_all is not None and np.all(np.isin(current_step, stop_when_all)):
            # detener simulacion cuando todos los individuos tengan un valor especifico
            break

    return {'matrix_list': matrix_list, 'payoff_list': payoff_list}


def custom_range(start, stop, step=1):
    n = int(round((stop - start) / float(step)))
    if n > 1:
        return [start + step * i for i in range(n + 1)]
    elif n == 1:
        return [start]
    else:
        return []
