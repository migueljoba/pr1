import utils
import numpy as np

"""
poblacion inicial:
0: free-rider
1: colaborador parcial
"""
population_strategies = utils.random_population([0, 1], [0.4, 0.6], (20, 20))
print(population_strategies)

"""
matriz de pagos inicial para la poblacion. El fondo propio de cada individuo es el mismo
"""
population_funds = np.full(fill_value=20, shape=(20, 20), dtype=np.int8)
print(population_funds)

""" valores de segmentación, para formar cluster"""
idx_ini = 5
idx_fin = 8

"""seleccionar cluster de vecinos"""
cluster_strategies = population_strategies[idx_ini:idx_fin, idx_ini:idx_fin]
print(cluster_strategies)

"""seleccionar cluster de pagos para los vecinos seleccionados en paso anterior"""
cluster_funds = population_funds[idx_ini:idx_fin, idx_ini:idx_fin]
print("cluster_funds")
print(cluster_funds)


def cluster_payoff(strategies: np.ndarray, contributions: np.ndarray):
    shape = strategies.shape

    # total individuos
    t = strategies.size

    # fondo total del cluster

    n = contributions.sum()

    common_pay = 1.5 * (n / t)
    print("common_pay")
    print(common_pay)

    payment_array = np.empty(shape=shape, dtype=np.float16)

    for idx_i, idx_j in np.ndindex(strategies.shape):

        strategy = strategies[idx_i, idx_j]

        if strategy == 0:
            # TODO ¿puedo ser defector incluso cuando aporto algo? Esto es superimportante definir
            #  porque afecta al aporte del pool
            payment_array[idx_i, idx_j] = common_pay - contributions[idx_i, idx_j]
        else:
            payment_array[idx_i, idx_j] = common_pay - contributions[idx_i, idx_j]

    return payment_array


def _generate_rand_contribution(cluster: np.ndarray) -> np.ndarray:
    result = np.empty(cluster.shape, dtype=np.int8)

    for idx_i, idx_j in np.ndindex(cluster.shape):
        result[idx_i, idx_j] = np.random.randint(0, cluster[idx_i, idx_j])

    return result


contributions = _generate_rand_contribution(cluster_funds)
print("contributions")
print(contributions)

p = cluster_payoff(cluster_strategies, contributions)
print("pago resultante")
print(p)
