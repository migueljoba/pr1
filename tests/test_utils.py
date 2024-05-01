import unittest

import numpy as np
from numpy.random import RandomState

import utils
from rule import Rule

rule = Rule()
rule.b = 1.3
rule.use_binary_rule()

regular_array_test = [
    [0, 1, 2, 3, 4, 5, 6],
    [7, 8, 9, 10, 11, 12, 13],
    [14, 15, 16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25, 26, 27],
    [28, 29, 30, 31, 32, 33, 34],
    [35, 36, 37, 38, 39, 40, 41],
    [42, 43, 44, 45, 46, 47, 48]
]

arrary_order_mn = [
    [0, 1, 2, 3, 4, 5, 6],
    [7, 8, 9, 10, 11, 12, 13],
    [14, 15, 16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25, 26, 27]
]

np_regular_array = np.array(regular_array_test)


def test_random_population():
    population = utils.random_population(elements=[0, 1], probability=[0.5, 0.5], size=(3, 10))
    expected = [[1, 1, 1, 1, 0, 1, 1, 0, 1, 1], [1, 1, 1, 0, 0, 0, 0, 0, 0, 1], [1, 1, 0, 0, 1, 1, 0, 0, 1, 0]]
    np.testing.assert_array_equal(population, expected)


def test_get_neighbours_idx():
    i = utils.get_neighbours_idx(regular_array_test, 0, 0)
    assert i[0] == [-1, -1, -1, 0, 0, 0, 1, 1, 1]
    assert i[1] == [-1, 0, 1, -1, 0, 1, -1, 0, 1]

    i = utils.get_neighbours_idx(regular_array_test, 0, 6)
    assert i[0] == [-1, -1, -1, 0, 0, 0, 1, 1, 1]
    assert i[1] == [5, 6, 0, 5, 6, 0, 5, 6, 0]

    i = utils.get_neighbours_idx(regular_array_test, 6, 0)
    assert i[0] == [5, 5, 5, 6, 6, 6, 0, 0, 0]
    assert i[1] == [-1, 0, 1, -1, 0, 1, -1, 0, 1]

    i = utils.get_neighbours_idx(regular_array_test, 6, 6)
    assert i[0] == [5, 5, 5, 6, 6, 6, 0, 0, 0]
    assert i[1] == [5, 6, 0, 5, 6, 0, 5, 6, 0]

    i = utils.get_neighbours_idx(regular_array_test, 1, 1)
    assert i[0] == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert i[1] == [0, 1, 2, 0, 1, 2, 0, 1, 2]

    i = utils.get_neighbours_idx(regular_array_test, 2, 2)
    assert i[0] == [1, 1, 1, 2, 2, 2, 3, 3, 3]
    assert i[1] == [1, 2, 3, 1, 2, 3, 1, 2, 3]

    i = utils.get_neighbours_idx(regular_array_test, 3, 3)
    assert i[0] == [2, 2, 2, 3, 3, 3, 4, 4, 4]
    assert i[1] == [2, 3, 4, 2, 3, 4, 2, 3, 4]

    i = utils.get_neighbours_idx(regular_array_test, 4, 4)
    assert i[0] == [3, 3, 3, 4, 4, 4, 5, 5, 5]
    assert i[1] == [3, 4, 5, 3, 4, 5, 3, 4, 5]

    i = utils.get_neighbours_idx(regular_array_test, 5, 5)
    assert i[0] == [4, 4, 4, 5, 5, 5, 6, 6, 6]
    assert i[1] == [4, 5, 6, 4, 5, 6, 4, 5, 6]


def test_get_neighbours_when_order_mn():
    i = utils.get_neighbours_idx(arrary_order_mn, 0, 6)
    assert i[0] == [-1, -1, -1, 0, 0, 0, 1, 1, 1]
    assert i[1] == [5, 6, 0, 5, 6, 0, 5, 6, 0]


def test_get_neighbours():
    n0 = utils.get_neighbours(np_regular_array, 0, 0)
    assert n0 == [[48, 42, 43], [6, 0, 1], [13, 7, 8]]

    n1 = utils.get_neighbours(np_regular_array, 0, 6)
    assert n1 == [[47, 48, 42], [5, 6, 0], [12, 13, 7]]

    n2 = utils.get_neighbours(np_regular_array, 6, 0)
    assert n2 == [[41, 35, 36], [48, 42, 43], [6, 0, 1]]

    n3 = utils.get_neighbours(np_regular_array, 6, 6)
    assert n3 == [[40, 41, 35], [47, 48, 42], [5, 6, 0]]

    n4 = utils.get_neighbours(np_regular_array, 3, 5)
    assert n4 == [[18, 19, 20], [25, 26, 27], [32, 33, 34]]


def test_compute_payoff_when_full_0():
    arr0 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    p = utils.compute_payoff(arr0, 1)
    assert p == 0


def test_compute_payoff_when_full_1():
    arr1 = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    p = utils.compute_payoff(arr1, 1)
    assert p == 9


def test_compute_payoff_when():
    arr0 = [[0, 1, 1], [1, 1, 1], [1, 1, 1]]
    p = utils.compute_payoff(arr0, 1)
    assert p == 8

    arr1 = [[0, 0, 1], [1, 1, 1], [1, 1, 1]]
    p = utils.compute_payoff(arr1, 1)
    assert p == 7

    arr2 = [[0, 0, 1], [1, 1, 1], [1, 1, 1]]
    p = utils.compute_payoff(arr2, 1)
    assert p == 7

    arr3 = [[0, 0, 0], [1, 1, 1], [1, 1, 1]]
    p = utils.compute_payoff(arr3, 1)
    assert p == 6

    arr4 = [[0, 0, 0], [0, 1, 1], [1, 1, 1]]
    p = utils.compute_payoff(arr4, 1)
    assert p == 5

    arr5 = [[0, 0, 0], [0, 0, 1], [1, 1, 1]]
    p = utils.compute_payoff(arr5, 1)
    assert p == 4

    arr6 = [[0, 0, 0], [0, 0, 0], [1, 1, 1]]
    p = utils.compute_payoff(arr6, 1)
    assert p == 3

    arr7 = [[0, 0, 0], [0, 0, 0], [0, 1, 1]]
    p = utils.compute_payoff(arr7, 1)
    assert p == 2

    arr8 = [[0, 0, 0], [0, 0, 0], [0, 0, 1]]
    p = utils.compute_payoff(arr8, 1)
    assert p == 1


def test_compute_payoff_with_rule():
    rule = Rule()
    rule.b = 1.5
    rule.use_binary_rule()

    arr1 = [[0, 1, 1], [1, 0, 1], [1, 1, 1]]
    r = utils.compute_payoff_with_rule(arr1, rule)
    assert r == 10.5


def test_resume_frequency_data():
    collection = []
    for rnd in range(10):
        rand_np = RandomState(rnd)
        rand_matrix = rand_np.choice([0, 1], p=[0.3, 0.7], size=(20, 20))
        collection.append(rand_matrix)

    # asserts strategy 0: defector
    freq_data = utils.resume_frequency_data(collection, strategy=0)
    assert freq_data == [0.3175, 0.2975, 0.3275, 0.32, 0.2525, 0.2875, 0.3, 0.3025, 0.295, 0.3]

    # asserts strategy 1: cooperator
    freq_data = utils.resume_frequency_data(collection, strategy=1)
    assert freq_data == [0.6825, 0.7025, 0.6725, 0.68, 0.7475, 0.7125, 0.7, 0.6975, 0.705, 0.7]


if __name__ == '__main__':
    unittest.main()
