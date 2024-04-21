import numpy as np
from numpy.random import RandomState

import utils

rule0 = utils.Rule()
rule0.b = 1.9
rule0.matrix = [
    [0, rule0.b],
    [0, 1]
]


def test_rule():
    assert rule0.b == 1.8
    assert rule0.matrix == [
        [0, 1.8],
        [0, 1]
    ]


def test_matrix_0():
    """
    Tests generated weight array from a population order 6,7 (not square) filled by zeros
    """
    full_0 = np.full(shape=(6, 7), fill_value=0)
    expected = np.full((6, 7), 0)
    weight0 = utils.generate_weight_array(full_0, rule0)
    np.testing.assert_array_equal(weight0, full_0)


def test_matrix_1():
    """
    Tests generated weight array from a population order 6,7 (not square) filled by ones
    """
    full_1 = np.full(shape=(6, 7), fill_value=1)
    expected = np.full((6, 7), 9)
    weight1 = utils.generate_weight_array(full_1, rule0)
    np.testing.assert_array_equal(expected, weight1)


def test_generate_weight_array_with_random_binary_population():
    """
    Tests generated weight array from a randomly filled population of 0 and 1.
    Seed MUST be 123456789
    """
    rand_np = RandomState(123456789)
    b = rand_np.randint(2, size=(10, 10), dtype=int)

    expected = [[3.6, 5.4, 7.2, 4., 7.2, 3., 9., 5., 7.2, 3.6],
                [2., 4., 5., 7.2, 7.2, 7.2, 7., 6., 7.2, 3.6],
                [5.4, 10.8, 6., 5., 5.4, 4., 6., 6., 4., 3.6],
                [5.4, 5., 5., 7.2, 4., 7.2, 10.8, 6., 9., 5.4],
                [4., 5., 5.4, 5.4, 7.2, 4., 9., 4., 4., 5.4],
                [4., 7.2, 3.6, 3.6, 3., 4., 9., 4., 7.2, 7.2],
                [7.2, 9., 4., 7.2, 7.2, 9., 6., 9., 7.2, 4.],
                [3., 5., 6., 5., 7.2, 5., 7., 6., 4., 5.4],
                [3.6, 7.2, 6., 6., 10.8, 5., 7., 7., 9., 5.4],
                [0., 1.8, 7.2, 5., 6., 7.2, 10.8, 5., 4., 1.8]]

    weight0 = utils.generate_weight_array(b, rule0)
    np.testing.assert_equal(expected, weight0)
