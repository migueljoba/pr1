import unittest

import numpy as np
from numpy.random import RandomState

import utils
from rule import Rule, RulePgg


def test_use_default_transition():
    r = RulePgg()

    # 0 -> 0 = 0
    assert 0 == r.transition[0][0]
    # 1 -> 1 = 1
    assert 1 == r.transition[1][1]
    # 1 -> 0 = 0
    assert 0 == r.transition[1][0]


def test_use_3s_transition():
    r = RulePgg()
    r.use_3s_transition()

    assert 0 == r.transition[0][0]
    assert 3 == r.transition[0][1]
    assert 0 == r.transition[0][2]
    assert 3 == r.transition[0][3]

    assert 2 == r.transition[1][0]
    assert 1 == r.transition[1][1]
    assert 2 == r.transition[1][2]
    assert 1 == r.transition[1][3]

    assert 0 == r.transition[2][0]
    assert 3 == r.transition[2][1]
    assert 0 == r.transition[2][2]
    assert 3 == r.transition[2][3]


def test_use_4s_transition():
    r = Rule()
    r.use_4s_transition()
    assert 0 == r.transition[0][0]
    assert 3 == r.transition[0][1]
    assert 0 == r.transition[0][2]
    assert 3 == r.transition[0][3]

    assert 2 == r.transition[1][0]
    assert 1 == r.transition[1][1]
    assert 2 == r.transition[1][2]
    assert 1 == r.transition[1][3]

    assert 0 == r.transition[2][0]
    assert 3 == r.transition[2][1]
    assert 0 == r.transition[2][2]
    assert 3 == r.transition[2][3]

    assert 2 == r.transition[3][0]
    assert 1 == r.transition[3][1]
    assert 2 == r.transition[3][2]
    assert 1 == r.transition[3][3]


if __name__ == '__main__':
    unittest.main()
