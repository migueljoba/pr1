import unittest

import numpy as np
from numpy.random import RandomState
import pytest
import utils_file as util
from rule import Rule, RulePgg


def test_parse_no_prefix():
    n0 = "dim40-prob-d0.2c0.8-radio1-pay1-factor3.0-tol00.csv"
    p0 = util.parse_filename(n0)
    assert p0["dim"] == 40
    assert p0["prob-d"] == 0.2
    assert p0["prob-c"] == 0.8
    assert p0["radio"] == 1
    assert p0["pay"] == 1
    assert p0["factor"] == 3.0
    assert p0["tol"] == 0
    assert p0["str"] == 'dim=40, factor=3.0, tol=0'

    n1 = "summary-dim40-prob-d0.2c0.8-radio1-pay1-factor3.0-tol05.csv"
    p1 = util.parse_filename(n1)
    assert p1["str"] == 'dim=40, factor=3.0, tol=5'

    n2 = "summary-dim40-prob-d0.2c0.8-radio1-pay1-factor3.5-tol05.csv"
    p2 = util.parse_filename(n2)
    assert p2["str"] == 'dim=40, factor=3.5, tol=5'


def test_parse():
    n0 = "summary-dim40-prob-d0.2c0.8-radio1-pay1-factor3.0-tol00.csv"
    p0 = util.parse_filename(n0)
    assert p0["dim"] == 40
    assert p0["prob-d"] == 0.2
    assert p0["prob-c"] == 0.8
    assert p0["radio"] == 1
    assert p0["pay"] == 1
    assert p0["factor"] == 3.0
    assert p0["tol"] == 0
    assert p0["str"] == 'dim=40, factor=3.0, tol=0'

    n1 = "summary-dim40-prob-d0.2c0.8-radio1-pay1-factor3.0-tol05.csv"
    p1 = util.parse_filename(n1)
    assert p1["str"] == 'dim=40, factor=3.0, tol=5'

    n2 = "summary-dim40-prob-d0.2c0.8-radio1-pay1-factor3.5-tol05.csv"
    p2 = util.parse_filename(n2)
    assert p2["str"] == 'dim=40, factor=3.5, tol=5'

    n3 = "summary-dim40-prob-d0.2c0.8-radio1-pay1-factor3.5-tol05.pdf"
    p3 = util.parse_filename(n3)
    assert p3["str"] == 'dim=40, factor=3.5, tol=5'
