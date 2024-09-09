import utils_population as up


def test_const_values():
    assert up.DEFECTOR == 0
    assert up.COOPERATOR == 1


def test_single_defector_when_rows_and_cols():
    population = up.single_defector(15, 15)
    assert population.shape == (15, 15)


def test_single_defector_when_sides_is_specified():
    population = up.single_defector(sides=14)
    assert population.shape == (14, 14)

    population = up.single_defector(cols=15, sides=30)
    assert population.shape == (30, 30)

    population = up.single_defector(15, 15, 33)
    assert population.shape == (33, 33)
