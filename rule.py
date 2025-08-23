TRANSITION_4S = [
    [0, 3, 0, 3],
    [2, 1, 2, 1],
    [0, 3, 0, 3],
    [2, 1, 2, 1],
]


class RulePgg:
    def __init__(self, pay=None):
        self.pay = pay
        self.tolerance = None

        # transicion de dos estados, por defecto
        self.transition = [
            [0, 1],
            [0, 1]
        ]

        # factor multiplicador del fondo comun
        self.factor = 1

        # numero de lados para matriz cuadrada
        self.sides = None

        # numero de generaciones para la simulacion
        self.generations = None

        # radio para vecindad de Moore
        self.radio = 1

        self.info_seed = 123456789

        self.info_generations = None

    def __str__(self):
        return f"Rule PGG. pay:{self.pay}, factor:{self.factor}, radio: {self.radio}, tolerance:{self.tolerance}"

    def use_3s_transition(self):
        self.transition = TRANSITION_4S

    def csv_row(self):
        return (f"{self.sides},{self.radio},{self.pay},{self.factor},"
                f"{self.tolerance},{self.info_seed},{self.info_generations}")

    def params_str(self):
        return f"dim{self.sides}-radio{self.radio}-pay{self.pay}-factor{self.factor}-tol{self.tolerance}"

    @staticmethod
    def csv_headers():
        return ["sides", "radio", "pay", "factor", "tolerance", "info_seed", "info_generations"]


class Rule:
    def __init__(self, b=None):
        self.b = b
        self.matrix = None
        self.transition = None

    def update_b(self, b: float):
        self.b = b

    def _check_value_b(self):
        if self.b is None:
            raise Exception(f"'b' value has not been defined or is invalid: {self.b}")

    def use_binary_rule(self):
        self._check_value_b()

        self.matrix = [
            [0, self.b],
            [0, 1]
        ]

    def use_binary_transition(self):
        self.transition = [
            [0, 1],
            [0, 1]
        ]

    def use_4s_rule(self):
        self._check_value_b()
        self.matrix = [
            [0, self.b, 0, self.b],
            [0, 1, 0, 1],
            [0, self.b, 0, self.b],
            [0, 1, 0, 1]
        ]

    def use_4s_transition(self):
        self.transition = TRANSITION_4S
