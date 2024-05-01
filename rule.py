class Rule:
    def __init__(self, b=None):
        self.b = b
        self.matrix = None
        self.transition = None

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
        self.transition = [
            [0, 3, 0, 3],
            [2, 1, 2, 1],
            [0, 3, 0, 3],
            [2, 1, 2, 1],
        ]
