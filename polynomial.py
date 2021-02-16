import numpy as np
import math


class Polynomial:
    def __init__(self, coefficients):
        self.coefficients = coefficients
        self.order = len(coefficients)

    def __add__(self, other: "Polynomial"):
        order = max(self.order, other.order)
        return Polynomial(np.pad(self.coefficients, (0, order - self.order)) + \
            + np.pad(other.coefficients, (0, order - other.order)))

    def __sub__(self, other):
        order = max(self.order, other.order)
        return Polynomial(np.pad(self.coefficients, (0, order - self.order)) - \
                          - np.pad(other.coefficients, (0, order - other.order)))

    def __getitem__(self, item: int):
        return self.coefficients[item]

    def __mul__(self, other: int):
        return Polynomial([c * other for c in self.coefficients])

    def __call__(self, x):
        return sum(self[i] * x ** i for i in range(len(self.coefficients)))

    def integrate(self, n):
        coefficients = [self[i] * math.factorial(i) / math.factorial(i + n)
                        for i in range(self.order)]
        return Polynomial(np.concatenate([np.zeros(n), coefficients]))

    def differentiate(self, n):
        return Polynomial(np.array([self[i] * math.factorial(i)/math.factorial(i - n)
                                    for i in range(n, self.order)]))