import math
from functools import cached_property
from typing import Tuple

import numpy as np

from polynomial import Polynomial
from knots import Knots


class IncidenceMatrix:
    def __init__(self, array: np.ndarray):
        assert np.array_equal(np.unique(array), [0, 1])
        assert array.sum() == array.shape[1]
        self.array = array
        self.row_sums = array.sum(axis=0)
        self.row_cumulative_sum = np.cumsum(self.row_sums)

        self.n = array.shape[1] - 1
        self.k = array.shape[0]

        self.p_n = Polynomial([0] * (self.n - 1) + [1])

    @staticmethod
    def create(array: np.ndarray):
        try:
            return LagrangianIncidenceMatrix(array)
        except AssertionError:
            return IncidenceMatrix(array)

    @cached_property
    def polya_condition(self):
        return np.all(self.row_cumulative_sum >= (np.arange(self.n + 1) + 1))

    @cached_property
    def is_irreducible(self):
        return self.n == 0 or np.all(self.row_cumulative_sum[:-1] > (np.arange(self.n) + 1))

    def decompose(self) -> Tuple["IncidenceMatrix", "IncidenceMatrix"]:
        for i in range(1, self.n + 1):
            arrays = np.hsplit(self.array, [i])
            try:
                left, right = [IncidenceMatrix.create(a) for a in arrays]
            except AssertionError:
                continue
            if left.is_irreducible:
                return left, right
        raise ValueError('Matrix has no decomposition to irreducible matrices.')

    def reduce(self) -> Tuple["IncidenceMatrix", "IncidenceMatrix"]:
        up = np.copy(self.array[:, :-1])
        up[0, np.nonzero(up[0, :])[0][-1]] = 0
        down = np.copy(self.array[:, :-1])
        down[-1, np.nonzero(down[-1, :])[0][-1]] = 0
        return IncidenceMatrix.create(up), IncidenceMatrix.create(down)

    def __repr__(self):
        return self.array.__repr__()

    def solution(self, x, w) -> Polynomial:
        assert len(x) == self.k
        if isinstance(w, np.ndarray):
            w = Knots(w, self)

        if self.is_irreducible:
            up, down = self.reduce()

            up_knots = Knots(w[:, :-1], up)
            up_solution = up.solution(x, up_knots)

            down_knots = Knots(w[:, :-1], down)
            down_solution = down.solution(x, down_knots)

            x_pow_n_derivatives = np.array([math.factorial(self.n)/math.factorial(self.n - j) * x ** (self.n - j)
                                            for j in range(self.n)]).T

            derivatives_knots_up = Knots(x_pow_n_derivatives, up)
            up_n_solution = up.solution(x, derivatives_knots_up)

            derivatives_knots_down = Knots(x_pow_n_derivatives, down)
            down_n_solution = down.solution(x, derivatives_knots_down)

            c = (up_solution[1] - down_solution[1]) / (up_n_solution[1] - down_n_solution[1])

            return up_solution - (self.p_n - up_n_solution) * c
        else:
            left, right = self.decompose()

            right_g_solution = right.solution(x, Knots(w[:, left.n + 1:], right))
            right_solution = right_g_solution.integrate(left.n + 1)

            left_w = Knots(w[:, :left.n + 1] -
                                np.array([right_solution.differentiate(i)(x) for i in range(left.n + 1)]).T,
                           left)
            left_solution = left.solution(x, left_w)
            return left_solution + right_solution


class LagrangianIncidenceMatrix(IncidenceMatrix):
    def __init__(self, array: np.ndarray):
        assert array[:, 1:].sum() == 0
        super().__init__(array)
        self.first_col = array[:, 0]

    def solution(self, x, w):
        assert len(x.shape) == 1
        x = x[self.first_col.astype(bool)]
        if isinstance(w, np.ndarray):
            w = Knots(w, self)

        denominator_matrix = np.array([x**i for i in range(len(x)-1, -1, -1)])
        denominator = np.linalg.det(denominator_matrix)

        numerator_matrix = np.vstack([w.long, denominator_matrix])
        coefficients = - np.array([(-1) ** (i + 2) * np.linalg.det(np.delete(numerator_matrix, obj=i, axis=0))
                                   for i in range(len(x), 0, -1)])
        coefficients /= denominator

        return Polynomial(coefficients)


array = np.array([[1,0, 0], [0,0, 0], [1,0, 0], [0,0,1]])
m = IncidenceMatrix.create(array)
p = m.solution(np.array([1,2,3,4]), np.array([4,5, 14]))
print()