import math
from functools import cached_property
from typing import Tuple

import numpy as np

from polynomial import Polynomial


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

    @cached_property
    def is_lagrangian(self) -> bool:
        return self.array[:, 1:].sum() == 0

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
                left, right = [IncidenceMatrix(a) for a in arrays]
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
        return IncidenceMatrix(up), IncidenceMatrix(down)

    def __repr__(self):
        return self.array.__repr__()
