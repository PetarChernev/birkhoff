import math
from functools import cached_property
from typing import Tuple

import numpy as np

from polynomial import Polynomial


class IncidenceMatrix:
    """
    Holds an incidence matrix in the array variable and defines properties of the incidence matrix.
    Implements the decompose and reduce methods to split the IncidenceMatrix into smaller ones.
    """
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
        """
        Decomposes the matrix based on Atkinson-Sharma's Theorem of decomposition into one irreducible IncidenceMatrix
        and one the satisfies Polya's condition.
        :return: (Tuple["IncidenceMatrix", "IncidenceMatrix"])
            (irreducible IncidenceMatrix, Polya condition IncidenceMatrix)
        """
        if self.is_irreducible:
            raise ValueError("Cannot decompose irreducible incidence matrix.")
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
        """
        Reduces an irreducible matrix as defined by Muhlbach by:
            1. removing the last row
            2. removing respectively the last one in the first row that has a one
                and the last one in the last row that has a one.
        :return: (Tuple["IncidenceMatrix", "IncidenceMatrix"])
            (last one in the first row with a one remove, last one in the last row with a one removed)
        """
        up = np.copy(self.array[:, :-1])
        first_non_zero_row = np.nonzero(up.sum(axis=1))[0][0]
        last_one_in_row = np.nonzero(up[first_non_zero_row, :])[0][-1]
        up[first_non_zero_row, last_one_in_row] = 0

        down = np.copy(self.array[:, :-1])
        last_non_zero_row = np.nonzero(down.sum(axis=1))[0][-1]
        last_one_in_row = np.nonzero(down[last_non_zero_row, :])[0][-1]
        down[last_non_zero_row, last_one_in_row] = 0
        return IncidenceMatrix(up), IncidenceMatrix(down)

    def __repr__(self):
        return self.array.__repr__()
