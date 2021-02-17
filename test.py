import unittest

import numpy as np
from algorithm import solve
from incidence_matrix import IncidenceMatrix
from knots import Knots


class TestAlgorithm(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.arrays = [
            np.array([[1, 0, 0],
                      [0, 0, 0],
                      [1, 0, 0],
                      [0, 0, 1]]),
            np.array([[1, 0, 0, 0, 0],
                      [0, 0, 1, 1, 0],
                      [1, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0]]),
            np.array([[1, 0, 0, 0],
                      [0, 1, 1, 0],
                      [1, 0, 0, 0]])
        ]

        cls.test_cases_x = 10
        cls.test_cases_w = 10

    def test_algorithm(self):
        for array in self.arrays:
            for i in range(self.test_cases_x):
                for j in range(self.test_cases_w):
                    x = np.sort(np.random.random(array.shape[0]))
                    w = np.random.random(array.sum())
                    print(f"Testing case ({i+1}, {j+1}) for incidence matrix \n{array}\n with x = {x}, w={w}")
                    self.run_single_tst(array, x, w)

    def run_single_tst(self, array, x, w):
        matrix = IncidenceMatrix(array)
        knots = Knots(w, matrix)
        p = solve(x, w, array)
        for j in range(matrix.n + 1):
            p_j = p.differentiate(j)
            results = p_j(x)
            for i in range(matrix.k):
                if np.isnan(knots.square[i,j]):
                    continue
                self.assertAlmostEqual(knots.square[i, j], results[i], 10)


if __name__ == '__main__':
    unittest.main()