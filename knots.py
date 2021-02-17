import numpy as np


class Knots:
    """
    Holds the RHS conditions w_ij in both long and square form.
    Long - an array of the w_ij ordered by their position in the IncidenceMatrix iterated row first.
    Square - an matrix holding the w_ij on the positions of the 1s in the incidence matrix and np.nan otherwise.
    """
    def __init__(self, array, incidence_matrix: "IncidenceMatrix"):
        self.incidence = incidence_matrix

        if isinstance(array, list):
            array = np.array(array)
        if len(array.shape) == 1:
            self.long = array
            self.square = self._square_form()
        else:
            self.square = self._mask(array, incidence_matrix)
            self.long = self._long_form()

    @staticmethod
    def _mask(knots, incidence_matrix):
        """
        Replaces some of the values in the squre format with np.nan to support removing knots.
        :param knots: (np.ndarray)
        :param incidence_matrix: (IncidenceMatrix)
        :return: (np.ndarray)
        """
        assert knots.shape[0] >= incidence_matrix.k
        assert knots.shape[1] > incidence_matrix.n
        square = []
        for i in range(incidence_matrix.k):
            row = []
            for j in range(incidence_matrix.n + 1):
                if incidence_matrix.array[i, j]:
                    row.append(knots[i, j])
                else:
                    row.append(np.nan)
            square.append(row)
        return np.array(square)

    def _square_form(self):
        w = list(self.long[::-1])
        square = []
        for i in range(self.incidence.k):
            row = []
            for j in range(self.incidence.n + 1):
                if self.incidence.array[i, j]:
                    row.append(w.pop())
                else:
                    row.append(np.nan)
            square.append(row)
        return np.array(square)

    def _long_form(self):
        return self.square[~np.isnan(self.square)]

    def __getitem__(self, item):
        return self.square.__getitem__(item)