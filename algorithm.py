import math
import numpy as np

from incidence_matrix import IncidenceMatrix
from knots import Knots
from polynomial import Polynomial


def solve_lagrangian(x, w, incidence_matrix):
    """
    :param x: (np.ndarray, shape=(k,))
    :param w: (Union[np.ndarray, Knots], shape=incidence_matrix.array.shape or |incidence_matrix|) can either be long
        form (the w_ij ordered as they match the 1s in the incidence matrix rows first), square form (with the same
        shape as incidence_matrix.array and having floats where there are 1s in incidence matrix and np.nan otherwise)
        or as an already parsed Knots object.
    :param incidence_matrix: (Union[IncidenceMatrix, np.ndarray, list])
    :return: (Polynomial)
    """
    assert len(x.shape) == 1
    assert incidence_matrix.array[:, 1:].sum() == 0
    x = x[incidence_matrix.array[:, 0].astype(bool)]
    if isinstance(w, np.ndarray):
        w = Knots(w, incidence_matrix)

    denominator_matrix = np.array([x ** i for i in range(len(x) - 1, -1, -1)])
    denominator = np.linalg.det(denominator_matrix)

    numerator_matrix = np.vstack([w.long, denominator_matrix])
    coefficients = - np.array([(-1) ** (i + 2) * np.linalg.det(np.delete(numerator_matrix, obj=i, axis=0))
                               for i in range(len(x), 0, -1)])
    coefficients /= denominator

    return Polynomial(coefficients)


def solve(x, w, incidence_matrix) -> Polynomial:
    """
    :param x: (np.ndarray, shape=(k,))
    :param w: (Union[np.ndarray, Knots], shape=incidence_matrix.array.shape or |incidence_matrix|) can either be long
        form (the w_ij ordered as they match the 1s in the incidence matrix rows first), square form (with the same
        shape as incidence_matrix.array and having floats where there are 1s in incidence matrix and np.nan otherwise)
        or as an already parsed Knots object.
    :param incidence_matrix: (Union[IncidenceMatrix, np.ndarray, list])
    :return: (Polynomial)
    """
    if isinstance(incidence_matrix, (np.ndarray, list)):
        incidence_matrix = IncidenceMatrix(incidence_matrix)
    if isinstance(w, (np.ndarray, list)):
        w = Knots(w, incidence_matrix)
    if isinstance(x, list):
        x = np.array(x)
    if not incidence_matrix.polya_condition:
        print("Warning: incidence_matrix does not satisfy Polya condition and it is not balanced. The problem "
              "migth not have a solution for the specified x and w.")
    assert len(x) == incidence_matrix.k

    if incidence_matrix.is_lagrangian:
        return solve_lagrangian(x, w, incidence_matrix)

    if incidence_matrix.is_irreducible:
        up, down = incidence_matrix.reduce()

        up_knots = Knots(w[:, :-1], up)
        up_solution = solve(x, up_knots, up)

        down_knots = Knots(w[:, :-1], down)
        down_solution = solve(x, down_knots, down)

        x_pow_n_derivatives = np.array([math.factorial(incidence_matrix.n) / math.factorial(incidence_matrix.n - j)
                                        * x ** (incidence_matrix.n - j)
                                        for j in range(incidence_matrix.n)]).T

        derivatives_knots_up = Knots(x_pow_n_derivatives, up)
        up_n_solution = solve(x, derivatives_knots_up, up)

        derivatives_knots_down = Knots(x_pow_n_derivatives, down)
        down_n_solution = solve(x, derivatives_knots_down, down)

        c = (up_solution[1] - down_solution[1]) / (up_n_solution[1] - down_n_solution[1])

        return up_solution - (incidence_matrix.p_n - up_n_solution) * c
    else:
        left, right = incidence_matrix.decompose()

        right_g_solution = solve(x, Knots(w[:, left.n + 1:], right), right)
        right_solution = right_g_solution.integrate(left.n + 1)

        left_w = Knots(w[:, :left.n + 1] -
                       np.array([right_solution.differentiate(i)(x) for i in range(left.n + 1)]).T,
                       left)
        left_solution = solve(x, left_w, left)
        return left_solution + right_solution
