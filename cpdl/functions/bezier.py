
import numpy as np


def bezier(t_, P1, P2, P3, P4):
    """
    2D Bézier curve.

    Matrix operations based on https://pomax.github.io/bezierinfo/#matrix
    """

    coefs = np.array([
        [1, 0, 0, 0],
        [-3, 3, 0, 0],
        [3, -6, 3, 0],
        [-1, 3, -3, 1]
    ])

    B = np.zeros((2, len(t_)))
    for i_t, t in enumerate(t_):

        T = np.array([1, t, t**2, t**3])

        for i_dim in range(B.shape[0]):

            P = np.array([P1[i_dim], P2[i_dim], P3[i_dim], P4[i_dim]]).T

            B[i_dim, i_t] = T @ coefs @ P

    return B


def rational_bezier(t_, P1, P2, P3, P4):
    """
    2D rational Bézier curve.

    Matrix operations based on https://pomax.github.io/bezierinfo/#matrix
    """

    ratios = np.array([1, 20, 20, 1]).T

    coefs = np.array([
        [1, 0, 0, 0],
        [-3, 3, 0, 0],
        [3, -6, 3, 0],
        [-1, 3, -3, 1]
    ])

    B = np.zeros((2, len(t_)))
    for i_t, t in enumerate(t_):

        T = np.array([1, t, t**2, t**3])

        basis = T @ coefs @ ratios

        for i_dim in range(B.shape[0]):

            P = np.array([P1[i_dim], P2[i_dim], P3[i_dim], P4[i_dim]]).T

            B[i_dim, i_t] = T @ coefs @ (P * ratios) / basis

    return B
