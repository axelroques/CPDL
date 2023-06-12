
import numpy as np


def catmull_rom(P1, P2, P3, P4, tau):
    """
    2D Catmull-Rom spline.
    tau defines the tension on the curve.

    Matrix operations based on https://lucidar.me/fr/mathematics/catmull-rom-splines/
    """

    coefs = np.array([
        [0, 1, 0, 0],
        [-tau, 0, tau, 0],
        [2*tau, tau-3, 3-2*tau, -tau],
        [-tau, 2-tau, tau-2, tau]
    ])

    CR = []

    points = [P1, P1, P2, P3, P4, P4]
    for i_points in range(3):

        sub_points = points[i_points:i_points+4]
        # t_ = np.arange(sub_points[-2][0] - sub_points[1][0])
        t_ = np.arange(100)
        t_ = t_ / len(t_)

        CR_ = np.zeros((2, len(t_)))
        for i_t, t in enumerate(t_):

            T = np.array([1, t, t**2, t**3])

            for i_dim in range(CR_.shape[0]):

                P = np.array([
                    sub_points[0][i_dim],
                    sub_points[1][i_dim],
                    sub_points[2][i_dim],
                    sub_points[3][i_dim]
                ]).T

                CR_[i_dim, i_t] = T @ coefs @ P

        CR.append(CR_)

    return np.concatenate(CR, axis=1)
