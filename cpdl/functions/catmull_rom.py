
import numpy as np
from random import uniform


def initCatmullRom(**kwargs):
    """
    Initialize spline parameters.
    """

    t = kwargs['t']

    E_0 = uniform(-53, 53)
    E_max = uniform(-53, 53)
    d_50 = uniform(2*(t[1]-t[0]), 5*(t[1]-t[0]))
    print('d_50 =', d_50)

    return np.array([E_0, E_max, d_50], dtype=np.float64)


def catmull_rom(t, E_0, E_max, d_50):
    """
    2D Catmull-Rom spline.
    tau defines the tension on the curve.

    Matrix operations based on 
    https://lucidar.me/fr/mathematics/catmull-rom-splines/
    """

    # Spline parameters
    tau = 0.2
    amplitude = np.abs(E_max-E_0)
    sign = np.sign(E_max - E_0)
    P1 = (t[0], E_0)
    P4 = (t[-1], E_max)
    P2 = (t[len(t)//2] - d_50, E_0 + sign*1/20*amplitude)
    P3 = (t[len(t)//2] + d_50, E_max - sign*1/20*amplitude)

    # print('\nt =', t)
    # print('\nE_0 =', E_0)
    # print('E_max =', E_max)
    # print('d_50 =', d_50)
    # print('P1 =', P1)
    # print('P2 =', P2)
    # print('P3 =', P3)
    # print('P4 =', P4)

    coefs = np.array([
        [0, 1, 0, 0],
        [-tau, 0, tau, 0],
        [2*tau, tau-3, 3-2*tau, -tau],
        [-tau, 2-tau, tau-2, tau]
    ])

    CR = []

    n_total = 0
    points = [P1, P1, P2, P3, P4, P4]
    for i_points in range(3):

        # Compute number of points in each segment
        sub_points = points[i_points:i_points+4]
        rel_duration = np.around(
            (sub_points[-2][0]-sub_points[1][0]) / (t[-1]-t[0]), 2
        )
        # print('\nlen(t) =', len(t))
        # print('rel_duration =', rel_duration)
        # print('n_point =', np.around(len(t)*rel_duration, 0))
        n_point = np.around(len(t)*rel_duration, 0).astype(int)
        n_total += n_point
        if i_points == 2 and n_total < len(t):
            n_point += 1
        if i_points == 2 and n_total > len(t):
            n_point -= 1
        local_t = np.linspace(0, 1, num=n_point, endpoint=False)

        CR_ = np.zeros((2, len(local_t)))
        for i_t, t_val in enumerate(local_t):

            T = np.array([1, t_val, t_val**2, t_val**3])

            # This loop is not really necessary as we only care about
            # the y dimension of the curve
            for i_dim in range(CR_.shape[0]):

                P = np.array([
                    sub_points[0][i_dim],
                    sub_points[1][i_dim],
                    sub_points[2][i_dim],
                    sub_points[3][i_dim]
                ]).T

                CR_[i_dim, i_t] = T @ coefs @ P

        CR.append(CR_)

    return np.concatenate(CR, axis=1)[1, :]


def catmullRomDerivatives(t, E_0, E_max, d_50, p):
    """
    Partial derivatives of the Catmull-Rom splines 
    w.r.t. the p^th parameter.

    Derivatives formula were found here 
    https://lucidar.me/en/mathematics/catmull-rom-splines/

    ** DOES NOT WORK **
    """

    # Spline parameters
    tau = 0.2
    amplitude = np.abs(E_max-E_0)
    sign = np.sign(E_max - E_0)
    P1 = (t[0], E_0)
    P4 = (t[-1], E_max)
    P2 = (t[len(t)//2] - d_50, E_0 + sign*1/20*amplitude)
    P3 = (t[len(t)//2] + d_50, E_max - sign*1/20*amplitude)

    coefs = np.array([
        [0, 1, 0, 0],
        [-tau, 0, tau, 0],
        [2*tau, tau-3, 3-2*tau, -tau],
        [-tau, 2-tau, tau-2, tau]
    ])

    CR = []

    # df/d(E_0)
    if p == 0:

        n_total = 0
        points = [P1, P1, P2, P3, P4, P4]
        for i_points in range(3):

            # Compute number of points in each segment
            sub_points = points[i_points:i_points+4]
            rel_duration = np.around(
                (sub_points[-2][0]-sub_points[1][0]) / (t[-1]-t[0]), 2
            )
            n_point = np.around(len(t)*rel_duration, 0).astype(int)
            n_total += n_point
            if i_points == 2 and n_total < len(t):
                n_point += 1
            if i_points == 2 and n_total > len(t):
                n_point -= 1
            local_t = np.linspace(0, 1, num=n_point, endpoint=False)

            CR_ = np.zeros((1, len(local_t)))
            for i_t, t_val in enumerate(local_t):

                T = np.array([0, 1, 2*t_val, 3*t_val**2])

                if i_points == 0:
                    P = np.array([
                        1,
                        1,
                        1-sign/20,
                        -sign/20
                    ]).T

                elif i_points == 1:
                    P = np.array([
                        1,
                        1-sign/20,
                        -sign/20,
                        0
                    ]).T

                elif i_points == 2:
                    P = np.array([
                        1-sign/20,
                        -sign/20,
                        0,
                        0
                    ]).T

                else:
                    pass

                CR_[:, i_t] = T @ coefs @ P

            CR.append(CR_)

        return np.concatenate(CR, axis=None)

    # df/d(E_max)
    elif p == 1:

        n_total = 0
        points = [(P1[0], 0), (P1[0], 0), (P2[0], 0), (P3[0], 0), P4, P4]
        for i_points in range(3):

            # Compute number of points in each segment
            sub_points = points[i_points:i_points+4]
            rel_duration = np.around(
                (sub_points[-2][0]-sub_points[1][0]) / (t[-1]-t[0]), 2
            )
            n_point = np.around(len(t)*rel_duration, 0).astype(int)
            n_total += n_point
            if i_points == 2 and n_total < len(t):
                n_point += 1
            if i_points == 2 and n_total > len(t):
                n_point -= 1
            local_t = np.linspace(0, 1, num=n_point, endpoint=False)

            CR_ = np.zeros((1, len(local_t)))
            for i_t, t_val in enumerate(local_t):

                T = np.array([0, 1, 2*t_val, 3*t_val**2])

                if i_points == 0:
                    P = np.array([
                        0,
                        0,
                        sign/20,
                        1+sign/20
                    ]).T

                elif i_points == 1:
                    P = np.array([
                        0,
                        sign/20,
                        1+sign/20,
                        1
                    ]).T

                elif i_points == 2:
                    P = np.array([
                        sign/20,
                        1+sign/20,
                        1,
                        1
                    ]).T

                CR_[:, i_t] = T @ coefs @ P

            CR.append(CR_)

        return np.concatenate(CR, axis=None)

    # df/d(d_50)
    elif p == 2:

        return np.zeros(len(t))
        n_total = 0
        points = [(P1[0], 0), (P1[0], 0), P2, P3, (P4[0], 0), (P4[0], 0)]
        for i_points in range(3):

            # Compute number of points in each segment
            sub_points = points[i_points:i_points+4]
            rel_duration = np.around(
                (sub_points[-2][0]-sub_points[1][0]) / (t[-1]-t[0]), 2
            )
            n_point = np.around(len(t)*rel_duration, 0).astype(int)
            n_total += n_point
            if i_points == 2 and n_total < len(t):
                n_point += 1
            if i_points == 2 and n_total > len(t):
                n_point -= 1
            local_t = np.linspace(0, 1, num=n_point, endpoint=False)

            CR_ = np.zeros((1, len(local_t)))
            for i_t, t_val in enumerate(local_t):

                T = np.array([0, 1, 2*t_val, 3*t_val**2])

                if i_points == 0:
                    P = np.array([
                        0,
                        0,
                        -1,
                        1
                    ]).T

                elif i_points == 1:
                    P = np.array([
                        0,
                        -1,
                        1,
                        0
                    ]).T

                elif i_points == 2:
                    P = np.array([
                        -1,
                        1,
                        0,
                        0
                    ]).T

                CR_[:, i_t] = T @ coefs @ P

            CR.append(CR_)

        return np.concatenate(CR, axis=None)

    else:
        raise RuntimeError('Unexpected parameter for partial derivative.')


def _catmull_rom(P1, P2, P3, P4, tau):
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
        t_ = np.arange(sub_points[-2][0] - sub_points[1][0])
        # t_ = np.arange(100)
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
