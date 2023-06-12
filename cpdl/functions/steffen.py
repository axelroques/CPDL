
import numpy as np


def steffen(t, P1, P2, P3, P4):
    """
    Steffen's interpolation .
    Cubic monotone interpolation method.
    """

    test = []
    interp = []

    points = [P1, P1, P2, P3, P4, P4]
    for i_points in range(3):

        sub_points = points[i_points:i_points+4]

        # Temporal parameters
        t_i_minus_1 = sub_points[0][0]
        t_i = sub_points[1][0]
        t_i_plus_1 = sub_points[2][0]
        t_i_plus_2 = sub_points[3][0]

        # Amplitude parameters
        y_i_minus_1 = sub_points[0][1]
        y_i = sub_points[1][1]
        y_i_plus_1 = sub_points[2][1]
        y_i_plus_2 = sub_points[3][1]

        # t_ = t[
        #     (t >= sub_points[1][0]) & (t < sub_points[-2][0])
        # ]
        t_ = np.linspace(t_i, t_i_plus_1, 50, endpoint=False)
        # print('t_ =', t_)
        test.append(t_)

        interp_ = interpolate(
            t_, t_i_minus_1, t_i, t_i_plus_1, t_i_plus_2,
            y_i_minus_1, y_i, y_i_plus_1, y_i_plus_2
        )

        interp.append(interp_)

    return np.concatenate(test, axis=None), np.concatenate(interp, axis=None)


def cubic_interpolation_function(t, t_i, a_i, b_i, c_i, d_i):
    return a_i*(t-t_i)**3 + b_i*(t-t_i)**2 + c_i*(t-t_i) + d_i


def h(t_i, t_i_plus_1):
    return t_i_plus_1 - t_i


def s(t_i, t_i_plus_1, y_i, y_i_plus_1):
    try:
        return (y_i_plus_1-y_i) / (t_i_plus_1-t_i)
    except ZeroDivisionError:
        return 0


def slope(s_i_minus_1, s_i, h_i_minus_1, h_i):

    if s_i_minus_1*s_i <= 0:
        return 0

    else:
        p_i = (s_i_minus_1*h_i + s_i*h_i_minus_1) / (h_i+h_i_minus_1)

        if (abs(p_i) > 2*abs(s_i_minus_1)) or \
                (abs(p_i) > 2*abs(s_i)):
            return 2*np.sign(s_i)*abs(min([s_i_minus_1, s_i], key=abs))

        else:
            return p_i


def a(y_p_i, y_p_i_plus_1, s_i, h_i):
    return (y_p_i + y_p_i_plus_1 - 2*s_i) / h_i**2


def b(y_p_i, y_p_i_plus_1, s_i, h_i):
    return (3*s_i - 2*y_p_i - y_p_i_plus_1) / h_i


def c(y_p_i):
    return y_p_i


def d(y_i):
    return y_i


def interpolate(
        t, t_i_minus_1, t_i, t_i_plus_1, t_i_plus_2,
        y_i_minus_1, y_i, y_i_plus_1, y_i_plus_2
):

    # Basic computations
    h_i_minus_1 = h(t_i_minus_1, t_i)
    h_i = h(t_i, t_i_plus_1)
    h_i_plus_1 = h(t_i_plus_1, t_i_plus_2)
    s_i_minus_1 = s(t_i_minus_1, t_i, y_i_minus_1, y_i)
    s_i = s(t_i, t_i_plus_1, y_i, y_i_plus_1)
    s_i_plus_1 = s(t_i_plus_1, t_i_plus_2, y_i_plus_1, y_i_plus_2)

    # Derivatives
    y_p_i = slope(s_i_minus_1, s_i, h_i_minus_1, h_i)
    y_p_i_plus_1 = slope(s_i, s_i_plus_1, h_i, h_i_plus_1)

    # Coefficients
    a_i = a(y_p_i, y_p_i_plus_1, s_i, h_i)
    b_i = b(y_p_i, y_p_i_plus_1, s_i, h_i)
    c_i = c(y_p_i)
    d_i = d(y_i)

    return cubic_interpolation_function(t, t_i, a_i, b_i, c_i, d_i)
