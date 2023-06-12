
import numpy as np


def gompertz(t, E_0, E_max, t_50, c):
    """
    Gompertz function.

    b is found such that the halfway point falls
    at the middle of the time vector.

    OR 

    b is found such that the point of maximum rate of 
    increase falls at the middle of the time vector.
    """

    # b = np.exp(t[t_50]*c + np.log2(np.log2(2)))
    b = np.exp(c*t[t_50])

    return E_0 + (E_max-E_0) * np.exp(-b * np.exp(-c*t))


def gompertzDerivatives(t, a, b, c, p):
    """
    Partial derivatives of the Gompertz function according
    to the p^th parameter.
    """

    # df/d(a)
    if p == 0:
        return np.exp(-b * np.exp(-c*t))

    # df/d(b)
    elif p == 1:
        return -a * np.exp(-b * np.exp(-c*t) - c*t)

    # df/d(c)
    elif p == 2:
        return a*b*t * np.exp(-b * np.exp(-c*t) - c*t)

    else:
        raise RuntimeError('Unexpected parameter for partial derivative.')
 