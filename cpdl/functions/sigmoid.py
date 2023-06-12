
import numpy as np


def sigmoid(t, E_0, E_max, t_50, alpha):
    """
    Sigmoid function according to Hill's equation.
    """

    # Initialize array
    s = np.zeros_like(t, dtype=np.float64)

    # Avoid dividing by zero
    t_50 = max(0.01, t_50)

    # Fill array
    s[:] = E_0 + (E_max-E_0)*np.power(t, alpha, dtype=np.float64) / \
        (np.power(t_50, alpha, dtype=np.float64) +
         np.power(t, alpha, dtype=np.float64))

    return s


def sigmoid2(t, t_50, alpha):
    """
    Sigmoid function according to Hill's equation.
    """

    # Initialize array
    s = np.zeros_like(t, dtype=np.float64)

    # Avoid dividing by zero
    t_50 = max(0.001, t_50)

    # Fill array
    s[:] = np.power(t, alpha, dtype=np.float64) / \
        (np.power(t_50, alpha, dtype=np.float64) +
         np.power(t, alpha, dtype=np.float64))

    return s


def sigmoidDerivatives(t, E_0, E_max, t_50, alpha, p):
    """
    Partial derivatives of the sigmoid function according
    to the p^th parameter.
    """

    # Avoid dividing by zero
    t_50 = max(t[1], t_50)

    # df/d(E_0)
    if p == 0:
        return 1 - np.power(t, alpha)/(np.power(t_50, alpha) + np.power(t, alpha))

    # df/d(E_max)
    elif p == 1:
        return np.power(t, alpha)/(np.power(t_50, alpha) + np.power(t, alpha))

    # df/d(t_50)
    elif p == 2:
        # return -(E_max-E_0)*alpha*np.power(t, alpha)*np.power(t_50, alpha-1) / \
        #     np.power(np.power(t_50, alpha) + np.power(t, alpha), 2)
        return np.zeros(len(t))

    # df/d(alpha)
    elif p == 3:
        return (E_max-E_0)*np.power(t_50, alpha)*np.power(t, alpha) * \
            (np.log2(t, where=(t != 0)) - np.log2(t_50)) / \
            np.power(np.power(t_50, alpha) + np.power(t, alpha), 2)

    else:
        raise RuntimeError('Unexpected parameter for partial derivative.')


def sigmoidDerivatives2(t, t_50, alpha, p):
    """
    Partial derivatives of the sigmoid function according
    to the p^th parameter.
    """

    # Avoid dividing by zero
    t_50 = max(0.001, t_50)

    # df/d(t_50)
    if p == 0:
        return -alpha*np.power(t, alpha)*np.power(t_50, alpha-1) / \
            np.power(np.power(t_50, alpha) + np.power(t, alpha), 2)

    # df/d(alpha)
    elif p == 1:
        return np.power(t_50, alpha)*np.power(t, alpha) * \
            (np.log2(t, where=(t != 0)) - np.log2(t_50)) / \
            np.power(np.power(t_50, alpha) + np.power(t, alpha), 2)

    else:
        raise RuntimeError('Unexpected parameter for partial derivative.')
