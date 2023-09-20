
import numpy as np
from random import uniform


class Logistic:

    def __init__(self, L) -> None:

        # Construct arbitrary time vector
        self.t = np.arange(L, dtype=np.float64)
        self.t /= len(self.t)

    def init(self):
        """
        Initialize parameters.
        """

        E_0 = uniform(-53, 53)
        E_max = uniform(-53, 53)
        alpha = uniform(10, 50)

        return np.array([E_0, E_max, alpha], dtype=np.float64)

    def get(self, E_0, E_max, alpha):
        """
        Logistic function.
        """
        return E_0 + (E_max-E_0) / (1 + np.exp(-alpha*(self.t-0.5), dtype=np.float64))

    def derivative(self, E_0, E_max, alpha, p):
        """
        Partial derivative of the logistic function w.r.t.
        the p^th parameter.
        """

        # Shortcut
        t = self.t

        # df/d(E_0)
        if p == 0:
            return 1 / (1 + np.exp(alpha*(t-0.5), dtype=np.float64))

        # df/d(E_max)
        elif p == 1:
            return 1 / (1 + np.exp(-alpha*(t-0.5), dtype=np.float64))

        # df/d(alpha)
        elif p == 2:
            numerator = (E_max-E_0) * (2*t-1) * \
                np.exp(alpha*(t-0.5), dtype=np.float64)
            denominator = 2 * \
                np.power(
                    np.exp(alpha*(t-0.5), dtype=np.float64) + 1, 2,
                    dtype=np.float64
                )
            return numerator / denominator

        else:
            raise RuntimeError('Unexpected parameter for partial derivative.')
