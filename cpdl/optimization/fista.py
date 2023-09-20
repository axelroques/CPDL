

from .solver import Solver

import numpy as np


class FISTA(Solver):

    def __init__(self, kwargs) -> None:
        super().__init__(kwargs)

        # Initialize Y_0 as Z_0
        self.Y_t = kwargs['Z'].getToeplitz()
        # Initialize T_0 with ones
        self.T_t = np.zeros(len(self.Y_t)) + 1

    def step(self):
        """
        Perform 1 iteration of the method.
        """

        # Get Toeplitz matrices
        T_D = self.D.getToeplitzFormalismCSC()
        T_Z = self.Z.getToeplitzFormalismCSC()

        # Compute prox operator
        T_Z_t = self._proximalOperator(T_D, self.Y_t)

        # Update activations
        Z_t_minus_1 = self.Z.getActivations()
        Z_t = np.reshape(T_Z_t, (self.K, -1))
        self.Z.updateActivations(Z_t)
        # print('Z Before =', Z_t_minus_1.shape, Z_t_minus_1[0])
        # print('Z After =', Z_t.shape, Z_t[0], '\n')

        # Update T_t
        T_t_minus_1 = self.T_t
        self.T_t = 1/2 * (1 + np.sqrt(1+4*np.power(self.T_t, 2)))

        # Update Y_t
        self.Y_t = T_Z_t + ((T_t_minus_1-1)/self.T_t) * (T_Z_t-T_Z)
