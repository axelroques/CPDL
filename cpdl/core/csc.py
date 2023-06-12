
import numpy as np


class CSC:
    """
    Solver for the convolutional sparse coding problem.
    """

    def __init__(self, N, K, L, regularization, X, D, Z, method='ISTA') -> None:

        # Basic parameters
        self.N = N
        self.K = K
        self.L = L
        self.regularization = regularization
        self.X = X
        self.D = D
        self.Z = Z
        self.method = method

        # Initialize step size
        self.step_size = None

        # Method-specific parameters
        if self.method == 'FISTA':
            # Initialize Y_0 as Z_0
            self.Y_t = Z.getToeplitz()
            # Initialize T_0 with ones
            self.T_t = np.zeros(len(self.Y_t)) + 1
        elif self.method == 'ISTA':
            pass
        else:
            raise RuntimeError("Unkown method, expected 'ISTA' or 'FISTA'.")

    def step(self):
        """
        Perform 1 iteration of the CSC optimization
        step.
        """

        if self.method == 'ISTA':
            return self._stepISTA()
        elif self.method == 'FISTA':
            return self._stepFISTA()
        else:
            RuntimeError('How did you get here?')

    def _stepISTA(self):
        """
        Perform 1 iteration of the CSC, with the 
        desired method ('ISTA' or 'FISTA).
        """

        # Get Toeplitz matrices
        T_D = self.D.getToeplitzFormalismCSC()
        T_Z = self.Z.getToeplitzFormalismCSC()

        # Compute prox operator
        T_Z_t = self._proximalOperator(T_D, T_Z)

        # Update activations
        Z_t = np.reshape(T_Z_t, (self.K, -1))
        self.Z.updateActivations(Z_t)
        # print('Z After =', Z_t.shape, Z_t[0], '\n')

        return

    def _stepFISTA(self):
        """
        Perform 1 iteration of the FISTA algorithm.
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

        return

    def _lineSearch(self):
        """
        Backtracking stepsize rule to avoid the costly
        computation of the Lipschitz constant.
        """
        return

    def _proximalOperator(self, T_D, T_Z):
        """
        Proximal operator according to Beck et. al. (2009),
        using the Toeplitz representation of D and Z.
        """

        # Compute step size
        if self.step_size == None:
            self.step_size = 1 / np.max(
                np.linalg.eigvals(np.dot(T_D.T, T_D))
            ).real
            # self.step_size = 0.029154280210046048
        # print('step size =', self.step_size)

        # Compute the data fidelity term
        data_fidelity = np.dot(T_D, T_Z) - self.X
        # print('data_fidelity =', data_fidelity.shape, data_fidelity[0])

        # Compute the gradient
        gradient = 2*np.dot(T_D.T, data_fidelity)
        # print('gradient =', gradient.shape, gradient[0])

        return self._shrinkageOperator(
            T_Z - self.step_size*gradient,
            self.regularization*self.step_size
        )

    # def _proximalOperatorFreq(self, T_D, T_Z):
    #     """
    #     Proximal operator according to Wohlberg (2016).
    #     Computes the gradient in the frequency domain.
    #     """

    #     # FFT of X, T_D and T_Z
    #     X_hat = np.fft.fft(self.X)  # To move in init later
    #     T_D_hat = np.fft.fft2(T_D)
    #     T_Z_hat = np.fft.fft(T_Z)
    #     print('X_hat =', X_hat.shape,
    #           'T_D_hat =', T_D_hat.shape,
    #           'T_Z_hat =', T_Z_hat.shape)

    #     # Compute the data fidelity term
    #     data_fidelity = np.dot(T_D_hat, T_Z_hat) - X_hat
    #     print('data_fidelity =', data_fidelity.shape, data_fidelity[0])

    #     # Compute the gradient
    #     gradient = 2*np.dot(T_D_hat.conj().T, data_fidelity)
    #     print('gradient =', gradient.shape, gradient[0])

    #     # Transform back to the spatial domain
    #     to_shrink = np.fft.ifft(T_Z_hat - gradient/self.step_size).real
    #     print('to_shrink =', to_shrink[0])

    #     return self._shrinkageOperator(to_shrink,
    #                                    self.regularization/self.step_size)

    # def _proximalOperatorConv(self, X, D):
    #     """
    #     Proximal operator according to Chalasani et. al. (2013).
    #     """
    #     # Compute gradient
    #     D_rotated = np.rot90(np.rot90(D))
    #     loss = X - np.sum([np.convolve(d, y, 'same')
    #                        for d, y in zip(D.T, self.Y_t)],
    #                       axis=0)
    #     print(loss.shape)

    #     gradient = []
    #     for atom in D_rotated.T:
    #         gradient.append(np.convolve(atom, loss, 'same'))
    #     gradient = np.stack(gradient)

    #     # print(gradient.shape)

    #     return self._shrinkageOperator(self.Y_t - gradient/self.step_size,
    #                                    self.regularization/self.step_size)

    @staticmethod
    def _shrinkageOperator(x, alpha):
        """
        Shrinkage operator T.
        """
        # print('to shrink =', x[0])
        # test = np.sign(x) * np.maximum(np.abs(x) - alpha, 0)
        # print('shrink factor =', alpha)
        # print('after shrink =', test[0])
        return np.sign(x) * np.maximum(np.abs(x) - alpha, 0)
