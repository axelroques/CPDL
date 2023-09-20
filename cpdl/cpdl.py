
from .core import ActivationMatrix
from .core import Dictionary
from .core import CSC
from .core import CDL

from .utils import convolution

import numpy as np


class CPDL:

    def __init__(
            self,
            X,
            function='logistic',
            K=2,
            L=30,
            regularization=1,
            CSC_solver='ISTA',
            CDL_solver='SD'
    ) -> None:

        # Original signal
        self.X = self._verifyInput(X)

        # Signal dimensions
        self.N, self.P, self.T = X.shape

        # Atom function
        self.function = function

        # Number of atoms
        self.K = K

        # Length of the atoms
        self.L = L

        # Regularisation (lambda)
        self.regularization = regularization

        # Number of iterations
        self.n_iter = 0

        # Parametric dictionary
        self.D = self._generateDictionary()

        # Activation matrix
        self.Z = self._generateActivationMatrix()

        # Initialization
        self.csc = CSC(
            N=self.N, P=self.P, T=self.T, K=self.K, L=self.L,
            regularization=self.regularization,
            X=self.X, D=self.D, Z=self.Z,
            method=CSC_solver
        )
        self.cdl = CDL(
            N=self.N, P=self.P, T=self.T, K=self.K, L=self.L,
            regularization=self.regularization,
            X=self.X, D=self.D, Z=self.Z,
            method=CDL_solver
        )

    def optimize(self, n_iter):
        """
        Main function.
        """

        for t in range(self.n_iter, self.n_iter+n_iter):
            self.csc.step()
            self.cdl.step()
            # self._log(t)

        self.n_iter += n_iter

        return

    def CSC_step(self):
        """
        CSC step.
        """
        return self.csc.step()

    def CDL_step(self):
        """
        CDL step.
        """
        return self.cdl.step()

    @staticmethod
    def _verifyInput(X):
        """
        Input X should be an array of size (N x P x T), where:
            - N is the number of multivariate time series,
            - P is the dimension of the time series,
            - T is the number of observations in the time series.
        """
        X = np.array(X)
        if len(X.shape) != 3:
            raise RuntimeError('X should be of dimension (N x P x T).')
        return X

    def _normalizeSignal(self):
        """
        Normalize input X signal.
        """
        return (self.raw_X - self.raw_X.mean()) / self.raw_X.std()

    def _generateDictionary(self):
        """
        Generate a dictionary object.
        """
        return Dictionary(self.T, self.P, self.function, self.K, self.L)

    def _generateActivationMatrix(self):
        """
        Generate an activation vector object.
        """
        return ActivationMatrix(self.N, self.T, self.K, self.L)

    def getDictionary(self):
        """
        Helper function to get the dictionary.
        """
        return self.D.getDictionary()

    def getActivationMatrix(self):
        """
        Helper function to get the activation matrix.
        """
        return self.Z.getActivations()

    def _log(self, t):
        """
        Print useful values after each iteration.
        """

        # Initialize verbose
        if t == 0:
            print('Iteration \t | \t Residuals \t ')
            print('------------------------------------')

        D = self.getDictionary()

        # Compute residuals (sum over all time series)
        residuals = 0
        for n in range(self.N):
            Z = self.Z.getActivations(n)

            conv = convolution(Z, D)
            residuals += np.linalg.norm(self.X[n] - conv)**2

        print(f'{t} \t\t | \t {residuals:.4f}')

        return

    def reconstruct(self):
        """
        Reconstruct the signal from the atoms and their activations.
        """
        return
