
from .core import ActivationMatrix
from .core import Dictionary
from .core import CSC
from .core import CDL

import numpy as np


class CPDL:

    def __init__(self, X, K, L, regularization) -> None:

        # Original signal
        self.raw_X = np.array(X)
        # self.X = self._normalizeSignal()
        self.X = np.array(X)

        # Signal length
        self.N = len(X)

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
            self.N, self.K, self.L,
            self.regularization,
            self.X, self.D, self.Z,
            method='ISTA'
        )
        self.cdl = CDL(
            self.N, self.K, self.L,
            self.regularization,
            self.X, self.D, self.Z
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

    def _normalizeSignal(self):
        """
        Normalize input X signal.
        """
        return (self.raw_X - self.raw_X.mean()) / self.raw_X.std()

    def _generateDictionary(self):
        """
        Generate a dictionary object.
        """
        return Dictionary(self.N, self.K, self.L)

    def _generateActivationMatrix(self):
        """
        Generate an activation vector object.
        """
        return ActivationMatrix(self.N, self.K, self.L)

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

        # Compute interesting values
        conv = np.sum(
            [
                np.convolve(z_k, d_k)
                for z_k, d_k
                in zip(
                    self.getActivationMatrix(),
                    self.getDictionary().T
                )
            ], axis=0
        )
        residuals = np.linalg.norm(self.X - conv)**2
        print(f'{t} \t\t | \t {residuals:.4f}')

        return

    def reconstruct(self):
        """
        Reconstruct the signal from the atoms and their activations.
        """
        return
