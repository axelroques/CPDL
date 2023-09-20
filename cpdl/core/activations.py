
from scipy.linalg import toeplitz
import numpy as np


class ActivationMatrix:

    def __init__(self, N, T, K, L) -> None:

        # Number of signals
        self.N = N

        # Signal length
        self.T = T

        # Number of atoms
        self.K = K

        # Atom length
        self.L = L

        # Initialize activations
        self.Z = self._initActivations()

    def _initActivations(self):
        """
        Initializes the activation matrix.
        """

        # Initialize activations with zeros
        activations = np.zeros(
            (self.N, self.K, self.T-self.L+1),
            dtype=np.float64
        )

        # # Include random activations
        # for i_atom in range(self.K):
        #     rng = np.random.default_rng()
        #     rand_indices = rng.integers(
        #         low=0, high=self.T-self.L+1,
        #         endpoint=False,
        #         size=1
        #     )
        #     activations[i_atom, rand_indices] = 1

        return activations

    def getActivations(self, n=None):
        """
        Return activations for all, or a single 
        (the nth) time series.

        Activation vector shape = (N x) K x (T-L+1) 
        """

        if n is not None:
            return self.Z[n]
        return self.Z

    def getToeplitzFormalismCSC(self):
        """
        Return matrix T, i.e. the Toeplitz formalism
        of the activation matrix in the CSC problem.

        Each time series returns a matrix of size 
        (T-L+1)K x 1.
        """

        # For each time series
        T_list = []
        for n in range(self.N):

            # Stack each atom activation vector vetically
            T_list.append(self.getActivations(n).flatten())

        return T_list

    def getToeplitzFormalismCDL(self):
        """
        Return matrix T, i.e. the Toeplitz formalism
        of the activation matrix in the CDL problem.

        Each time series returns a matrix of size T x LK.
        """

        # For each time series
        T_list = []
        for n in range(self.N):

            T = []
            for activation in self.getActivations(n):

                # The Toeplitz matrix is constructed from a
                # single column, thanks to scipy.toeplitz
                T_col = np.zeros(self.T, dtype=np.float64)
                T_col[:self.T-self.L+1] = activation
                T.append(toeplitz(T_col, np.zeros(self.L)))

            T_list.append(np.hstack(T))

        return T_list

    def updateActivations(self, Z, n=None):
        """
        Update activation matrix.
        """

        if n is not None:
            self.Z[n] = Z
        else:
            self.Z = Z

    def yieldActivations(self):
        """
        Activations iterator.
        """
        for activation in self.Z:
            yield activation
