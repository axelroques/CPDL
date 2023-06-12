
from scipy.linalg import toeplitz
import numpy as np


class ActivationMatrix:

    def __init__(self, N, K, L) -> None:

        # Signal length
        self.N = N

        # Number of atoms
        self.K = K

        # Atom length
        self.L = L

        # Initialize activations
        self.Z = self._initActivations()

    def _initActivations(self):
        """
        Initializes the activation matrix with random
        activations.
        """

        # Initialize activations with zeros
        activations = np.zeros((self.K, self.N-self.L+1), dtype=np.float16)

        # Include random activations
        for i_atom in range(self.K):
            rng = np.random.default_rng()
            rand_indices = rng.integers(
                low=0, high=self.N-self.L+1,
                endpoint=False,
                # size=self.N//self.L
                size=1
            )
            activations[i_atom, rand_indices] = 1

        # return activations
        return np.ones((self.K, self.N-self.L+1), dtype=np.float16)*0.2

    def getActivations(self):
        """
        Return activations.

        Activation vector shape = K x (N-L+1) 
        """
        return self.Z

    def getToeplitzFormalismCSC(self):
        """
        Return matrix T, i.e. the Toeplitz formalism
        of the activation matrix in the CSC problem.

        Each atom gets converted to a Toeplitz matrix
        and then each matrix is stacked.

        The resulting matrix is of size (N-L+1)K x 1.
        """

        # Stack each atom activation vector vetically
        T = self.getActivations().flatten()

        return T

    def getToeplitzFormalismCDL(self):
        """
        Return matrix T, i.e. the Toeplitz formalism
        of the activation matrix in the CDL problem.

        Each atom gets converted to a Toeplitz matrix
        and then each matrix is stacked.

        The resulting matrix is of size N x LK.
        """

        T = []
        for activation in self.getActivations():

            # The Toeplitz matrix is constructed from a
            # single column, thanks to scipy.toeplitz
            T_col = np.zeros(self.N)
            T_col[:self.N-self.L+1] = activation
            T.append(toeplitz(T_col, np.zeros(self.L)))

        return np.hstack(T)

    def updateActivations(self, Z):
        """
        Update activation matrix.
        """

        self.Z = Z

        return

    def yieldActivations(self):
        """
        Activations iterator.
        """

        for activation in self.Z:
            yield activation
