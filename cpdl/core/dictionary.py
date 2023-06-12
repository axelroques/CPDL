
from ..functions import sigmoid, sigmoidDerivatives

from scipy.linalg import toeplitz
import numpy as np
import random


class Dictionary:

    def __init__(self, N, K, L) -> None:

        # Length of the signal
        self.N = N

        # Number of atoms
        self.K = K

        # Length of the atoms
        self.L = L

        # Dictionary initialization
        self.D = self._initDictionary()

    def _initDictionary(self):
        """
        Initializes the dictionary with sigmoids whose 
        parameters are randomly assigned.
        """
        return [Atom(i, self.L) for i in range(self.K)]

    def getDictionary(self):
        """
        Return concatenated dictionary.

        Dictionary shape = L x K.
        """
        return np.stack([atom.getSigmoid() for atom in self.D]).T

    def yieldAtoms(self):
        """
        Atom iterator.
        """
        for atom in self.D:
            yield atom

    def getToeplitzFormalismCSC(self):
        """
        Return matrix T, i.e. the Toeplitz formalism
        of the dictionary in the CSC problem.

        Each atom gets converted to a Toeplitz matrix
        and then each matrix is stacked.

        The resulting matrix is of size N x NK.
        """

        T = []
        for atom in self.getDictionary().T:

            # The Toeplitz matrix is constructed from a
            # single column, thanks to scipy.toeplitz
            T_col = np.zeros(self.N)
            T_col[:self.L] = atom
            # T.append(toeplitz(T_col, np.zeros(self.N-self.L+1)))
            T.append(toeplitz(T_col, np.zeros(self.N-self.L+1)))

        return np.hstack(T)

    def getToeplitzFormalismCDL(self):
        """
        Return matrix T, i.e. the Toeplitz formalism
        of the dictionary in the CDL problem.

        Each atom gets converted to a Toeplitz matrix
        and then each matrix is stacked.

        The resulting matrix is of size LK x 1.
        """

        # Stack each atom activation vector vetically
        T = self.getDictionary().flatten('F')

        return T

    def getDelta_k(self, atom):
        """
        Return matrix Delta_k, whose columns are constructed
        like the Toeplitz formalism of D but contain the 
        partial derivatives of the atoms with respect to 
        alpha_k.
        E.g., the first column of Delta_k will 
        contain the partial derivatives of all atoms
        phi_1, ..., phi_K with respect to alpha_k.

        Hence Delta_k mostly contains zeros, except at the 
        position of atom k.
        """

        # Initialize matrix
        Delta_k = np.zeros((self.L*self.K, len(atom.parameters)))

        # Get the atom's partial derivatives. Shape = L x 4
        derivatives = np.stack(
            [atom.getDerivative(p) for p in range(len(atom.parameters))]
        ).T

        # Fill Delta_k
        Delta_k[atom.id*self.L:(atom.id+1)*self.L, :] = derivatives

        return Delta_k


class Atom:

    def __init__(self, id, L) -> None:

        # Atom id
        self.id = id

        # Atom length
        self.L = L

        # Arbitrary time vector to construct the sigmoids
        self.t = np.arange(self.L, dtype=np.float64)

        # Atom parameters
        self.parameters = self._initParameters()

    def _initParameters(self):
        """
        Initialize random parameters.
        Values are derived from physiological
        data:
            - E_0 and E_max = oculomotor range
            - t_50 = typical half duration of 
            a saccade
            - alpha = reasonable values
        """

        E_0 = random.uniform(-53, 53)
        E_max = random.uniform(-53, 53)
        t_50 = self.t[self.L//2]
        alpha = random.uniform(10, 50)

        return np.array([E_0, E_max, t_50, alpha], dtype=np.float64)

    def getSigmoid(self, bypass_norm=True):
        """
        Return values of a sigmoid function
        using the atom parameters.
        """

        if bypass_norm:
            return sigmoid(self.t, *self.parameters)
        else:
            s = sigmoid(self.t, *self.parameters)
            norm = np.linalg.norm(s)
            return s/norm

    def getDerivative(self, p):
        """
        Get partial derivative with regards to
        parameter p.
        """
        return sigmoidDerivatives(self.t, *self.parameters, p)

    def updateParameter(self, i, value):
        """
        Update parameter self.parameters[i] 
        with value.
        """

        self.parameters[i] = value

        return
