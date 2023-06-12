
from .function import Function

from scipy.linalg import toeplitz
import numpy as np


class Dictionary:

    def __init__(self, N, K, L, function='sigmoid') -> None:

        # Length of the signal
        self.N = N

        # Number of atoms
        self.K = K

        # Length of the atoms
        self.L = L

        # Dictionary initialization
        self.D = self._initDictionary(function)

    def _initDictionary(self, function):
        """
        Initializes the dictionary with sigmoids whose 
        parameters are randomly assigned.
        """
        return [Atom(i, self.L, function) for i in range(self.K)]

    def getDictionary(self):
        """
        Return concatenated dictionary.

        Dictionary shape = L x K.
        """
        return np.stack([atom.getFunction() for atom in self.D]).T

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

    def __init__(self, id, L, function) -> None:

        # Atom id
        self.id = id

        # Atom length
        self.L = L

        # Atom parametric function
        self.F = Function(function)

        # Arbitrary time vector to construct the function
        self.t = np.arange(self.L, dtype=np.float64)

        # Atom parameters
        self.parameters = self._initParameters()

    def _initParameters(self):
        """
        Initialize with random function parameters.
        Returns an array of parameters.
        """
        return self.F.init(
            t=self.t,
            L=self.L
        )

    def getFunction(self):
        """
        Return values of the parametric function
        given the atom parameters.
        """
        return self.F.get(self.t, *self.parameters)

    def getDerivative(self, p):
        """
        Get partial derivative with regards to
        parameter p.
        """
        return self.F.derivative(self.t, *self.parameters, p)

    def updateParameter(self, i, value):
        """
        Update parameter self.parameters[i] 
        with value.
        """

        self.parameters[i] = value

        return
