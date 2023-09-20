
# from .function import Function
from ..functions import Logistic, Sigmoid

from scipy.linalg import toeplitz
import numpy as np


class Dictionary:

    def __init__(self, T, P, function, K, L) -> None:

        # Length of the signal
        self.T = T

        # Dimensionality of the signal
        self.P = P

        # Number of atoms
        self.K = K

        # Length of the atoms
        self.L = L

        # Dictionary initialization
        self.D = self._initDictionary(function)

    def _initDictionary(self, function):
        """
        Initializes the dictionary with atoms.
        """
        return [Atom(i, self.L, function) for i in range(self.K)]

    def getDictionary(self):
        """
        Return concatenated dictionary.

        Dictionary shape = K x P x L.
        """
        return np.stack([atom.getFunction() for atom in self.D])

    def yieldAtoms(self):
        """
        Atom iterator.
        """
        for atom in self.D:
            yield atom

    def getToeplitzFormalismCSC(self):
        """
        Return matrix T_D, i.e. the Toeplitz formalism
        of the dictionary in the CSC problem.

        Each atom gets converted to a Toeplitz matrix
        and then each matrix is stacked.

        The resulting matrix is of size T x TK.
        """

        T_D = []
        for atom in self.getDictionary():

            # The Toeplitz matrix is constructed from a
            # single column, thanks to scipy.toeplitz
            col = np.zeros(self.T, dtype=np.float64)
            col[:self.L] = atom
            T_D.append(toeplitz(col, np.zeros(self.T-self.L+1)))

        return np.hstack(T_D)

    def getToeplitzFormalismCDL(self):
        """
        Return matrix T_D, i.e. the Toeplitz formalism
        of the dictionary in the CDL problem.

        Each atom gets converted to a Toeplitz matrix
        and then each matrix is stacked.

        The resulting matrix is of size LK x 1.
        """

        # Stack the atoms
        T_D = self.getDictionary().T.flatten('F')

        return T_D

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
        Delta_k = np.zeros(
            (self.L*self.K, len(atom.parameters)),
            dtype=np.float64
        )

        # Get the atom's partial derivatives. Shape = L x 4
        derivatives = np.stack(
            [atom.getDerivative(p) for p in range(len(atom.parameters))]
        ).T

        # Fill Delta_k
        Delta_k[atom.id*self.L:(atom.id+1)*self.L, :] = derivatives

        return Delta_k

    @property
    def n_parameters(self):
        """
        Return the atoms' number of parameters.
        """
        return len(self.D[0].parameters)


class Atom:

    def __init__(self, id, L, function) -> None:

        # Atom id
        self.id = id

        # Atom length
        self.L = L

        # Atom parametric function
        self.F = self._selectFunction(function, L)

        # Atom parameters
        self.parameters = self._initParameters()

    def _initParameters(self):
        """
        Initialize atom with random function parameters.
        Returns an array of parameters.
        """
        return self.F.init()

    def getFunction(self):
        """
        Return values of the parametric function
        given the atom parameters.
        """

        f = self.F.get(*self.parameters)

        # Reshape as (1, len(f)) instead of (len(f),)
        if f.ndim == 1:
            return f.reshape((1, len(f)))

        return f

    def getDerivative(self, p):
        """
        Get partial derivative with regards to
        parameter p.
        """
        return self.F.derivative(*self.parameters, p)

    def updateParameter(self, i, value):
        """
        Update parameter self.parameters[i] 
        with value.
        """

        self.parameters[i] = value

        return

    @staticmethod
    def _selectFunction(function, L):
        """
        Select function instance given the input
        function name.
        """

        choice = {
            'sigmoid': Sigmoid,
            'logistic': Logistic
        }

        if function in choice:
            function = choice[function](L)
        else:
            raise RuntimeError('Unknown function.')

        return function
