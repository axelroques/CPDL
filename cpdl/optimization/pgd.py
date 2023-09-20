
from .solver import Solver

from scipy.optimize._linesearch import scalar_search_armijo
import numpy as np


class PGD(Solver):

    def __init__(self, kwargs) -> None:
        super().__init__(kwargs)

        self.step_size = None
        self.init_step_size = 0.01
        self.min_step_size = 1e-10
        self.tol = 1e-8
        self.max_iter = 300

    def step(self):
        """
        Perform 1 iteration of the method.
        """

        # Precompute constants
        self._computeConstants()

        # Initialization
        if not self.step_size:
            self.step_size = self.init_step_size
        D = self.D.getDictionary()
        objective = self._objectiveFunction(D)

        # Iterate over all atoms
        for i_atom, atom in enumerate(self.D.yieldAtoms()):
            # print(f'\nAtom {i_atom}')

            # Iterate over parameter
            for i_param in range(len(atom.parameters)):
                # print(f'\t Parameter {i_param}')

                # Get atom derivative
                atom_derivative = atom.getDerivative(i_param)

                # Compute gradient
                gradient = self._gradPartial(
                    D, atom_derivative, i_atom
                )

                # Get step size
                self.step_size, objective = self._ArmijoLineSearch(
                    self.step_size, D, gradient, objective
                )
                # print(
                #     'Armijo step size =', self.step_size,
                #     'gradient =', gradient
                # )

                # Step size check: if we did not find a valid step size,
                # restart. Otherwise, update the parameters
                if (self.step_size is None) or (self.step_size < self.min_step_size):
                    self.step_size = 0.01
                else:
                    atom.updateParameter(
                        i_param,
                        atom.parameters[i_param] - self.step_size*gradient
                    )
                    D = self.D.getDictionary()

        return

    def _objectiveFunction(self, D):
        """
        Compute the value of the objective function.

        Parameters
        ----------
        D : array, shape (n_atoms, n_channels, *atom_support)
            Current dictionary
        ztz, ztX, XtX : Constant to accelerate the computation
        when updating D.
        """

        grad_D = 0.5 * self._gradD(D)
        cost = (D * grad_D).sum()

        return cost + 0.5*self.XtX

    def _gradD(self, D):
        """
        Compute the gradient of the reconstruction loss relative to D.

        Parameters
        ----------
        constants : dict or None
            Constant to accelerate the computation of the gradient

        Returns
        -------
        grad : array, shape (n_atoms * n_times_valid)
            The gradient
        """
        return self._tensordot_convolve(self.L, self.ztz, D) - self.ztX

    def _gradPartial(self, D, atom_derivative, i_atom):
        """
        Compute the gradient of the reconstruction loss relative to the
        atom parameters using the chain rule.
        """

        # Get gradient relative to D
        grad_D = self._gradD(D)

        # Chain rule
        grad_atom = np.dot(grad_D, atom_derivative)

        return grad_atom[i_atom]

    def _ArmijoLineSearch(self, step_size, D_hat, grad, obj):
        """
        Armijo backtracking line search to find an adequate step size.
        """

        # Local function for the line search
        def phi(step_size):
            D = D_hat - step_size*grad
            return self._objectiveFunction(D)

        norm_grad = np.dot(grad.ravel(), grad.ravel())
        step_size, obj_next = scalar_search_armijo(
            phi=phi,
            phi0=obj,
            derphi0=-norm_grad,
            c1=1e-5,
            alpha0=step_size,
            amin=self.min_step_size
        )

        return step_size, obj_next

    def _computeConstants(self):
        """
        Precompute constants to simplify the gradient computation.
        """

        Z = self.Z.getActivations()

        self.ztz = self._compute_ztz(Z, self.N, self.K, self.L)
        self.ztX = self._compute_ztX(Z, self.X, self.P, self.K, self.L)
        self.XtX = self._compute_XtX(self.X)

    @staticmethod
    def _compute_ztz(z, N, K, L):
        """
        ztz.shape = n_atoms, n_atoms, 2 * n_times_atom - 1
        z.shape = n_trials, n_atoms, n_times - n_times_atom + 1)
        """

        ztz = np.zeros(shape=(K, K, 2*L-1))
        t0 = L - 1
        for i in range(N):
            for k0 in range(K):
                for k in range(K):
                    for t in range(L):
                        if t == 0:
                            ztz[k0, k, t0] += (z[i, k0] * z[i, k]).sum()
                        else:
                            ztz[k0, k, t0 + t] += (
                                z[i, k0, :-t] * z[i, k, t:]).sum()
                            ztz[k0, k, t0 - t] += (
                                z[i, k0, t:] * z[i, k, :-t]).sum()
        return ztz

    @staticmethod
    def _compute_ztX(z, X, P, K, L):
        """
        z.shape = n_trials, n_atoms, n_times - n_times_atom + 1)
        X.shape = n_trials, n_channels, n_times
        ztX.shape = n_atoms, n_channels, n_times_atom
        """

        ztX = np.zeros((K, P, L))
        for n, k, t in zip(*z.nonzero()):
            ztX[k, :, :] += z[n, k, t] * X[n, :, t:t+L]

        return ztX

    @staticmethod
    def _compute_XtX(X):
        return np.dot(X.ravel(), X.ravel())

    @staticmethod
    def _tensordot_convolve(L, ztz, D):
        """
        Compute the multivariate (valid) convolution of ztz and D

        Parameters
        ----------
        ztz: array, shape = (n_atoms, n_atoms, 2 * n_times_atom - 1)
            Activations
        D: array, shape = (n_atoms, n_channels, n_times_atom)
            Dictionnary

        Returns
        -------
        G : array, shape = (n_atoms, n_channels, n_times_atom)
            Gradient
        """

        D_revert = D[:, :, ::-1]

        G = np.zeros_like(D)
        for t in range(L):
            G[:, :, t] = np.tensordot(
                ztz[:, :, t:t+L],
                D_revert,
                axes=([1, 2], [0, 2])
            )

        return G
