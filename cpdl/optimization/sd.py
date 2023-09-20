
from .solver import Solver

import numpy as np


class SD(Solver):

    def __init__(self, kwargs) -> None:
        super().__init__(kwargs)

        # Initialize step size
        self.step_size = 0.0001

    def step(self):
        """
        Perform 1 iteration of the method.
        Inspired by Ataee et. al. (2010).
        """

        # Get Toeplitz matrices
        T_D = self.D.getToeplitzFormalismCDL()
        T_Z_list = self.Z.getToeplitzFormalismCDL()

        # For each time series
        for T_Z in T_Z_list:
            # print('T_D =', T_D.shape)
            # print('T_Z =', T_Z.shape)

            # Compute the data fidelity term
            # .squeeze() mandatory here!
            data_fidelity = np.dot(T_Z, T_D) - self.X.squeeze()
            # print('data_fidelity =', data_fidelity.shape, data_fidelity[0])

            # Compute constant derivative term
            R = np.dot(T_Z.T, data_fidelity)
            # print('R =', R.shape, R[0])

            # Iterate over all atoms
            for atom in self.D.yieldAtoms():

                # print('Atom =', atom.id)

                # Construct matrix Delta_k
                Delta_k = self.D.getDelta_k(atom)
                # print('\tDelta_k =', Delta_k.shape)

                # Iterate over parameter
                for i_param in range(len(atom.parameters)):

                    # print('\t Param', i_param)
                    # Get the i_paramth column of Delta_k
                    delta_k_i = Delta_k[:, i_param]

                    # Compute the gradient
                    gradient = 2 * np.dot(delta_k_i.T, R)

                    # print('\t\tgradient =', gradient)
                    # print('\t\tcorrection =', -self.step_size*gradient)

                    # print('\t\tparameters before =', atom.parameters)
                    atom.updateParameter(
                        i_param,
                        atom.parameters[i_param]-self.step_size*gradient
                    )
                    # print('\t\tparameters after =', atom.parameters)
