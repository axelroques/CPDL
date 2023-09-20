
from ..optimization import PGD, SD


class CDL:
    """
    Solver for the convolutional parametric dictionary 
    learning problem.
    """

    def __init__(self, **kwargs) -> None:

        # Get solver method
        if kwargs['method'] not in ['PGD', 'SD']:
            raise RuntimeError(
                "Unimplemented CSC solver. Try 'PGD', or 'SD'."
            )
        self.method = self._getMethod(kwargs)

    def step(self):
        """
        Perform 1 iteration of the CDL optimization step.
        """

        self.method.step()

        return

    def _getMethod(self, kwargs):
        """
        Instantiate object for the different possible
        optimization algorithms.
        """

        options = {
            'PGD': PGD,
            'SD': SD
        }

        return options[kwargs['method']](kwargs)

    # def _stepISTA(self):
    #     """
    #     Perform 1 iteration of the CSC, with the
    #     desired method ('ISTA' or 'FISTA).
    #     """

    #     # Get Toeplitz matrices
    #     T_D = self.D.getToeplitzFormalismCDL()
    #     T_Z = self.Z.getToeplitzFormalismCDL()
    #     # print('T_D =', T_D.shape)
    #     # print('T_Z =', T_Z.shape)

    #     # Compute the data fidelity term
    #     data_fidelity = np.dot(T_Z, T_D) - self.X
    #     # print('data_fidelity =', data_fidelity.shape, data_fidelity[0])

    #     # Compute constant derivative term
    #     R = np.dot(T_Z.T, data_fidelity)
    #     # print('R =', R.shape, R[0])

    #     # Iterate over all atoms
    #     for atom in self.D.yieldAtoms():
    #         # print('Atom =', atom.id)

    #         # Update parameters using the prox operator
    #         self._proximalOperator(atom, R)

    #     return

    # def _proximalOperator(self, atom, R):
    #     """
    #     Proximal operator for the CDL update,
    #     using the Topeplitz representation of D and Z.
    #     """

    #     # Construct matrix Delta_k
    #     Delta_k = self.D.getDelta_k(atom)
    #     # print('\tDelta_k =', Delta_k.shape)

    #     # Iterate over parameter
    #     for i_param in range(len(atom.parameters)):

    #         # print('\t Param', i_param)
    #         # Get column the i_paramth column of Delta_k
    #         delta_k_i = Delta_k[:, i_param]

    #         # Compute the gradient
    #         gradient = 2 * np.dot(delta_k_i.T, R)
    #         # print('\t\tgradient =', gradient)
    #         # print('\t\tcorrection =', -self.step_size*gradient)

    #         # TODO get prox formula
    #         prox = atom.parameters[i_param]-self.step_size*gradient
    #         # print('\t\tparameters before =', atom.parameters)
    #         atom.updateParameter(
    #             i_param,
    #             prox
    #         )
    #         # print('\t\tparameters after =', atom.parameters)

    #     return
