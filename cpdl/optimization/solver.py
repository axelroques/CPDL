

class Solver:

    def __init__(self, kwargs) -> None:

        # Basic parameters
        self.N = kwargs['N']
        self.P = kwargs['P']
        self.T = kwargs['T']
        self.K = kwargs['K']
        self.L = kwargs['L']
        self.regularization = kwargs['regularization']
        self.X = kwargs['X']
        self.D = kwargs['D']
        self.Z = kwargs['Z']

        # TODO
        self.step_size = None

    def _lineSearch(self):
        """
        Backtracking stepsize rule to avoid the costly
        computation of the Lipschitz constant.
        """
        return
