
from ..optimization import FISTA, ISTA, LGCD


class CSC:
    """
    Solver for the convolutional sparse coding problem.
    """

    def __init__(self, **kwargs) -> None:

        # Get solver method
        if kwargs['method'] not in ['FISTA', 'ISTA', 'LGCD']:
            raise RuntimeError(
                "Unimplemented CSC solver. Try 'FISTA', 'ISTA', or 'LGCD'."
            )
        self.method = self._getMethod(kwargs)

    def step(self):
        """
        Perform 1 iteration of the CSC optimization step.
        """

        self.method.step()

    def _getMethod(self, kwargs):
        """
        Instantiate object for the different possible
        optimization algorithms.
        """

        options = {
            'FISTA': FISTA,
            'ISTA': ISTA,
            'LGCD': LGCD
        }

        return options[kwargs['method']](kwargs)
