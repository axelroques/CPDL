
from ..functions import initSigmoid, sigmoid, sigmoidDerivatives


class Function:

    def __init__(self, function='sigmoid') -> None:

        self.function = function

    def init(self, **kwargs):
        """
        Init function parameters.
        """

        choice = {
            'sigmoid': initSigmoid
        }

        if self.function in choice:
            return choice[self.function](**kwargs)
        else:
            raise RuntimeError('Unknown function.')

    def get(self, *args):
        """
        Get function values given the parameters.
        """

        choice = {
            'sigmoid': sigmoid
        }

        if self.function in choice:
            return choice[self.function](*args)
        else:
            raise RuntimeError('Unknown function.')

    def derivative(self, *args):
        """
        Get the partial derivatives.
        """

        choice = {
            'sigmoid': sigmoidDerivatives
        }

        if self.function in choice:
            return choice[self.function](*args)
        else:
            raise RuntimeError('Unknown function.')
