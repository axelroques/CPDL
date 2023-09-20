
import numpy as np


def convolution(X, Y):
    """
    Convolution product between X (K x T)
    and Y (K x P x L). 
    """
    return np.sum(
        [
            [
                np.convolve(X_ik, Y_kp)
                for Y_kp in Y_k
            ]
            for X_ik, Y_k in zip(X, Y)
        ], 0
    )


def convolution_1D(Z, D):
    """
    Convolution product between Z (K x T)
    and D (K x L).

    Note: this function is for archiving purposes as
    the convolution function above generalizes it.
    """

    conv = np.sum(
        [
            np.convolve(z_k, d_k)
            for z_k, d_k
            in zip(Z, D)
        ], axis=0
    )

    return conv
