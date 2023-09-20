
import matplotlib.pyplot as plt
import numpy as np


def plot_atoms(X, D, Z):
    """
    Plot all atoms in a dictionary.
    Only plot the first time series.
    """

    # Dimension of the time series
    p = X.shape[1]
    # Arbitrary time vector
    t = np.arange(X.shape[-1])

    _, axes = plt.subplots(p, 1, figsize=(10, 3))

    # Quick and ugly fix if ony a single AxesSubplot is returned
    if p == 1:
        axes = [axes]

    # Plot original signal
    for i_p, ax in enumerate(axes):
        ax.plot(t, X[0, i_p, :], label='Signal')

    # Colors
    colors = ['#83a83b', '#c44e52', '#8172b2', '#ff914d', '#77BEDB']

    for i_atom, atom in enumerate(D.yieldAtoms()):

        # Get atom content
        f = atom.getFunction()

        # Get activation vector
        activation = Z.getActivations(0)[i_atom]

        # Get non-zero activation values
        t_non_zero_values = np.where(abs(activation) > 0.1)[0]
        non_zero_values = activation[t_non_zero_values]

        # Plot parameters
        n_plot = 0
        max_activation_value = np.abs(activation).max()
        for t_activation, val_activation in zip(
            t_non_zero_values, non_zero_values
        ):

            for i_p, ax in enumerate(axes):

                alpha = np.abs(val_activation) / max_activation_value
                ax.plot(
                    np.arange(t_activation, t_activation+atom.L),
                    f[i_p] * val_activation,
                    label=f'Atom {atom.id}' if n_plot == 0 else '',
                    c=colors[i_atom % len(colors)],
                    alpha=alpha
                )
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

                n_plot += 1

    plt.tight_layout()
    plt.show()

    return


def plot_reconstructed_signal(X, D, Z):
    """
    Plot reconstructed signal.
    """

    # Dimension of the time series
    p = X.shape[1]
    # Arbitrary time vector
    t = np.arange(X.shape[-1])

    _, axes = plt.subplots(p, 1, figsize=(10, 3))

    # Quick and ugly fix if ony a single AxesSubplot is returned
    if p == 1:
        axes = [axes]

    # Plot original signal
    for i_p, ax in enumerate(axes):
        ax.plot(t, X[0, i_p, :], label='Signal')

    # Reconstruct signal
    X_reconstructed = np.zeros((p, len(t)))

    for i_atom, atom in enumerate(D.yieldAtoms()):

        # Get atom content
        f = atom.getFunction()

        # Get activation vector
        activation = Z.getActivations(0)[i_atom]

        # Iterate over the activation vector
        activated = np.where(np.abs(activation) > 0)[0]
        for i_start in activated:
            for i_p in range(p):
                X_reconstructed[
                    i_p, i_start:i_start + atom.L
                ] += f[i_p] * activation[i_start]

    for i_p, ax in enumerate(axes):
        ax.plot(t, X_reconstructed[i_p], label='Reconstructed')
        ax.legend()

    plt.tight_layout()
    plt.show()

    return
