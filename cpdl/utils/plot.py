
import matplotlib.pyplot as plt
import numpy as np


def plot_atoms(X, D, Z):
    """
    Plot all atoms in a dictionary.
    """

    _, ax = plt.subplots(figsize=(10, 3))

    # Arbitrary time vector
    t = np.arange(len(X))

    # Plot original signal
    ax.plot(t, X, label='Signal')

    # Colors
    colors = ['#83a83b', '#c44e52', '#8172b2', '#ff914d', '#77BEDB']

    for i_atom, (atom, activation) in enumerate(
        zip(D.yieldAtoms(), Z.yieldActivations())
    ):

        # Get atom content
        sigmoid = atom.getFunction()

        # Iterate over the activation vector
        n_plot = 0
        max_activation_value = np.abs(activation).max()
        for t_activation, val_activation in enumerate(activation):

            # If activation value is non-zero, plot the atom
            if abs(val_activation) > 0.1:
                alpha = np.abs(val_activation) / max_activation_value

                ax.plot(
                    np.arange(t_activation, t_activation+atom.L),
                    sigmoid*val_activation,
                    label=f'Atom {atom.id}' if n_plot == 0 else '',
                    c=colors[i_atom % len(colors)],
                    alpha=alpha
                )

                n_plot += 1

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

    return


def plot_reconstructed_signal(X, D, Z):
    """
    Plot reconstructed signal.
    """

    _, ax = plt.subplots(figsize=(10, 3))

    # Arbitrary time vector
    t = np.arange(len(X))

    # Plot original signal
    ax.plot(t, X, label='Signal')

    # Reconstruct signal
    X_reconstructed = np.zeros_like(X)
    for atom, activation in zip(D.yieldAtoms(), Z.yieldActivations()):

        # Get atom content
        sigmoid = atom.getFunction()

        # Iterate over the activation vector
        activated = np.where(activation > 0)[0]
        for i_start in activated:
            X_reconstructed[i_start:i_start +
                            atom.L] += sigmoid * activation[i_start]

    ax.plot(t, X_reconstructed, label='Reconstructed')
    ax.legend()
    plt.show()

    return
