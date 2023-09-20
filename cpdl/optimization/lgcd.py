
from .solver import Solver

from joblib import Parallel, delayed
import numpy as np


class LGCD(Solver):

    def __init__(self, kwargs) -> None:
        super().__init__(kwargs)

    def step(self):
        """
        Perform 1 iteration of the method.
        """

        # Prepare parallelization
        delayed_update_z = delayed(self._update_z_multi_idx)

        results = Parallel(n_jobs=1)(
            delayed_update_z(
                self.X[n], self.D.getDictionary(),
                self.regularization,
                self.Z.getActivations(n),
            ) for n in range(self.N)
        )

        # Post process the results to get separate objects
        z_hats = []
        for z_hat in results:
            z_hats.append(z_hat)

        # Stack and reorder the columns
        z_hats = np.array(z_hats, dtype=np.float64).reshape(
            self.N, self.K, self.T-self.L+1
        )

        # Update Z
        self.Z.updateActivations(z_hats)

    def _update_z_multi_idx(self, X_i, D, reg, z0_i):

        DtD = self._compute_DtD(D)

        z_hat = self._coordinate_descent_idx(
            X_i, D, DtD, reg=reg, z0=z0_i
        )
        z_hat = z_hat.reshape(self.K, self.T-self.L+1)

        return z_hat

    def _compute_DtD(self, D):
        """
        Compute the DtD matrix
        """

        DtD = np.zeros(shape=(self.K, self.K, 2*self.L-1), dtype=np.float64)
        t0 = self.L - 1
        for k0 in range(self.K):
            for k in range(self.K):
                for t in range(self.L):
                    if t == 0:
                        DtD[k0, k, t0] = np.dot(D[k0].ravel(), D[k].ravel())
                    else:
                        DtD[k0, k, t0 + t] = np.dot(D[k0, :, :-t].ravel(),
                                                    D[k, :, t:].ravel())
                        DtD[k0, k, t0 - t] = np.dot(D[k0, :, t:].ravel(),
                                                    D[k, :, :-t].ravel())
        return DtD

    def _coordinate_descent_idx(
            self, Xi, D, DtD, reg, z0, max_iter=1000, tol=1e-5
    ):
        """
        Compute the coding signal associated to Xi with coordinate descent.

        Parameters
        ----------
        Xi : array, shape (n_channels, n_times)
            The signal to encode.
        D : array
            The atoms. Can either be full rank with shape shape
            (n_atoms, n_channels, n_times_atom) or rank 1 with
            shape shape (n_atoms, n_channels + n_times_atom)
        constants : dict
            Constants containing DtD to speedup computation
        z0 : array, shape (n_atoms, n_times_valid)
            Initial estimate of the coding signal, to warm t_start the algorithm.
        tol : float
            Tolerance for the stopping criterion of the algorithm
        max_iter : int
            Maximal number of iterations run by the algorithm
        n_seg : int or 'auto'
            Number of segments used to divide the coding signal. The updates are
            performed successively on each of these segments.
        """

        t0 = self.L - 1
        t_max = self.T - self.L + 1

        z_hat = z0.copy()

        # Compute sizes for the segments. By default, segment sizes
        # are twice the support of the atoms
        n_times_seg = 2 * np.array(self.L) - 1
        n_seg = max(
            1, t_max // n_times_seg + ((t_max % n_times_seg) != 0)
        )

        max_iter *= n_seg

        norm_Dk = np.array(
            [DtD[k, k, t0] for k in range(self.K)],
            dtype=np.float64
        )[:, None]

        beta, dz_opt, tol = self._init_beta(
            Xi, z_hat, D, reg, norm_Dk, tol
        )

        accumulator = n_seg
        active_segs = np.array([True] * n_seg)
        i_seg = 0
        seg_bounds = [0, n_times_seg]
        t0, k0 = -1, 0
        for _ in range(int(max_iter)):
            k0, t0, dz = self._select_coordinate(
                self.K, t_max, dz_opt,
                active_segs[i_seg], n_times_seg, seg_bounds
            )

            # Update the selected coordinate and beta, only if the update is
            # greater than the convergence tolerance.
            if abs(dz) > tol:
                # update the selected coordinate
                z_hat[k0, t0] += dz

                # Update beta
                beta, dz_opt, accumulator, active_segs = self._update_beta(
                    self.L, t_max, beta, dz_opt, accumulator, active_segs, z_hat, DtD, norm_Dk,
                    dz, k0, t0, reg, seg_bounds, i_seg
                )

            elif active_segs[i_seg]:
                accumulator -= 1
                active_segs[i_seg] = False

            # Check stopping criterion
            if accumulator == 0:
                break

            # Increment to next segment
            i_seg += 1
            seg_bounds[0] += n_times_seg
            seg_bounds[1] += n_times_seg

            if seg_bounds[0] >= t_max:
                # reset to first segment
                i_seg = 0
                seg_bounds = [0, n_times_seg]

        return z_hat

    def _init_beta(self, Xi, z_hat, D, reg, norm_Dk, tol):
        """
        Parameters
        ----------
        X_i : ndarray, shape (n_channels, *sig_support)
            Image to encode on the dictionary D
        z_i : ndarray, shape (n_atoms, *valid_support)
            Warm start value for z_hat
        D : ndarray, shape (n_atoms, n_channels, *atom_support)
            Current dictionary for the sparse coding
        reg : float
            Regularization parameter
        constants : dictionary, optional
            Pre-computed constants for the computations
        return_dE : boolean
            If set to true, return a vector holding the value of cost update when
            updating coordinate i to value dz_opt[i].
        """

        # Init beta with -DtX
        beta = self._gradient_zi(
            Xi, z_hat, D, reg
        )

        for k, t in zip(*z_hat.nonzero()):
            beta[k, t] -= z_hat[k, t] * norm_Dk[k]  # np.sum(DtD[k, k, t0])

        dz_opt = -(beta - np.clip(beta, - reg, reg)) / norm_Dk - z_hat
        tol = tol * np.std(Xi)

        return beta, dz_opt, tol

    def _gradient_zi(self, Xi, zi, D, reg):
        """
        Parameters
        ----------
        Xi : array, shape (n_channels, n_times)
            The data array for one trial
        z_i : array, shape (n_atoms, n_times_valid)
            The activations
        D : array
            The current dictionary, it can have shapes:
            - (n_atoms, n_channels + n_times_atom) for rank 1 dictionary
            - (n_atoms, n_channels, n_times_atom) for full rank dictionary

        Returns
        -------
        (func) : float
            The objective function l2
        grad : array, shape (n_atoms, n_times_valid)
            The gradient
        """
        return self._l2_gradient_zi(Xi, zi, D=D) + reg

    def _l2_gradient_zi(self, Xi, z_i, D):

        Dz_i = self._dense_convolve_multi(z_i, D)

        if Xi is not None:
            Dz_i -= Xi

        grad = self._dense_transpose_convolve_d(Dz_i, D)

        return grad

    @staticmethod
    def _update_beta(
        L, t_max, beta, dz_opt, accumulator, active_segs, z_hat, DtD, norm_Dk,
        dz, k0, t0, reg, seg_bounds, i_seg
    ):

        # Define the bounds for the beta update
        t_start_up = max(0, t0 - L + 1)
        t_end_up = min(t0 + L, t_max)

        # Update beta
        beta_i0 = beta[k0, t0]
        ll = t_end_up - t_start_up
        offset = max(0, L-t0-1)
        beta[:, t_start_up:t_end_up] += DtD[:, k0, offset:offset + ll] * dz
        beta[k0, t0] = beta_i0

        # Update dz_opt
        seg = beta[:, t_start_up:t_end_up]
        tmp = -(seg - np.clip(seg, -reg, reg)) / norm_Dk
        dz_opt[:, t_start_up:t_end_up] = tmp - z_hat[:, t_start_up:t_end_up]
        dz_opt[k0, t0] = 0

        # Reenable greedy updates in the segments immediately before or after
        # if beta was updated outside the segment
        t_start_seg, t_end_seg = seg_bounds
        if t_start_up < t_start_seg and not active_segs[i_seg - 1]:
            accumulator += 1
            active_segs[i_seg - 1] = True
        if t_end_up > t_end_seg and not active_segs[i_seg + 1]:
            accumulator += 1
            active_segs[i_seg + 1] = True

        return beta, dz_opt, accumulator, active_segs

    @staticmethod
    def _select_coordinate(
        K, t_max, dz_opt, active_seg, n_times_seg, seg_bounds
    ):
        """
        Parameters
        ----------
        dz_opt : ndarray, shape (n_atoms, *valid_support)
            Difference between the current value and the optimal value for each
            coordinate.
        dE : ndarray, shape (n_atoms, *valid_support) or None
            Value of the reduction of the cost when moving a given coordinate to
            the optimal value dz_opt. This is only necessary when strategy is
            'gs-q'.
        segments : dicod.utils.Segmentation
            Segmentation info for LGCD
        i_seg : int
            Current segment indices in the Segmentation object.
        """

        # Focus on the current active segment
        t_start_seg, t_end_seg = seg_bounds
        if active_seg:
            # The coordinate to select is the one associated with the
            # largest absolute value in dz
            i0 = abs(dz_opt[:, t_start_seg:t_end_seg]).argmax()

            # Get absolute position of i0 in dz
            n_times_current = min(n_times_seg, t_max - t_start_seg)
            k0, t0 = np.unravel_index(i0, (K, n_times_current))
            t0 += t_start_seg

            # Retrieve maximal difference
            dz = dz_opt[k0, t0]

        # If segment is inactive, dz should be zero
        else:
            k0, t0, dz = None, None, 0

        return k0, t0, dz

    @staticmethod
    def _dense_convolve_multi(z_i, ds):
        """Convolve z_i[k] and ds[k] for each atom k, and return the sum."""
        return np.sum([[np.convolve(zik, dkp) for dkp in dk]
                       for zik, dk in zip(z_i, ds)], 0)

    @staticmethod
    def _dense_transpose_convolve_d(residual_i, D):
        """Convolve residual[i] with the transpose for each atom k

        Parameters
        ----------
        residual_i : array, shape (n_channels, n_times)
        D : array, shape (n_atoms, n_channels, n_times_atom)

        Return
        ------
        grad_zi : array, shape (n_atoms, n_times_valid)

        """
        return np.sum(
            [
                [
                    np.correlate(res_ip, d_kp, mode='valid')
                    for res_ip, d_kp in zip(residual_i, d_k)
                ]
                for d_k in D
            ], axis=1
        )
