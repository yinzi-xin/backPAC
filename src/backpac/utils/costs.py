from backpac.systems.generic import MPLCSystem
import jax.numpy as np

def loss_WFM(res, target_modes):
    """
    Wavefront matching loss: propagate a system's input modes through it and
    measure how well the outputs overlap with the target modes.

    Parameters
    ----------
    res : np.ndarray
        The output complex fields from the MPLC system, shape (n_batch, n_modes, n_wavelengths, target_coordspec_n, target_coordspec_n).
    target_modes : np.ndarray
        The target modes (shared across the batch), shape (n_target_modes, wf_npixels, wf_npixels).

    Returns
    -------
    loss : float
        1 minus the mean squared overlap (fidelity) between the propagated
        output modes and their target modes, averaged over the batch,
        wavelengths and modes. A perfect match yields a loss of 0.
    """

    #calculate the loss (summed over wavelengths and modes)
    #expand the target modes to broadcast over the batch and wavelength axes of res
    target = target_modes[None, :, None, ...]

    #calculate the overlap integral between the output modes and the target modes
    overlap = np.sum(np.conj(res) * target, axis=(-2, -1))

    #calculate the loss as 1 - the mean of the squared magnitude of the overlap
    return 1 - np.mean(np.abs(overlap)**2)

#from Andrea, for cleanliness of the output beam
def loss_cleanliness(res, target_modes):
    target = target_modes[None, :, None, ...]
    g = np.sum(np.conj(target) * res, axis=(-2, -1))
    residual = res - g[..., None, None] * target
    power_r = np.sum(np.abs(residual) ** 2, axis=(-2, -1))
    power_g = np.abs(g) ** 2
    c = power_r / power_g

    cleanliness = np.mean(np.log10(c + 1e-12))
    return cleanliness

def loss_crosstalk(res, target_modes):
    """
    Calculate the crosstalk loss for a given MPLC system.

    Parameters
    ----------
    res : np.ndarray
        The output complex fields from the MPLC system, shape (n_batch, n_modes, n_wavelengths, target_coordspec_n, target_coordspec_n).
    target_modes : np.ndarray
        The target modes (shared across the batch), shape (n_target_modes, wf_npixels, wf_npixels).

    Returns
    -------
    loss : float
        The crosstalk: squared overlap of each mode's output with every other
        mode's target, summed over wavelengths and mode pairs, averaged over the batch.
    """
    n_modes = target_modes.shape[0]

    # overlap[b, w, i, j] = <target_i | res_j>: leakage of output mode j into target i
    overlap = np.einsum('ixy,bjwxy->bwij', np.conj(target_modes), res)

    # keep only i != j (genuine cross-mode leakage); drop the i == j (correctly-coupled) terms
    off_diagonal_mask = 1.0 - np.eye(n_modes)
    crosstalk_power = np.abs(overlap) ** 2 * off_diagonal_mask

    # sum over wavelengths and mode pairs per batch sample, then average over the batch
    term = np.sum(crosstalk_power, axis=(1, 2, 3))
    return np.mean(term)

def loss_nuisance_crosstalk(res, res_nuisance, target_modes):
    """
    Calculate the crosstalk loss from nuisance modes to target modes for a given MPLC system.

    Parameters
    ----------
    res : np.ndarray
        The output complex fields from the MPLC system, shape (n_batch, n_modes, n_wavelengths, target_coordspec_n, target_coordspec_n).
    res_nuisance : np.ndarray
        The output complex fields from the nuisance modes, shape (n_batch, n_nuisance_modes, n_wavelengths, target_coordspec_n, target_coordspec_n).
    target_modes : np.ndarray
        The target modes (shared across the batch), shape (n_target_modes, wf_npixels, wf_npixels).

    Returns
    -------
    loss : float
        The crosstalk from nuisance modes to target modes, normalized by the power in the target modes and averaged over the batch.
    """
    target = target_modes[None, :, None, ...]

    # overlap[b, w, i, k] = <target_i | res_nuisance_k>: leakage of nuisance mode k into target i
    overlap_nuisance = np.einsum('ixy,bkwxy->bwik', np.conj(target_modes), res_nuisance)

    # per-batch total nuisance leakage, summed over wavelengths, target modes and nuisance modes
    crosstalk = np.sum(np.abs(overlap_nuisance) ** 2, axis=(1, 2, 3))

    g = np.sum(np.conj(target) * res, axis=(-2, -1))
    power_g = np.abs(g) ** 2

    return np.squeeze(np.mean(np.log10(crosstalk[:, None, None] / power_g + 1e-12)))