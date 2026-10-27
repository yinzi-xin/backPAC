from backpac.systems.generic import MPLCSystem
import jax.numpy as np

def loss_WFM(mplc_system: MPLCSystem):
    """
    Wavefront matching loss: propagate a system's input modes through it and
    measure how well the outputs overlap with the target modes.

    Parameters
    ----------
    mplc_system : MPLCSystem
        A single `MPLCSystem`.

    Returns
    -------
    loss : float
        1 minus the mean squared overlap (fidelity) between the propagated
        output modes and their target modes, summed over wavelengths and
        modes. A perfect match yields a loss of 0.
    """
    #propagate every input mode through the MPLC system
    res = mplc_system()

    #calculate the loss (summed over wavelengths and modes)
    #expand the target modes to match the shape of res
    target_modes_expanded = np.expand_dims(mplc_system.target_modes, axis=1)

    #calculate the overlap integral between the output modes and the target modes
    overlap = np.sum(np.conj(res) * target_modes_expanded, axis=(-2, -1))

    #calculate the loss as 1 - the mean of the squared magnitude of the overlap
    return 1 - np.mean(np.abs(overlap)**2)

#from Andrea, for cleanliness of the output beam
def loss_cleanliness(mplc_system: MPLCSystem):
    res = mplc_system()
    target = mplc_system.target_modes[:, None, ...]

    g = np.sum(np.conj(target) * res, axis=(-2, -1))
    residual = res - g[..., None, None] * target
    power_r = np.sum(np.abs(residual) ** 2, axis=(-2, -1))
    power_g = np.abs(g) ** 2
    c = power_r / power_g

    cleanliness = np.mean(np.log10(c + 1e-12))
    return cleanliness

#cross talk amongst the target modes
def loss_crosstalk(mplc_system: MPLCSystem):
    """
    Calculate the crosstalk loss for a given MPLC system.

    Parameters
    ----------
    mplc_system : MPLCSystem
        A single `MPLCSystem`.

    Returns
    -------
    loss : float
        The crosstalk
    """
    res = mplc_system()
    target = mplc_system.target_modes[:, None, ...]

    # Calculate the crosstalk as the sum of the squared magnitudes of the overlaps
    # between the output modes and the target modes, excluding the diagonal.
    crosstalk = np.sum(np.abs(np.sum(np.conj(res) * target, axis=(-2, -1))) ** 2) - np.sum(np.abs(np.sum(np.conj(target) * target, axis=(-2, -1))) ** 2)
    return crosstalk

#cross talk from nuisance modes to target modes
def loss_nuisance_crosstalk(mplc_system: MPLCSystem):
    """
    Calculate the crosstalk loss from nuisance modes to target modes for a given MPLC system.

    Parameters
    ----------
    mplc_system : MPLCSystem
        A single `MPLCSystem`.

    Returns
    -------
    loss : float
        The crosstalk from nuisance modes to target modes.
    """
    res_nuisance = mplc_system.propagate_nuisance_modes()
    target = mplc_system.target_modes[:, None, ...]

    # Calculate the crosstalk as the sum of the squared magnitudes of the overlaps
    # between the output nuisance modes and the target modes.
    crosstalk = np.sum(np.abs(np.sum(np.conj(res_nuisance) * target, axis=(-2, -1))) ** 2)
    return crosstalk