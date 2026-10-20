import jax.numpy as np

def loss_WFM(mplc_systems):
    """
    Wavefront matching loss: for each `MPLCSystem`, propagate its input modes
    through the system and measure how well the outputs overlap with the
    target modes, averaged over all systems.

    Parameters
    ----------
    mplc_systems : Sequence[MPLCSystem]
        MPLC systems to evaluate.

    Returns
    -------
    loss : float
        Mean over all systems of 1 minus the mean squared overlap (fidelity)
        between the propagated output modes and their target modes, summed
        over wavelengths and modes. A perfect match yields a loss of 0.
    """
    total_loss = 0
    for mplc_system in mplc_systems:
        #propagate every input mode through the MPLC system
        res = mplc_system()

        #calculate the loss (summed over wavelengths and modes)
        #expand the target modes to match the shape of res
        target_modes_expanded = np.expand_dims(mplc_system.target_modes, axis=1)

        #calculate the overlap integral between the output modes and the target modes
        overlap = np.sum(np.conj(res) * target_modes_expanded, axis=(-2, -1))

        #calculate the loss as 1 - the mean of the squared magnitude of the overlap
        loss = 1 - np.mean(np.abs(overlap)**2)

        total_loss += loss
    return total_loss / len(mplc_systems)