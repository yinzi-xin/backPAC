import jax
import jax.numpy as np

def calc_eta(mplc_system, position):
    """
    Calculate the throughput (eta) of an MPLC system for a given input position.

    Parameters
    ----------
    mplc_system : MPLCSystem
        The MPLC system to evaluate.
    position : tuple
        The (x, y) coordinates of the input position in the input plane.

    Returns
    -------
    eta : float
        The throughput of the MPLC system for the given input position.
    """
    
    return NotImplementedError("calc_eta function is not yet implemented.")

def loss_negative_log_cfi(mplc_systems, star_positions, planet_position, planet_c):
    """
    Negative log10 Classical Fisher Information (CFI) loss: measures -log10(CFI)
    of planet_c under photon-noise statistics, where CFI is averaged over all
    MPLC systems.

    Parameters
    ----------
    mplc_systems : Sequence[MPLCSystem]
        MPLC systems to evaluate.
    star_positions : np.ndarray
        Array of (x, y) input-plane positions to sample the star over, shape
        (n_positions, 2).
    planet_position : tuple
        The (x, y) input-plane position of the planet.
    planet_c : float
        The relative intensity of the planet with respect to the star, in [0, 1].

    Returns
    -------
    loss : float
        The negative log10 of the mean, over all systems, of the CFI of the planet_c.
    """

    total_cfi = 0
    for mplc_system in mplc_systems:
        #calculate average eta_s over the star positions
        eta_s = jax.vmap(calc_eta, in_axes=(None, 0))(mplc_system, star_positions)
        avg_eta_s = np.mean(eta_s)

        #calculate eta_p for the planet position
        eta_p = calc_eta(mplc_system, planet_position)

        cfi = np.sum((eta_p - avg_eta_s)**2 / ( (1-planet_c) * avg_eta_s + planet_c * eta_p ))

        total_cfi += cfi
    return - np.log10(total_cfi / len(mplc_systems))

def gen_star_positions(diameter, n_across):
    """
    Generate a grid of star positions across the input plane.

    Parameters
    ----------
    diameter : float
        The diameter of the input plane.
    n_across : int
        The number of positions to sample across the diameter.

    Returns
    -------
    star_positions : np.ndarray
        Array of (x, y) input-plane positions to sample the star over, shape
        (n_positions, 2).
    """
    x = np.linspace(-diameter/2, diameter/2, n_across)
    y = np.linspace(-diameter/2, diameter/2, n_across)
    xv, yv = np.meshgrid(x, y)

    star_positions = np.stack([xv.flatten(), yv.flatten()], axis=-1)
    star_positions = star_positions[np.linalg.norm(star_positions, axis=1) <= diameter/ 2]
    return star_positions