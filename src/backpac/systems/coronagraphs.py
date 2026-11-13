"""
MPLC coronagraph systems for wavefront control and imaging.

Provides the MPLCCoronagraph class, which wraps an MPLC layer with utilities
for coronagraphic applications including input/target modes and wavefront
propagation.
"""

import dLux as dl
import equinox as eqx
import jax.numpy as np
import zodiax as zdx

from ..mplc_model import MPLC, InputField


class MPLCCoronagraph(zdx.Base):
    """
    Wraps an `MPLC` layer with utilities for coronagraphy

    Parameters
    ----------
    wf_npixels : int
        Number of pixels representing the wavefront at each phase screen.
    diameter : float
    num_planes : int
        Number of phase screens in the MPLC system.
    total_length : float
        Total length of the MPLC system (distance from input to output).
    target_coordspec_n : int
    target_coordspec_d : float
    wavelengths : np.ndarray
        Array of wavelengths to propagate through the MPLC system,
        shape (n_wavelengths,).
    """

    wf_npixels: int
    diameter: float
    wavelengths: np.ndarray
    mplc: MPLC

    def __init__(
        self,
        wf_npixels,
        diameter,
        num_planes,
        total_length,
        target_coordspec_n,
        target_coordspec_d,
        wavelengths,
    ):
        self.diameter = diameter
        self.wavelengths = wavelengths
        self.wf_npixels = wf_npixels

        self.mplc = MPLC(
            phase_screens=np.zeros((num_planes, self.wf_npixels, self.wf_npixels)),
            total_length=total_length,
            num_planes=num_planes,
            diameter=diameter,
            wf_npixels=self.wf_npixels,
            target_coordspec_n=target_coordspec_n,
            target_coordspec_d=target_coordspec_d,
        )

    def __call__(self) -> np.ndarray:
        """
        Propagate every input mode, at every wavelength, through the MPLC system.

        Returns
        -------
        output_fields : np.ndarray
            Array of shape (n_modes, n_wavelengths,
                target_coordspec_n, target_coordspec_n)
            containing the output complex fields for each mode and wavelength.
        """

        def propagate_mode(mode: np.ndarray) -> np.ndarray:
            input_layer = InputField(field=mode, diameter=self.diameter)

            def propagate_wavelength(wavelength: np.ndarray) -> np.ndarray:
                wavefront = dl.Wavefront(
                    wavelength, self.wf_npixels, diameter=self.diameter
                )
                wavefront = input_layer(wavefront)
                wavefront = self.mplc(wavefront)
                return wavefront.phasor

            return eqx.filter_vmap(propagate_wavelength)(self.wavelengths)

        return eqx.filter_vmap(propagate_mode)(self.input_modes)


def calc_eta_modal(mplc_coronagraph, position):
    """
    Calculate the throughput (eta) per output mode of an MPLC system for a
    given input position.

    Parameters
    ----------
    mplc_coronagraph : MPLCCoronagraph
        The MPLC coronagraph to evaluate.
    position : tuple
        The (x, y) coordinates of the input position in the input plane.

    Returns
    -------
    eta : float
        The modal throughput of the MPLC system for the given input position.
    """

    return NotImplementedError("calc_eta_modal function is not yet implemented.")


def calc_eta_aperture(mplc_coronagraph, position):
    """
    Calculate the throughput (eta) within a photometric aperture surrounding the
    outputs of an MPLC system for a given input position.

    Parameters
    ----------
    mplc_coronagraph : MPLCCoronagraph
        The MPLC coronagraph to evaluate.
    position : tuple
        The (x, y) coordinates of the input position in the input plane.

    Returns
    -------
    eta : float
        The aperture throughput of the MPLC system for the given input position.
    """

    return NotImplementedError("calc_eta_aperture function is not yet implemented.")


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
    x = np.linspace(-diameter / 2, diameter / 2, n_across)
    y = np.linspace(-diameter / 2, diameter / 2, n_across)
    xv, yv = np.meshgrid(x, y)

    star_positions = np.stack([xv.flatten(), yv.flatten()], axis=-1)
    star_positions = star_positions[
        np.linalg.norm(star_positions, axis=1) <= diameter / 2
    ]
    return star_positions


def loss_negative_log_cfi_aperture(
    mplc_coronagraphs, star_positions, planet_position, planet_c
):
    """
    Negative log10 Classical Fisher Information (CFI) loss: measures -log10(CFI)
    of planet_c under photon-noise statistics, where CFI is averaged over all
    MPLC coronagraphs.

    Parameters
    ----------
    mplc_coronagraphs : Sequence[MPLCCoronagraph]
        MPLC coronagraphs to evaluate.
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
    for mplc_coronagraph in mplc_coronagraphs:
        # calculate average eta_s over the star positions
        eta_s = eqx.filter_vmap(calc_eta_aperture, in_axes=(None, 0))(
            mplc_coronagraph, star_positions
        )
        avg_eta_s = np.mean(eta_s)

        # calculate eta_p for the planet position
        eta_p = calc_eta_aperture(mplc_coronagraph, planet_position)

        cfi = np.sum(
            (eta_p - avg_eta_s) ** 2 / ((1 - planet_c) * avg_eta_s + planet_c * eta_p)
        )

        total_cfi += cfi
    return -np.log10(total_cfi / len(mplc_coronagraphs))
