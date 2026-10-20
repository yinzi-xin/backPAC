import jax.numpy as np
import dLux as dl
import equinox as eqx
import zodiax as zdx
from .mplc_model import MPLC, InputField

class MPLCSystem(zdx.Base):
    """
    Wraps an `MPLC` layer with the input/target modes and wavelengths needed to
    propagate a full set of modes through the MPLC system.

    Parameters
    ----------
    input_modes : np.ndarray
        Array of input complex field amplitudes, shape (n_modes, wf_npixels, wf_npixels).
    target_modes : np.ndarray
        Array of target complex field amplitudes, shape (n_modes, wf_npixels, wf_npixels).
    diameter : float
    num_planes : int
        Number of phase screens in the MPLC system.
    total_length : float
        Total length of the MPLC system (distance from input to output).
    target_coordspec_n : int
    target_coordspec_d : float
    wavelengths : np.ndarray
        Array of wavelengths to propagate through the MPLC system, shape (n_wavelengths,).
    """
    input_modes: np.ndarray
    target_modes: np.ndarray
    diameter: float
    wavelengths: np.ndarray
    wf_npixels: int
    mplc: MPLC

    def __init__(self, input_modes, target_modes, diameter, num_planes, total_length, target_coordspec_n, target_coordspec_d, wavelengths):
        self.input_modes = input_modes
        self.target_modes = target_modes
        self.diameter = diameter
        self.wavelengths = wavelengths
        self.wf_npixels = input_modes.shape[1]  # Assuming input_modes has shape (n_modes, wf_npixels, wf_npixels)

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
            Array of shape (n_modes, n_wavelengths, target_coordspec_n, target_coordspec_n)
            containing the output complex fields for each mode and wavelength.
        """

        def propagate_mode(mode: np.ndarray) -> np.ndarray:
            input_layer = InputField(field=mode, diameter=self.diameter)

            def propagate_wavelength(wavelength: np.ndarray) -> np.ndarray:
                wavefront = dl.Wavefront(wavelength, self.wf_npixels, diameter=self.diameter)
                wavefront = input_layer(wavefront)
                wavefront = self.mplc(wavefront)
                return wavefront.phasor

            return eqx.filter_vmap(propagate_wavelength)(self.wavelengths)

        return eqx.filter_vmap(propagate_mode)(self.input_modes)

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