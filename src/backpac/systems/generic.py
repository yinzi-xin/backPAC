import jax
import jax.numpy as np
import dLux as dl
import equinox as eqx
import zodiax as zdx
from ..mplc_model import MPLC, InputField
from ..utils.operations import bin_phase_screens, mask_to_beam_size

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
    aperture_diameter : float
        Diameter of the input aperture.
    mplc_diameter : float
        Diameter of the MPLC system.
    array_diameter : float
        Diameter of the array of the simulation.
    bin_size : int
        Size of the binning to apply to the phase screens.
    num_planes : int
        Number of phase screens in the MPLC system.
    total_length : float
        Total length of the MPLC system (distance from input to output).
    target_coordspec: dl.CoordSpec
        Coordinate specification for the target plane.
    wavelengths : np.ndarray
        Array of wavelengths to propagate through the MPLC system, shape (n_wavelengths,).
    """
    input_modes: np.ndarray
    target_modes: np.ndarray
    aperture_diameter: float
    mplc_diameter: float
    array_diameter: float
    bin_size: int = eqx.field(static=True)
    target_coordspec: dl.CoordSpec = eqx.field(static=True)
    wavelengths: np.ndarray
    wf_npixels: int = eqx.field(static=True)
    mplc: MPLC

    def __init__(self, input_modes, target_modes, aperture_diameter, mplc_diameter, array_diameter, bin_size, num_planes, total_length, target_coordspec, wavelengths):
        self.input_modes = input_modes
        self.target_modes = target_modes
        self.aperture_diameter = aperture_diameter
        self.mplc_diameter = mplc_diameter
        self.array_diameter = array_diameter
        self.bin_size = bin_size
        self.target_coordspec = target_coordspec
        self.wavelengths = wavelengths
        self.wf_npixels = input_modes.shape[1]  # Assuming input_modes has shape (n_modes, wf_npixels, wf_npixels)

        self.mplc = MPLC(
            phase_screens=np.zeros((num_planes, self.wf_npixels, self.wf_npixels)),
            total_length=total_length,
            num_planes=num_planes,
            array_diameter=array_diameter,
            wf_npixels=self.wf_npixels,
            target_coordspec=target_coordspec
        )

    def with_params(self, params: dict) -> "MPLCSystem":
        """
        Derive the physical phase screens from raw optimizable `params` and
        return a new `MPLCSystem` with them applied.

        Parameters
        ----------
        params : dict
            Raw optimizable parameters, with keys "phase_screens" (pre-bin,
            pre-mask) and "total_length".

        Returns
        -------
        MPLCSystem
            A new instance with the binned and beam-masked phase screens and
            the updated total length set on its `mplc` layer.
        """
        phase_screens = params["phase_screens"]
        n = phase_screens.shape[-1]
        binned_phase_screens = bin_phase_screens(phase_screens, self.bin_size)
        filtered_phase_screens = mask_to_beam_size(binned_phase_screens, npixels=n, diameter=self.mplc_diameter, array_diameter=self.array_diameter)

        return self.set("mplc.phase_screens", filtered_phase_screens).set("mplc.total_length", params["total_length"])

    def __call__(self) -> np.ndarray:
        """
        Propagate every input mode, at every wavelength, through the MPLC system.

        Returns
        -------
        output_fields : np.ndarray
            Array of shape (n_modes, n_wavelengths, target_coordspec.n, target_coordspec.d)
            containing the output complex fields for each mode and wavelength.
        """

        def propagate_mode(mode: np.ndarray) -> np.ndarray:
            input_layer = InputField(field=mode, array_diameter=self.array_diameter)

            def propagate_wavelength(wavelength: np.ndarray) -> np.ndarray:
                wavefront = dl.Wavefront(wavelength, self.wf_npixels, diameter=self.array_diameter)
                wavefront = input_layer(wavefront)
                wavefront = self.mplc(wavefront)
                return wavefront.phasor

            return eqx.filter_vmap(propagate_wavelength)(self.wavelengths)

        return eqx.filter_vmap(propagate_mode)(self.input_modes)

def stack_systems(mplc_systems) -> MPLCSystem:
    """
    Combine same-structure `MPLCSystem`s into a single batched `MPLCSystem`
    pytree with a leading system axis on every leaf, suitable for `eqx.filter_vmap`.

    Parameters
    ----------
    mplc_systems : Sequence[MPLCSystem]
        MPLC systems to stack. Must all share the same pytree structure
        (e.g. same array shapes, num_planes, etc.).

    Returns
    -------
    MPLCSystem
        A single `MPLCSystem` whose leaves are the stacked leaves of the
        inputs, with shape `(n_systems, ...)`.
    """
    return jax.tree_util.tree_map(lambda *leaves: np.stack(leaves), *mplc_systems)

def loss_WFM(mplc_system: MPLCSystem):
    """
    Wavefront matching loss: propagate a system's input modes through it and
    measure how well the outputs overlap with the target modes.

    Parameters
    ----------
    mplc_system : MPLCSystem
        A single (unbatched) `MPLCSystem`.

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