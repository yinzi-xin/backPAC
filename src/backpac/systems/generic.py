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
        Array of input complex field amplitudes, shape (n_modes, wf_npixels, wf_npixels). Expected to be orthogonal and normalized.
    target_modes : np.ndarray
        Array of target complex field amplitudes, shape (n_modes, wf_npixels, wf_npixels). Expected to be (close to) orthogonal and normalized.
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
    telescope_modes : np.ndarray, optional
        Array of telescope eigenmodes, shape (n_telescope_modes, wf_npixels, wf_npixels).
    """
    input_modes: np.ndarray
    target_modes: np.ndarray
    aperture_diameter: float
    mplc_diameter: float
    array_diameter: float
    bin_size: int = eqx.field(static=True)
    target_coordspec: dl.CoordSpec
    wavelengths: np.ndarray
    wf_npixels: int = eqx.field(static=True)
    mplc: MPLC
    telescope_modes: np.ndarray

    def __init__(self, input_modes, target_modes, aperture_diameter, mplc_diameter, array_diameter, bin_size, num_planes, total_length, target_coordspec, wavelengths, telescope_modes=None):
        self.input_modes = input_modes
        self.target_modes = target_modes
        self.aperture_diameter = aperture_diameter
        self.mplc_diameter = mplc_diameter
        self.array_diameter = array_diameter
        self.bin_size = bin_size
        self.target_coordspec = target_coordspec
        self.wavelengths = wavelengths
        self.wf_npixels = input_modes.shape[1]  # Assuming input_modes has shape (n_modes, wf_npixels, wf_npixels)
        self.telescope_modes = telescope_modes

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

    def get_params(self) -> dict:
        """
        Extract the raw optimizable parameters from the `MPLCSystem`.

        Returns
        -------
        params : dict
            Raw optimizable parameters, with keys "phase_screens" (pre-bin,
            pre-mask) and "total_length".
        """
        return {
            "phase_screens": self.mplc.phase_screens,
            "total_length": self.mplc.total_length
        }

    def calculate_nuisance_modes(self) -> np.ndarray:
        """
        Calculate the nuisance modes of the MPLC system, which are the modes that
        are orthogonal to the target modes and can be used to assess the
        performance of the system.

        Returns
        -------
        nuisance_modes : np.ndarray
            Array of nuisance modes, shape (n_nuisance_modes, wf_npixels, wf_npixels).
        """
        if self.telescope_modes is None:
            raise ValueError("Telescope modes must be provided to calculate nuisance modes.")

        # Calculate the projection of telescope modes onto target modes
        target_modes_expanded = np.expand_dims(self.target_modes, axis=1)
        overlap = np.sum(np.conj(target_modes_expanded) * self.telescope_modes, axis=(-2, -1))
        
        # Calculate the residual modes (nuisance modes)
        residual = self.telescope_modes - np.sum(overlap[..., None, None] * target_modes_expanded, axis=0)

        # Orthonormalize the residual modes using QR decomposition
        q, r = np.linalg.qr(residual.reshape(residual.shape[0], -1))
        nuisance_modes = q.reshape(residual.shape)
        
        return nuisance_modes

    def propagate_all_modes(self) -> np.ndarray:
        """
        Propagate all input modes and nuisance modes through the MPLC system.

        Returns
        -------
        output_fields : np.ndarray
            Array of shape (n_modes + n_nuisance_modes, n_wavelengths, target_coordspec.n, target_coordspec.d)
            containing the output complex fields for each mode and wavelength.
        """
        if self.telescope_modes is not None:
            nuisance_modes = self.calculate_nuisance_modes()
            all_modes = np.concatenate([self.input_modes, nuisance_modes], axis=0)
        else:
            all_modes = self.input_modes

        def propagate_mode(mode: np.ndarray) -> np.ndarray:
            input_layer = InputField(field=mode, array_diameter=self.array_diameter)

            def propagate_wavelength(wavelength: np.ndarray) -> np.ndarray:
                wavefront = dl.Wavefront(wavelength, self.wf_npixels, diameter=self.array_diameter)
                wavefront = input_layer(wavefront)
                wavefront = self.mplc(wavefront)
                return wavefront.phasor

            return eqx.filter_vmap(propagate_wavelength)(self.wavelengths)

        return eqx.filter_vmap(propagate_mode)(all_modes)

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

#This currently doesn't work when dl.CoordSpec is passed in
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