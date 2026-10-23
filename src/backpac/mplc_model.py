import jax.numpy as np
import equinox as eqx
import zodiax as zdx
import dLux as dl

class InputField(dl.OpticalLayer):
    field: np.ndarray
    diameter: float

    def __init__(self, field, diameter):
        self.field = field
        self.diameter = diameter

    def __call__(self, wavefront):
        wavelength = wavefront.wavelength
        wavefront = wavefront.from_phasor(self.field, wavelength, diameter=self.diameter)
        wavefront = wavefront.normalise()
        return wavefront

class MPLC(dl.OpticalLayer):
    """
    A dLux OpticalLayer representing a multi-plane light conversion (MPLC) system:
    a sequence of (zero thickness) complex-valued phase screens separated by
    free-space propagation.

    Parameters
    ----------
    phase_screens : np.ndarray
        Array of phase values for each plane, shape (num_planes, wf_npixels, wf_npixels).
    total_length : float
        Total length of the MPLC system (distance from input to output).
    num_planes : int
        Number of phase screens in the MPLC system.
    diameter : float
    wf_npixels : int
        Number of pixels representing the wavefront at each phase screen.
    target_coordspec: dl.CoordSpec
    """
    phase_screens: np.ndarray
    total_length: float
    num_planes: int
    diameter: float
    wf_npixels: int
    target_coordspec: dl.CoordSpec

    def __init__(self, phase_screens, total_length, num_planes, diameter, wf_npixels, target_coordspec):
        self.phase_screens = np.asarray(phase_screens, dtype=float)
        self.total_length = np.asarray(total_length, dtype=float)
        self.num_planes = num_planes
        self.diameter = diameter
        self.wf_npixels = wf_npixels
        self.target_coordspec = target_coordspec

    def __call__(self, wavefront):
        between_planes = dl.MFTPropagator(
            [
                ("FreeSpace", dl.ABCDFreeSpace(self.total_length / self.num_planes)),
            ],
            dl.CoordSpec(n=self.wf_npixels, d=self.diameter / self.wf_npixels),
        )

        for i in range(self.num_planes):
            wavefront = dl.AberratedLayer(phase=self.phase_screens[i])(wavefront)
            if i < self.num_planes - 1:
                wavefront = between_planes(wavefront)

        output_propagator = dl.MFTPropagator(
            [
                ("FreeSpace", dl.ABCDFreeSpace(self.total_length / self.num_planes)),
            ],
            self.target_coordspec,
        )

        return output_propagator(wavefront)


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
    target_coordspec : dl.CoordSpec
        Coordinate specification for the target plane.
    wavelengths : np.ndarray
        Array of wavelengths to propagate through the MPLC system, shape (n_wavelengths,).
    """
    input_modes: np.ndarray
    target_modes: np.ndarray
    diameter: float
    wavelengths: np.ndarray
    wf_npixels: int
    mplc: MPLC

    def __init__(self, input_modes, target_modes, diameter, num_planes, total_length, target_coordspec, wavelengths):
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
            target_coordspec=target_coordspec,
        )

    def __call__(self) -> np.ndarray:
        """
        Propagate every input mode, at every wavelength, through the MPLC system.

        Returns
        -------
        output_fields : np.ndarray
            Array of shape (n_modes, n_wavelengths, target_coordspec.n, target_coordspec.n)
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