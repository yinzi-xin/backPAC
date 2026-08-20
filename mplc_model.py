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

class MPLC(zdx.Base):
    """
    A class representing a multi-plane light conversion (MPLC) system, which
    consists of a sequence of complex-valued phase screens separated by free-space
    propagation.

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
    wavelengths : np.ndarray
        Array of wavelengths to propagate through the MPLC system, shape (n_wavelengths,).

    """
    input_modes: np.ndarray
    target_modes: np.ndarray
    diameter: float
    num_planes: int
    total_length: float
    wavelengths: np.ndarray
    wf_npixels: int
    optics: dl.LayeredOpticalSystem

    def __init__(self, input_modes, target_modes, diameter, num_planes, total_length, wavelengths):
        self.input_modes = input_modes
        self.target_modes = target_modes
        self.diameter = diameter
        self.num_planes = num_planes
        self.total_length = total_length
        self.wavelengths = wavelengths
        self.wf_npixels = input_modes.shape[1]  # Assuming input_modes has shape (n_modes, wf_npixels, wf_npixels)

        # Define the input mode layer
        input_layer = InputField(field=input_modes[0], diameter=self.diameter)  # Use the first input mode as a placeholder for the input layer

        # Define the propagator between phase screens
        between_planes = dl.MFTPropagator(
            [
                ("FreeSpace", dl.ABCDFreeSpace(self.total_length / self.num_planes)),
            ],
            dl.CoordSpec(n=self.wf_npixels, d=self.diameter / self.wf_npixels),
        )

        # Define the optical layers for propagating through the MPLC
        layers_mplc =  [
            ("input_layer", input_layer),
        ]

        for i in range(num_planes):
            phase_screen = dl.AberratedLayer(phase=np.zeros((self.wf_npixels, self.wf_npixels)))
            layers_mplc.append((f"phase_screen_{i}", phase_screen))
            if i < num_planes - 1:
                layers_mplc.append((f"between_planes_{i}", between_planes))

        # Construct the optics object for the MPLC
        self.optics = dl.LayeredOpticalSystem(
            wf_npixels=self.wf_npixels,
            diameter=self.diameter,
            layers=layers_mplc,
        )

    def update_optics(self, mode: np.ndarray, phase_screens: np.ndarray, total_length: float) -> dl.LayeredOpticalSystem:
        """
        Update the optical system with the aperture and phase screen values.

        Parameters
        ----------
        mode : np.ndarray
            Array of shape (wf_npixels, wf_npixels) containing the input mode field
        phase_screens : np.ndarray
            Array of shape (n_planes, wf_npixels, wf_npixels) containing the phase values for each plane.
        total_length : float
            Total length of the MPLC system.

        Returns
        -------
        dl.LayeredOpticalSystem
            Updated optical system with the new phase screen values.
        """
        optics = self.optics.set("input_layer", InputField(field=mode, diameter=self.diameter))

        between_planes = dl.MFTPropagator(
            [
                ("FreeSpace", dl.ABCDFreeSpace(total_length / self.num_planes)),
            ],
            dl.CoordSpec(n=self.wf_npixels, d=self.diameter / self.wf_npixels),
        )

        for i in range(self.num_planes):
            optics = optics.set(
                f"phase_screen_{i}", dl.AberratedLayer(phase=phase_screens[i])
            )
            if i < self.num_planes - 1:
                optics = optics.set(f"between_planes_{i}", between_planes)
        return optics

    def __call__(self, phase_screens: np.ndarray, total_length: float) -> np.ndarray:
        """
        Propagate through the MPLC system for each input mode and return the output complex fields

        Parameters
        ----------
        phase_screens : np.ndarray
            Array of shape (n_planes, wf_npixels, wf_npixels) containing the phase values for each plane.
        total_length : float
            Total length of the MPLC system.

        Returns
        -------
        output_fields : np.ndarray
            Array of shape (n_modes, n_wavelengths, wf_npixels, wf_npixels) containing the output complex fields for each mode and wavelength.
        """

        def propagate_mode(mode: np.ndarray) -> np.ndarray:
            optics = self.update_optics(mode, phase_screens, total_length)
            return optics.propagate(self.wavelengths, return_wf=True).phasor

        return eqx.filter_vmap(propagate_mode)(self.input_modes)