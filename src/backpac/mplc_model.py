import jax.numpy as np
import dLux as dl
import equinox as eqx

class InputField(dl.OpticalLayer):
    field: np.ndarray
    array_diameter: float

    def __init__(self, field, array_diameter):
        self.field = field
        self.array_diameter = array_diameter

    def __call__(self, wavefront):
        wavelength = wavefront.wavelength
        wavefront = wavefront.from_phasor(self.field, wavelength, diameter=self.array_diameter)
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
    array_diameter : float
        Diameter of the array of the simulation.
    wf_npixels : int
        Number of pixels representing the wavefront at each phase screen.
    target_coordspec: dl.CoordSpec
    """
    phase_screens: np.ndarray
    total_length: float
    num_planes: int = eqx.field(static=True)
    array_diameter: float
    wf_npixels: int = eqx.field(static=True)
    target_coordspec: dl.CoordSpec

    def __init__(self, phase_screens, total_length, num_planes, array_diameter, wf_npixels, target_coordspec):
        self.phase_screens = np.asarray(phase_screens, dtype=float)
        self.total_length = np.asarray(total_length, dtype=float)
        self.num_planes = num_planes
        self.array_diameter = array_diameter
        self.wf_npixels = wf_npixels
        self.target_coordspec = target_coordspec

    def __call__(self, wavefront):
        between_planes = dl.MFTPropagator(
            [
                ("FreeSpace", dl.ABCDFreeSpace(self.total_length / self.num_planes)),
            ],
            dl.CoordSpec(n=self.wf_npixels, d=self.array_diameter / self.wf_npixels),
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