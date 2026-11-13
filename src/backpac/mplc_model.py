import dLux as dl
import jax.numpy as np


class InputField(dl.OpticalLayer):
    field: np.ndarray
    array_diameter: float

    def __init__(self, field, array_diameter):
        self.field = field
        self.array_diameter = array_diameter

    def __call__(self, wavefront):
        wavelength = wavefront.wavelength
        wavefront = wavefront.from_phasor(
            self.field, wavelength, diameter=self.array_diameter
        )
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
        Array of phase values for each plane, shape (num_planes, wf_npixels, wf_npixels)
    delta_zs : np.ndarray
        Array of distances between each phase screen, shape (num_planes,).
    array_diameter : float
        Diameter of the array of the simulation.
    target_coordspec_n : int
        Number of pixels across the target plane.
    target_coordspec_d : float
        Physical size of the target plane pixels in metres.
    """

    phase_screens: np.ndarray
    num_planes: int
    wf_npixels: int
    delta_zs: np.ndarray
    array_diameter: float
    target_coordspec_n: int
    target_coordspec_d: float

    def __init__(
        self,
        phase_screens,
        delta_zs,
        array_diameter,
        target_coordspec_n,
        target_coordspec_d,
    ):
        self.phase_screens = np.asarray(phase_screens, dtype=float)
        self.num_planes = self.phase_screens.shape[0]
        self.wf_npixels = self.phase_screens.shape[1]

        self.delta_zs = np.asarray(delta_zs, dtype=float)

        self.array_diameter = array_diameter
        self.target_coordspec_n = target_coordspec_n
        self.target_coordspec_d = target_coordspec_d

    def __call__(self, wavefront):

        for i in range(self.num_planes):
            wavefront = dl.AberratedLayer(phase=self.phase_screens[i])(wavefront)
            if i < self.num_planes - 1:
                between_planes = dl.MFTPropagator(
                    [
                        ("FreeSpace", dl.ABCDFreeSpace(self.delta_zs[i])),
                    ],
                    dl.CoordSpec(
                        n=self.wf_npixels, d=self.array_diameter / self.wf_npixels
                    ),
                )
                wavefront = between_planes(wavefront)

        output_propagator = dl.MFTPropagator(
            [
                ("FreeSpace", dl.ABCDFreeSpace(self.delta_zs[-1])),
            ],
            dl.CoordSpec(n=self.target_coordspec_n, d=self.target_coordspec_d),
        )

        return output_propagator(wavefront)
