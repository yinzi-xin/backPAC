"""
Optical layer definitions for multi-plane light conversion (MPLC) systems.

Defines dLux optical layer components for constructing MPLC systems,
including the input field generator and the MPLC propagation layer
with phase screens and free-space propagation between planes.
"""

import dLux as dl
import jax.numpy as np
from jax import Array


class InputField(dl.OpticalLayer):
    field: Array
    array_diameter: float

    def __init__(self, field: Array, array_diameter: float) -> None:
        self.field = field
        self.array_diameter = array_diameter

    def __call__(self, wavefront: dl.Wavefront) -> dl.Wavefront:
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
    phase_screens
        Array of phase values for each plane, shape (num_planes, wf_npixels, wf_npixels)
    delta_zs
        Array of distances between each phase screen, shape (num_planes,).
    array_diameter
        Diameter of the array of the simulation.
    target_coordspec_n
        Number of pixels across the target plane.
    target_coordspec_d
        Physical size of the target plane pixels in metres.
    windowing_fwhm
        Full width half maximum of the super Gaussian windowing function applied to the phase screens, in metres.
        If None, no windowing is applied.
    """

    phase_screens: Array
    num_planes: int
    wf_npixels: int
    delta_zs: Array
    array_diameter: float
    target_coordspec_n: int
    target_coordspec_d: float
    windowing_fwhm: float

    def __init__(
        self,
        phase_screens: Array,
        delta_zs: Array,
        array_diameter: float,
        target_coordspec_n: int,
        target_coordspec_d: float,
        windowing_fwhm: float = None,
    ) -> None:
        self.phase_screens = np.asarray(phase_screens, dtype=np.float32)
        self.num_planes = self.phase_screens.shape[0]
        self.wf_npixels = self.phase_screens.shape[1]

        self.delta_zs = np.asarray(delta_zs, dtype=np.float32)

        self.array_diameter = array_diameter
        self.target_coordspec_n = target_coordspec_n
        self.target_coordspec_d = target_coordspec_d
        self.windowing_fwhm = windowing_fwhm

    def __call__(self, wavefront: dl.Wavefront) -> dl.Wavefront:
        if self.windowing_fwhm is not None:
            window = make_super_gaussian_aperture(
                (self.wf_npixels, self.wf_npixels),
                self.array_diameter,
                self.windowing_fwhm,
                dtype=self.phase_screens.dtype,
            )
        else:
            window = None

        for i in range(self.num_planes):
            if window is not None:
                self.phase_screens[i] *= window
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


def make_super_gaussian_aperture(
    wf_shape,
    array_diameter,
    full_width_half_maximum,
    shape_parameter=5,
    center=None,
    dtype=None,
):
    """Make a super Gaussian aperture.

    Parameters

    ----------
    wf_shape
        The shape of the aperture
    array_diameter
        The physical size of the array in metres.
    full_width_half_maximum
        The Full Width Half Maximum of the super Gaussian aperture in metres.
    shape_parameter
        The shape parameter determines the exponential Gaussian fall off. p=1 is identical to a normal Gaussian and p=0.5 results in a Laplacian distribution.
    dtype
        Data type for output array. Defaults to np.float32.

    Returns
    -------
    super_gaussian_aperture
        The super Gaussian aperture.
    """
    if dtype is None:
        dtype = np.float32

    if center is None:
        shift = np.zeros(2, dtype=dtype)
    else:
        shift = center * np.ones(2, dtype=dtype)

    sigma = (
        0.5 * full_width_half_maximum / (2 * np.log(2)) ** (1 / (2 * shape_parameter))
    )
    grid = np.meshgrid(
        np.linspace(-array_diameter / 2, array_diameter / 2, wf_shape[0], dtype=dtype),
        np.linspace(-array_diameter / 2, array_diameter / 2, wf_shape[1], dtype=dtype),
        indexing="ij",
    )
    r = np.sqrt((grid[0] - shift[0]) ** 2 + (grid[1] - shift[1]) ** 2)
    super_gaussian_aperture = np.exp(-((r / sigma) ** (2 * shape_parameter)))
    return super_gaussian_aperture.astype(dtype)
