"""
Generic MPLC systems for optical mode propagation and transformation.

Provides the MPLCSystem class, a general-purpose wrapper around MPLC layers
for propagating input optical modes through a multi-plane light converter
and analyzing the output modes against targets. Includes utilities for
stacking multiple systems into batched operations.
"""

import dLux as dl
import equinox as eqx
import jax.numpy as np
from jax import Array
import zodiax as zdx

from ..mplc_model import MPLC, InputField
from ..utils.phase_screen_transforms import bin_phase_screens, mask_to_beam_size


class MPLCSystem(zdx.Base):
    """
    Wraps an `MPLC` layer with the input/target modes and wavelengths needed to
    propagate a full set of modes through the MPLC system.

    Parameters
    ----------
    input_modes
        Array of input complex field amplitudes, shape (n_modes, wf_npixels, wf_npixels).
        Expected to be orthogonal and normalized.
    target_modes
        Array of target complex field amplitudes, shape (n_modes, wf_npixels, wf_npixels).
        Expected to be (close to) orthogonal and normalized.
    aperture_diameter
        Diameter of the input aperture.
    mplc_diameter
        Diameter of the MPLC system.
    array_diameter
        Diameter of the array of the simulation.
    bin_size
        Size of the binning to apply to the phase screens.
    num_planes
        Number of phase screens in the MPLC system.
    total_length
        Total length of the MPLC system (distance from input to output).
    target_coordspec_n
    target_coordspec_d
    wavelengths
        Array of wavelengths to propagate through the MPLC system,
        shape (n_wavelengths,).
    telescope_modes, optional
        Array of telescope modes, shape (n_telescope_modes, wf_npixels, wf_npixels).
    """

    input_modes: Array
    wf_npixels: int = eqx.field(static=True)
    target_modes: Array
    aperture_diameter: float
    mplc_diameter: float
    array_diameter: float
    bin_size: int = eqx.field(static=True)
    num_planes: int = eqx.field(static=True)
    total_length: float = eqx.field(static=True)
    target_coordspec_n: int
    target_coordspec_d: float
    wavelengths: Array
    mplc: MPLC
    telescope_modes: Array | None
    delta_zs: Array

    def __init__(
        self,
        input_modes: Array,
        target_modes: Array,
        aperture_diameter: float,
        mplc_diameter: float,
        array_diameter: float,
        bin_size: int,
        num_planes: int,
        total_length: float,
        target_coordspec_n: int,
        target_coordspec_d: float,
        wavelengths: Array,
        telescope_modes: Array | None = None,
    ) -> None:
        self.input_modes = input_modes
        self.wf_npixels = input_modes.shape[
            1
        ]  # Assuming input_modes has shape (n_modes, wf_npixels, wf_npixels)
        self.target_modes = target_modes
        self.aperture_diameter = aperture_diameter
        self.mplc_diameter = mplc_diameter
        self.array_diameter = array_diameter
        self.bin_size = bin_size
        self.num_planes = num_planes
        self.total_length = total_length
        self.target_coordspec_n = target_coordspec_n
        self.target_coordspec_d = target_coordspec_d
        self.wavelengths = wavelengths
        self.telescope_modes = telescope_modes

        self.delta_zs = (
            np.linspace(0, self.total_length, self.num_planes + 1)[1:]
            - np.linspace(0, self.total_length, self.num_planes + 1)[:-1]
        )

        self.mplc = MPLC(
            phase_screens=np.zeros((self.num_planes, self.wf_npixels, self.wf_npixels)),
            delta_zs=self.delta_zs,
            array_diameter=self.array_diameter,
            target_coordspec_n=self.target_coordspec_n,
            target_coordspec_d=self.target_coordspec_d,
        )

    def with_params(self, params: dict[str, Array]) -> "MPLCSystem":
        """
        Derive the physical phase screens from raw optimizable `params` and
        return a new `MPLCSystem` with them applied.

        Parameters
        ----------
        params
            Raw optimizable parameters, with keys "phase_screens" (pre-bin,
            pre-mask).

        Returns
        -------
            A new instance with the binned and beam-masked phase screens and
            the updated total length set on its `mplc` layer.
        """
        phase_screens = params["phase_screens"]
        n = phase_screens.shape[-1]
        binned_phase_screens = bin_phase_screens(phase_screens, self.bin_size)
        filtered_phase_screens = mask_to_beam_size(
            binned_phase_screens,
            npixels=n,
            diameter=self.mplc_diameter,
            array_diameter=self.array_diameter,
        )

        return self.set("mplc.phase_screens", filtered_phase_screens)

    def get_processed_params(self) -> dict[str, Array]:
        """
        Extract the processed MPLC parameters.

        Returns
        -------
        params
            MPLC parameters as processed by any operations on phase screens and length.
        """
        return {
            "phase_screens": self.mplc.phase_screens,
            "total_length": self.mplc.total_length,
        }

    def calculate_nuisance_modes(self, n_nuisance: int | None = None) -> Array:
        """
        Calculate the nuisance modes of the MPLC system, which are the modes that
        are orthogonal to the target modes and can be used to assess the
        performance of the system.

        Returns
        -------
            Array of nuisance modes, shape (n_nuisance_modes, wf_npixels, wf_npixels).
        """
        if self.telescope_modes is None:
            raise ValueError(
                "Telescope modes must be provided to calculate nuisance modes."
            )

        # Calculate the projection of each telescope mode onto each input mode,
        # keeping the telescope-mode axis (t) and input-mode axis (m) distinct
        overlap = np.einsum(
            "mij,tij->tm", np.conj(self.input_modes), self.telescope_modes
        )

        # Calculate the residual modes (nuisance modes) by removing the span
        # of the input modes
        projection = np.einsum("tm,mij->tij", overlap, self.input_modes)
        residual = self.telescope_modes - projection

        # Orthonormalize the residual modes using QR decomposition.
        flat_residual = residual.reshape(residual.shape[0], -1)
        q, r = np.linalg.qr(flat_residual.T)

        nuisance_modes = q.T.reshape(
            residual.shape[0], self.wf_npixels, self.wf_npixels
        )

        # If n_nuisance is specified, return only the first n_nuisance nuisance modes
        if n_nuisance is not None:
            nuisance_modes = nuisance_modes[:n_nuisance]

        return nuisance_modes

    def propagate_all_modes(self) -> Array:
        """
        Propagate all input modes and nuisance modes through the MPLC system.

        Returns
        -------
            Array of shape (n_modes + n_nuisance_modes, n_wavelengths,
            target_coordspec_n, target_coordspec_n)
            containing the output complex fields for each mode and wavelength.
        """
        if self.telescope_modes is not None:
            nuisance_modes = self.calculate_nuisance_modes()
            all_modes = np.concatenate([self.input_modes, nuisance_modes], axis=0)
        else:
            all_modes = self.input_modes

        def propagate_mode(mode: Array) -> Array:
            input_layer = InputField(field=mode, array_diameter=self.array_diameter)

            def propagate_wavelength(wavelength: Array) -> Array:
                wavefront = dl.Wavefront(
                    wavelength, self.wf_npixels, diameter=self.array_diameter
                )
                wavefront = input_layer(wavefront)
                wavefront = self.mplc(wavefront)
                return wavefront.phasor

            return eqx.filter_vmap(propagate_wavelength)(self.wavelengths)

        return eqx.filter_vmap(propagate_mode)(all_modes)

    def propagate_modes(self, modes: Array) -> Array:
        """
        Propagate the defined modes through the MPLC system.

        Returns
        -------
            Array of shape (n_modes, n_wavelengths,
            target_coordspec_n, target_coordspec_n)
            containing the output complex fields for each mode and wavelength.
        """

        def propagate_mode(mode: Array) -> Array:
            input_layer = InputField(field=mode, array_diameter=self.array_diameter)

            def propagate_wavelength(wavelength: Array) -> Array:
                wavefront = dl.Wavefront(
                    wavelength, self.wf_npixels, diameter=self.array_diameter
                )
                wavefront = input_layer(wavefront)
                wavefront = self.mplc(wavefront)
                return wavefront.phasor

            return eqx.filter_vmap(propagate_wavelength)(self.wavelengths)

        return eqx.filter_vmap(propagate_mode)(modes)

    def __call__(self) -> Array:
        """
        Propagate every input mode, at every wavelength, through the MPLC system.

        Returns
        -------
            Array of shape (n_modes, n_wavelengths,
            target_coordspec_n, target_coordspec_n)
            containing the output complex fields for each mode and wavelength.
        """

        def propagate_mode(mode: Array) -> Array:
            input_layer = InputField(field=mode, array_diameter=self.array_diameter)

            def propagate_wavelength(wavelength: Array) -> Array:
                wavefront = dl.Wavefront(
                    wavelength, self.wf_npixels, diameter=self.array_diameter
                )
                wavefront = input_layer(wavefront)
                wavefront = self.mplc(wavefront)
                return wavefront.phasor

            return eqx.filter_vmap(propagate_wavelength)(self.wavelengths)

        return eqx.filter_vmap(propagate_mode)(self.input_modes)
