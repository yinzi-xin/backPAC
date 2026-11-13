import dLux.utils as dlu
import jax
import jax.numpy as np


def bin_phase_screens(phase_screens, bin_size=2):
    """Bin phase screens into coarser pixel blocks by local averaging.

    Each `bin_size x bin_size` block of simulation pixels is averaged into
    one coarse pixel, then the result is repeated back up to the original
    resolution so adjacent pixels within a block share the same value.

    Parameters
    ----------
    phase_screens : ndarray
        Array of phase screens, shaped `(num_planes, n, n)`.
    bin_size : int, optional
        Side length of the square pixel block to average over, by default 2.

    Returns
    -------
    ndarray
        Phase screens of the same shape as `phase_screens`, binned by
        `bin_size`.
    """
    num_planes, n, _ = phase_screens.shape
    assert n % bin_size == 0, (
        "wf_npixels must be divisible by bin_size to bin"
        + " into bin_size x bin_size LC pixel blocks"
    )

    lc_phase_screens = phase_screens.reshape(
        num_planes, n // bin_size, bin_size, n // bin_size, bin_size
    ).mean(axis=(2, 4))
    binned_phase_screens = np.repeat(
        np.repeat(lc_phase_screens, bin_size, axis=1), bin_size, axis=2
    )

    return binned_phase_screens


def mask_to_beam_size(phase_screens, npixels, diameter, array_diameter, oversample=5):
    """Zero out phase screen values outside the beam aperture.

    A single circular aperture mask is computed and applied to every plane in
    `phase_screens`.

    Parameters
    ----------
    phase_screens : ndarray
        Array of phase screens.
    npixels : int
        Number of pixels across the array.
    diameter : float
        Diameter of the circular aperture (beam), in metres.
    array_diameter : float
        Diameter of the full array, in metres.
    oversample : int
        Oversampling factor used when generating the aperture mask.

    Returns
    -------
    ndarray
        `phase_screens` with values outside the aperture set to zero.
    """
    mask = dlu.circular_aperture(
        npixels=npixels,
        diameter=diameter,
        oversample=oversample,
        array_diameter=array_diameter,
    )
    masked_phase_screens = phase_screens * mask

    return masked_phase_screens


def _shift(image, shift):
    """Shift a 2D image by a (possibly non-integer) number of pixels
    using the Fourier shift theorem."""
    ny, nx = image.shape
    fy, fx = np.meshgrid(np.fft.fftfreq(ny), np.fft.fftfreq(nx), indexing="ij")
    phase_ramp = np.exp(-2j * np.pi * (fy * shift[0] + fx * shift[1]))
    return np.real(np.fft.ifft2(np.fft.fft2(image) * phase_ramp))


def apply_random_shifts(phase_screens, key, sigma_pix=1.0):
    """Apply random shifts in X and Y to each phase screen in the array.

    Parameters
    ----------
    phase_screens : ndarray
        Array of phase screens, shaped `(num_planes, n, n)`.
    key : jax.random.PRNGKey
        PRNG key used to generate the random shifts.
    sigma_pix : float, optional
        Standard deviation of the random shifts in pixels, by default 1.0.

    Returns
    -------
    ndarray
        Phase screens with random shifts applied.
    """
    num_planes, n, _ = phase_screens.shape
    shifts = sigma_pix * jax.random.normal(key, shape=(num_planes, 2))

    shifted_phase_screens = jax.vmap(_shift)(phase_screens, shifts)

    return shifted_phase_screens
