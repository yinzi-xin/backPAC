"""
Utilities for visualizing complex-valued fields as RGB images.

Provides functions to map complex-valued arrays to RGB colormap representations,
where hue is determined by phase and brightness by magnitude.
"""

from collections.abc import Callable

import cmocean
import numpy as onp


def apply_complex_map(
    field: onp.ndarray,
    cmap: Callable[[onp.ndarray], onp.ndarray] = cmocean.cm.phase,
    power: float = 1.0,
    normalise: bool = True,
    shift: float = 0.0,
) -> onp.ndarray:
    """
    Maps a complex-valued 2D array to an RGB image using a colormap.

    The hue is determined by the phase (angle) of each complex value, and the brightness
    is scaled by the magnitude (absolute value) raised to the given power. The result is
    normalized to the [0, 1] range for display.

    Parameters
    ----------
    field
        2D array of complex values to visualize.
    cmap
        A matplotlib colormap function (e.g., plt.cm.hsv).
    power, optional
        Exponent to apply to the magnitude for brightness scaling/gamma (default is 1.0)

    Returns
    -------
        3D array representing the RGB image (shape: field.shape + (3,)).
    """
    angles = onp.angle(field)
    angles_norm = onp.mod(angles + shift, 2 * onp.pi) / (2 * onp.pi)

    if normalise:
        amp = onp.abs(field) ** power / onp.max(onp.abs(field) ** power)
    else:
        amp = onp.abs(field) ** power

    img = cmap(angles_norm) * (amp)[:, :, None]

    if normalise:
        img = img[..., :3] / onp.max(img[..., 0:3])
    else:
        img = img[..., :3]

    return img
