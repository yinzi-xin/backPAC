import cmocean
import numpy as onp

def apply_complex_map(
    field, cmap=cmocean.cm.phase, power=1.0, normalise=True, shift=0.0
):
    """
    Maps a complex-valued 2D array to an RGB image using a colormap.

    The hue is determined by the phase (angle) of each complex value, and the brightness
    is scaled by the magnitude (absolute value) raised to the given power. The result is
    normalized to the [0, 1] range for display.

    Parameters
    ----------
    field : np.ndarray
        2D array of complex values to visualize.
    cmap : callable
        A matplotlib colormap function (e.g., plt.cm.hsv).
    power : float, optional
        Exponent to apply to the magnitude for brightness scaling/gamma (default is 1.0).

    Returns
    -------
    np.ndarray
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