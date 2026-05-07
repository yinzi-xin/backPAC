import jax.numpy as np
import dLux as dl

class ComplexAperture(dl.OpticalLayer):
    '''
    A custom dLux optical layer that applies a complex-valued field

    Parameters
    ----------
    field : np.ndarray
        Complex-valued array representing the aperture field, shape (npix, npix).
    '''
    field: np.ndarray

    def __init__(self, field):
        self.field = field

    def __call__(self):
        return self.field