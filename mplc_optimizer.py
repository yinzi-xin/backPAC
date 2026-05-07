import zodiax as zdx
import optax
import jax
import jax.numpy as np
import dLux as dl


class MPLCOptimizer(zdx.Base):
    """
    Optimizer for an MPLC system built with dLux.

    Wraps a dLux LayeredOpticalSystem representing the MPLC and exposes
    loss functions, gradient computation, and an optax training loop.
    Inherits from zdx.Base so the system is a JAX pytree and parameters
    can be selected by path string for optax-based gradient updates.

    Parameters
    ----------
    optics_mplc : dl.LayeredOpticalSystem
        The MPLC optical system containing the phase screens to optimise.
    input_modes : np.ndarray
        Array of input complex field amplitudes, shape (n_modes, npix, npix).
    target_modes : np.ndarray
        Array of target complex field amplitudes, shape (n_modes, npix, npix).
    wavelengths : np.ndarray
        Array of wavelengths (metres) to simulate, shape (n_wavelengths,).
    weights : np.ndarray | None
        Per-wavelength spectral weights, shape (n_wavelengths,). Defaults to
        uniform weighting when None.
    """

    optics_mplc: dl.LayeredOpticalSystem
    input_modes: np.ndarray
    target_modes: np.ndarray
    wavelengths: np.ndarray
    weights: np.ndarray

    def __init__(
        self,
        optics_mplc: dl.LayeredOpticalSystem,
        input_modes: np.ndarray,
        target_modes: np.ndarray,
        wavelengths: np.ndarray,
        weights: np.ndarray | None = None,
    ):
        self.optics_mplc = optics_mplc
        self.input_modes = input_modes
        self.target_modes = target_modes
        self.wavelengths = wavelengths
        self.weights = (
            weights
            if weights is not None
            else np.ones(len(wavelengths)) / len(wavelengths)
        )

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def get_monochromatic_field(self, wavelength: float) -> np.ndarray:
        """
        Propagate a single wavelength through the MPLC and return the
        complex output field amplitude array.
        """
        wf = self.optics_mplc.propagate_mono(wavelength, return_wf=True)
        return wf.phasor

    def get_monochromatic_intensity(self, wavelength: float) -> np.ndarray:
        """
        Propagate a single wavelength through the MPLC and return the
        output intensity (PSF) array.
        """
        intensity = self.optics_mplc.propagate_mono(wavelength)
        return intensity

    def get_spectral_field(self) -> np.ndarray:
        """
        Propagate all wavelengths and return the stack of output complex field
        arrays, shape (n_wavelengths, npix, npix).
        """
        wfs = self.optics_mplc.propagate(self.wavelengths, return_wf=True)
        return wfs.phasor

    def get_spectral_intensity(self) -> np.ndarray:
        """
        Propagate all wavelengths and return the stack of output intensity
        arrays, shape (n_wavelengths, npix, npix).
        """
        intensity = self.optics_mplc.propagate(self.wavelengths)
        return intensity

    def get_broadband_intensity(self) -> np.ndarray:
        """
        Propagate all wavelengths and return the spectrally weighted
        intensity sum, shape (npix, npix).
        """
        intensity = self.get_spectral_intensity()
        broadband_intensity = np.tensordot(self.weights, intensity, axes=1)
        return broadband_intensity

    # ------------------------------------------------------------------
    # Loss functions
    # ------------------------------------------------------------------

    def overlap_integral(
        self, output_field: np.ndarray, target_field: np.ndarray
    ) -> float:
        """
        Compute the normalised scalar overlap integral between an output
        complex field and a target complex field.

        Returns a value in [0, 1], where 1 is perfect overlap.
        """
        ...

    def overlap_loss(self) -> float:
        """
        Loss based on the mean overlap integral across all target modes.
        Minimising this loss maximises coupling into the target modes.
        """
        ...

    def intensity_loss(self) -> float:
        """
        MSE between the simulated output intensity and the target intensity
        summed over all modes and wavelengths.
        """
        ...

    def crosstalk_loss(self) -> float:
        """
        Penalise power leaking into off-diagonal (undesired) mode pairs.
        Computed from the mode-coupling matrix |<output_i | target_j>|^2.
        """
        ...

    def total_variation_regularisation(self) -> float:
        """
        Total variation regularisation on the phase screens to encourage
        smooth, fabrication-friendly phase profiles.
        """
        ...

    def combined_loss(
        self,
        alpha: float = 1.0,
        beta: float = 0.0,
        gamma: float = 0.0,
        delta: float = 0.0,
    ) -> float:
        """
        Weighted combination of the available loss terms:
            L = alpha * overlap_loss
              + beta  * intensity_loss
              + gamma * crosstalk_loss
              + delta * total_variation_regularisation

        Parameters
        ----------
        alpha, beta, gamma, delta : float
            Scalar weights for each loss term.
        """
        ...

    # ------------------------------------------------------------------
    # Gradient utilities
    # ------------------------------------------------------------------

    def get_grads(
        self,
        params: list[str],
        loss_fn: str = "overlap",
        loss_kwargs: dict | None = None,
    ) -> tuple["MPLCOptimizer", float]:
        """
        Compute gradients of the chosen loss with respect to the parameters
        at the given zodiax path strings.

        Parameters
        ----------
        params : list[str]
            zodiax path strings identifying which leaves to differentiate,
            e.g. ["optics_mplc.phase_screen_0.phase", ...].
        loss_fn : str
            Which loss to differentiate: "overlap", "intensity", "crosstalk",
            or "combined".
        loss_kwargs : dict | None
            Extra keyword arguments forwarded to the loss function (e.g.
            alpha, beta weights for "combined").

        Returns
        -------
        grads : MPLCOptimizer
            A pytree of the same structure with gradient values at the
            selected leaves.
        loss_value : float
            Scalar loss evaluated at the current parameters.
        """
        ...

    def mode_coupling_matrix(self) -> np.ndarray:
        """
        Return the N_modes x N_modes mode-coupling matrix whose (i, j) entry
        is |<output_i | target_j>|^2, useful for diagnosing crosstalk.
        """
        ...

    # ------------------------------------------------------------------
    # Optimisation loop
    # ------------------------------------------------------------------

    def make_optimiser(
        self,
        params: list[str],
        learning_rate: float = 1e-3,
        optimiser: optax.GradientTransformation | None = None,
    ) -> tuple[optax.GradientTransformation, optax.OptState]:
        """
        Initialise an optax optimiser and its state for the given parameter
        paths.

        Parameters
        ----------
        params : list[str]
            zodiax path strings for the parameters to optimise.
        learning_rate : float
            Learning rate passed to Adam when no custom optimiser is supplied.
        optimiser : optax.GradientTransformation | None
            Custom optax optimiser; defaults to ``optax.adam(learning_rate)``.

        Returns
        -------
        optimiser : optax.GradientTransformation
        opt_state : optax.OptState
        """
        ...

    def train_step(
        self,
        params: list[str],
        optimiser: optax.GradientTransformation,
        opt_state: optax.OptState,
        loss_fn: str = "overlap",
        loss_kwargs: dict | None = None,
    ) -> tuple["MPLCOptimizer", optax.OptState, float]:
        """
        Perform a single gradient-descent update step.

        Returns
        -------
        updated_model : MPLCOptimizer
            Model with updated parameters.
        new_opt_state : optax.OptState
        loss_value : float
        """
        ...

    def optimise(
        self,
        params: list[str],
        n_steps: int = 500,
        learning_rate: float = 1e-3,
        optimiser: optax.GradientTransformation | None = None,
        loss_fn: str = "overlap",
        loss_kwargs: dict | None = None,
        log_every: int = 50,
        jit: bool = True,
    ) -> tuple["MPLCOptimizer", list[float]]:
        """
        Run the full optimisation loop.

        Parameters
        ----------
        params : list[str]
            zodiax path strings for the parameters to optimise.
        n_steps : int
            Number of gradient steps.
        learning_rate : float
            Learning rate for the default Adam optimiser.
        optimiser : optax.GradientTransformation | None
            Custom optax optimiser; overrides ``learning_rate`` when provided.
        loss_fn : str
            Loss function to minimise: "overlap", "intensity", "crosstalk",
            or "combined".
        loss_kwargs : dict | None
            Extra kwargs forwarded to the loss function.
        log_every : int
            Print loss every this many steps.
        jit : bool
            Whether to JIT-compile ``train_step``.

        Returns
        -------
        optimised_model : MPLCOptimizer
            The model after ``n_steps`` updates.
        loss_history : list[float]
            Scalar loss recorded at every step.
        """
        ...

    # ------------------------------------------------------------------
    # Diagnostics / visualisation helpers
    # ------------------------------------------------------------------

    def get_phase_screens(self) -> list[np.ndarray]:
        """
        Extract and return a list of the current phase screen arrays from
        ``optics_mplc``, one per plane.
        """
        ...

    def eval_efficiency(self) -> float:
        """
        Compute the total coupling efficiency: fraction of input power
        successfully directed into the target modes.
        """
        ...

    def summary(self) -> None:
        """
        Print a human-readable summary of the system configuration and
        current loss values.
        """
        ...
