import equinox as eqx
import zodiax as zdx
import optax
import jax
import jax.numpy as np
import dLux as dl
from mplc_model import MPLC

class MPLCOptimizer():
    """
    Optimizer for an MPLC system built with dLux.

    Wraps a dLux LayeredOpticalSystem representing the MPLC and exposes
    loss functions, gradient computation, and an optax training loop.

    Parameters
    ----------
    mplc: MPLC
        An instance of the MPLC class, containing the optical system and its parameters.
    wavelengths : np.ndarray
        Array of wavelengths (metres) to simulate, shape (n_wavelengths,).
    wavelength_weights : np.ndarray | None
        Per-wavelength spectral weights, shape (n_wavelengths,). Defaults to
        uniform weighting when None.
    mode_weights : np.ndarray | None
        Per-mode weights for the loss function, shape (n_modes,). Defaults to
        uniform weighting when None.
    """

    def __init__(
        self,
        mplc: MPLC,
        wavelengths: np.ndarray,
        wavelength_weights: np.ndarray | None = None,
        mode_weights: np.ndarray | None = None,
    ):
        self.mplc = mplc
        self.wavelengths = wavelengths
        self.wavelength_weights = (
            wavelength_weights
            if wavelength_weights is not None
            else np.ones(len(wavelengths)) / len(wavelengths)
        )
        self.mode_weights = (
            mode_weights
            if mode_weights is not None
            else np.ones(self.mplc.input_modes.shape[0]) / self.mplc.input_modes.shape[0]
        )

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def get_spectral_field(self, phase_screens: np.ndarray) -> np.ndarray:
        """
        Propagate all wavelengths and return the stack of output complex field
        arrays, shape (n_modes, n_wavelengths, wf_npixels, wf_npixels).
        """
        output = self.mplc(phase_screens)
        return output

    def get_spectral_intensity(self, phase_screens: np.ndarray) -> np.ndarray:
        """
        Propagate all wavelengths and return the stack of output intensity
        arrays, shape (n_modes, n_wavelengths, wf_npixels, wf_npixels).
        """
        output_field = self.get_spectral_field(phase_screens)
        intensity = np.abs(output_field) ** 2
        return intensity

    def get_broadband_intensity(self, phase_screens: np.ndarray) -> np.ndarray:
        """
        Propagate all wavelengths and return the spectrally weighted
        intensity sum, shape (n_modes, wf_npixels, wf_npixels).
        """
        intensity = self.get_spectral_intensity(phase_screens)
        broadband_intensity = np.tensordot(self.wavelength_weights, intensity, axes=1)
        return broadband_intensity

    # ------------------------------------------------------------------
    # Loss functions
    # ------------------------------------------------------------------

    def overlap_integral(
        self, output_field: np.ndarray, target_field: np.ndarray
    ) -> float:
        """
        Compute the scalar overlap integral between an output
        complex field and a target complex field.
        """
        overlap_integral = np.abs(np.sum(output_field * np.conj(target_field))) ** 2
        return overlap_integral

    def overlap_loss(self, phase_screens: np.ndarray) -> float:
        """
        Loss based on the mean overlap integral across all target modes.
        Minimising this loss maximises coupling into the target modes.
        """

        output_fields = self.get_spectral_field(phase_screens)

        def per_mode(output_field, target_field, weight):
            return weight * (1 - self.overlap_integral(output_field, target_field))

        per_mode_losses = eqx.filter_vmap(per_mode)(
            output_fields, self.mplc.target_modes, self.mode_weights
        )
        return np.sum(per_mode_losses)

    def intensity_loss(self, phase_screens: np.ndarray) -> float:
        """
        MSE between the simulated output intensity and the target intensity
        summed over all modes and wavelengths.
        """
        ...

    def crosstalk_loss(self, phase_screens: np.ndarray) -> float:
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
