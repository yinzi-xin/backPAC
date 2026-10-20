import equinox as eqx
import zodiax as zdx
import optax
import jax
import jax.numpy as np
import dLux as dl
from .mplc_model import MPLCSystem

class MPLCOptimizer():
    """
    Optimizer for an MPLC system built with dLux.

    Wraps an `MPLCSystem` (which itself wraps the `MPLC` OpticalLayer) and exposes
    loss functions, gradient computation, and an optax training loop.

    Parameters
    ----------
    mplc_system: MPLCSystem
        An instance of the MPLCSystem class, containing the MPLC layer and its parameters.
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
        mplc_system: MPLCSystem,
        wavelengths: np.ndarray,
        wavelength_weights: np.ndarray | None = None,
        mode_weights: np.ndarray | None = None,
    ):
        self.mplc_system = mplc_system
        self.wavelengths = wavelengths
        self.wavelength_weights = (
            wavelength_weights
            if wavelength_weights is not None
            else np.ones(len(wavelengths)) / len(wavelengths)
        )

        self.num_modes = self.mplc_system.input_modes.shape[0]
        self.mode_weights = (
            mode_weights
            if mode_weights is not None
            else np.ones(self.num_modes) / self.num_modes
        )

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def get_spectral_field(self) -> np.ndarray:
        """
        Propagate all wavelengths and return the stack of output complex field
        arrays, shape (n_modes, n_wavelengths, wf_npixels, wf_npixels).
        """
        output = self.mplc_system()
        return output

    def get_spectral_intensity(self) -> np.ndarray:
        """
        Propagate all wavelengths and return the stack of output intensity
        arrays, shape (n_modes, n_wavelengths, wf_npixels, wf_npixels).
        """
        output_field = self.get_spectral_field()
        intensity = np.abs(output_field) ** 2
        return intensity

    def get_broadband_intensity(self) -> np.ndarray:
        """
        Propagate all wavelengths and return the spectrally weighted
        intensity sum, shape (n_modes, wf_npixels, wf_npixels).
        """
        intensity = self.get_spectral_intensity()
        broadband_intensity = np.tensordot(self.wavelength_weights, intensity, axes=1)
        return broadband_intensity

    def get_mode_coupling_matrix(self) -> np.ndarray:
        """
        Return the N_modes x N_modes mode-coupling matrix whose (i, j) entry
        is |<output_i | target_j>|^2, useful for diagnosing crosstalk.
        """
        ...

    # ------------------------------------------------------------------
    # Optimization loop
    # ------------------------------------------------------------------

    def make_optimizer(
        self,
        params: list[str],
        learning_rate: float = 1e-3,
        optimizer: optax.GradientTransformation | None = None,
    ) -> tuple[optax.GradientTransformation, optax.OptState]:
        """
        Initialise an optax optimizer and its state for the given parameter
        paths.

        Parameters
        ----------
        params : list[str]
            zodiax path strings for the parameters to optimize.
        learning_rate : float
            Learning rate passed to Adam when no custom optimizer is supplied.
        optimizer : optax.GradientTransformation | None
            Custom optax optimizer; defaults to ``optax.adam(learning_rate)``.

        Returns
        -------
        optimizer : optax.GradientTransformation
        opt_state : optax.OptState
        """
        ...

    def train_step(
        self,
        params: list[str],
        optimizer: optax.GradientTransformation,
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

    def optimize(
        self,
        params: list[str],
        n_steps: int = 500,
        learning_rate: float = 1e-3,
        optimizer: optax.GradientTransformation | None = None,
        loss_fn: str = "overlap",
        loss_kwargs: dict | None = None,
        log_every: int = 50,
        jit: bool = True,
    ) -> tuple["MPLCOptimizer", list[float]]:
        """
        Run the full optimization loop.

        Parameters
        ----------
        params : list[str]
            zodiax path strings for the parameters to optimize.
        n_steps : int
            Number of gradient steps.
        learning_rate : float
            Learning rate for the default Adam optimizer.
        optimizer : optax.GradientTransformation | None
            Custom optax optimizer; overrides ``learning_rate`` when provided.
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
        optimized_model : MPLCOptimizer
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
        ``mplc_system.mplc.phase_screens``, one per plane.
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
