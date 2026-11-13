"""
High-level optimizer for multi-plane light conversion (MPLC) systems.

Provides the MPLCOptimizer class for driving joint optimization of one or more
MPLC systems using JAX and Optax, with support for batched optimization,
spectral weighting, and mode-specific weights.
"""

from typing import Sequence

import equinox as eqx
import jax
import jax.numpy as np
import optax

from .systems.generic import MPLCSystem, stack_systems


class MPLCOptimizer:
    """
    Optimizer for one or more MPLC systems built with dLux.

    Wraps a sequence of `MPLCSystem`s (even a single one, as a length-1
    sequence), stacking them into one batched pytree and driving all forward
    passes with `eqx.filter_vmap`, so the same class covers both single- and
    multi-system optimization.

    Parameters
    ----------
    mplc_systems : Sequence[MPLCSystem]
        MPLC systems to optimize jointly. Must all share the same pytree
        structure (shapes, num_planes, etc.), since they are stacked into a
        single batched `MPLCSystem` via `stack_systems`.
    params : Sequence[dict] | None
        Raw optimizable parameters ("phase_screens", "total_length") for each
        system, pre-bin and pre-mask. Defaults to each system's current
        `mplc.phase_screens` / `mplc.total_length` when None.
    wavelength_weights : np.ndarray | None
        Per-wavelength spectral weights, shape (n_wavelengths,). Defaults to
        uniform weighting when None. Assumes all systems share the same
        wavelengths (taken from the first system).
    mode_weights : np.ndarray | None
        Per-mode weights for the loss function, shape (n_modes,). Defaults to
        uniform weighting when None.
    """

    def __init__(
        self,
        mplc_systems: Sequence[MPLCSystem],
        params: Sequence[dict] | None = None,
        wavelength_weights: np.ndarray | None = None,
        mode_weights: np.ndarray | None = None,
    ):
        self.mplc_systems = list(mplc_systems)
        self.batched_system = stack_systems(self.mplc_systems)

        if params is None:
            params = [
                {
                    "phase_screens": system.mplc.phase_screens,
                    "total_length": system.mplc.total_length,
                }
                for system in self.mplc_systems
            ]
        self.params = jax.tree_util.tree_map(lambda *leaves: np.stack(leaves), *params)

        # assumes every system shares the same wavelengths
        self.wavelengths = self.mplc_systems[0].wavelengths
        self.wavelength_weights = (
            wavelength_weights
            if wavelength_weights is not None
            else np.ones(len(self.wavelengths)) / len(self.wavelengths)
        )

        self.num_modes = self.mplc_systems[0].input_modes.shape[0]
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
        Apply the current raw `params` to every system, propagate all modes
        and wavelengths, and return the stack of output complex field arrays,
        shape (n_systems, n_modes, n_wavelengths, wf_npixels, wf_npixels).
        """
        configured_system = eqx.filter_vmap(MPLCSystem.with_params)(
            self.batched_system, self.params
        )
        output = eqx.filter_vmap(lambda system: system())(configured_system)
        return output

    def get_spectral_intensity(self) -> np.ndarray:
        """
        Propagate all wavelengths and return the stack of output intensity
        arrays, shape (n_systems, n_modes, n_wavelengths, wf_npixels, wf_npixels).
        """
        output_field = self.get_spectral_field()
        intensity = np.abs(output_field) ** 2
        return intensity

    def get_broadband_intensity(self) -> np.ndarray:
        """
        Propagate all wavelengths and return the spectrally weighted
        intensity sum, shape (n_systems, n_modes, wf_npixels, wf_npixels).
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
        paths into ``self.params`` (the batched raw phase_screens/total_length).

        Parameters
        ----------
        params : list[str]
            zodiax path strings, relative to ``self.params``, for the
            parameters to optimize.
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
        Perform a single gradient-descent update step: apply ``self.params``
        to ``self.batched_system`` via `MPLCSystem.with_params` (vmapped),
        evaluate ``loss_fn`` (e.g. `loss_WFM`) on the configured batch, and
        update ``self.params`` with the resulting gradients.

        Returns
        -------
        updated_model : MPLCOptimizer
            Model with updated ``params``.
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
        Run the full optimization loop by repeatedly calling `train_step`.

        Parameters
        ----------
        params : list[str]
            zodiax path strings, relative to ``self.params``, for the
            parameters to optimize.
        n_steps : int
            Number of gradient steps.
        learning_rate : float
            Learning rate for the default Adam optimizer.
        optimizer : optax.GradientTransformation | None
            Custom optax optimizer; overrides ``learning_rate`` when provided.
        loss_fn : str
            Loss function to minimise: "overlap" (`loss_WFM`), "intensity",
            "crosstalk", or "combined".
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
        Extract and return a list of the current, per-system phase screen
        arrays obtained by applying ``self.params`` to ``self.batched_system``
        (i.e. ``MPLCSystem.with_params(...).mplc.phase_screens`` for each
        system in the batch).
        """
        ...

    def eval_efficiency(self) -> float:
        """
        Compute the total coupling efficiency, averaged over all systems in
        the batch: fraction of input power successfully directed into the
        target modes.
        """
        ...

    def summary(self) -> None:
        """
        Print a human-readable summary of each system's configuration and
        current loss values.
        """
        ...
