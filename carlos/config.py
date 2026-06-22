"""Contract and CARLOS hyperparameter configuration (Tables 2, 6)."""

from __future__ import annotations

from dataclasses import dataclass

B1_TOLERANCE = 0.05
B1_TRAINING_SEED = 0


def validation_bank_seed(training_seed: int) -> int:
    """Deterministic validation path bank for a benchmark run."""
    return training_seed + 1000


@dataclass
class CarlosConfig:
    """CARLOS contract + algorithm hyperparameters."""

    dim: int = 1
    x0: float = 36.0
    strike: float = 40.0
    T: float = 1.0
    r: float = 0.05
    sigma: float = 0.2
    delta: float = 0.0
    x_min: float = 30.0
    x_max: float = 40.0

    # Stage 1 / solver grid: N=20 steps => dt^(tr,0) = T/20
    num_steps: int = 20

    # Table 6 defaults
    stage1_paths: int = 10_000
    stage1_epochs: int = 5
    rl_training_inputs: int = 10_000

    # Sampling mixture defaults (Eq. 20)
    lambda_exl: float = 0.55
    lambda_plus: float = 0.20
    lambda_minus: float = 0.20
    lambda_ter: float = 0.05
    c_expl: float = 0.25

    # Delayed stopping (Eq. 31)
    c_dlst: float = 1.3
    delay_k: int = 3  # v1 stub fallback

    # Optimizer / training
    batch_size: int = 64
    lr: float = 1e-4
    lr_decay: float = 0.7
    rl_epochs: int = 5

    @property
    def epochs(self) -> int:
        """Alias used by v1 agent wrapper."""
        return self.rl_epochs

    @epochs.setter
    def epochs(self, value: int) -> None:
        self.rl_epochs = value

    # Validation / benchmark
    val_paths: int = 10_000
    target_price: float = 4.592
    target_tolerance: float = B1_TOLERANCE

    # Dev mode: smaller paths for smoke tests (not scored as B1)
    dev_mode: bool = False

    # Grid schedule
    min_dt: float = 1.0 / 150.0
    grid_transition_alpha: float = 0.05

    def __post_init__(self) -> None:
        if self.dev_mode:
            self.stage1_paths = min(self.stage1_paths, 1_000)
            self.rl_training_inputs = min(self.rl_training_inputs, 512)
            self.val_paths = min(self.val_paths, 1_000)

    @property
    def dt(self) -> float:
        return self.T / self.num_steps

    @property
    def deltas(self) -> list[float]:
        return [self.delta] * self.dim

    @property
    def sigmas(self) -> list[float]:
        return [self.sigma] * self.dim

    @property
    def x0s(self) -> list[float]:
        return [self.x0] * self.dim

    def grid_steps_for_level(self, level: int) -> int:
        """Number of steps at grid level b (halving dt each level)."""
        dt_b = self.dt / (2**level)
        return max(1, int(round(self.T / dt_b)))

    def dt_for_level(self, level: int) -> float:
        return self.T / self.grid_steps_for_level(level)

    def max_grid_levels(self) -> int:
        level = 0
        while self.dt_for_level(level) > self.min_dt + 1e-12:
            level += 1
        return level

    def finest_grid_steps(self) -> int:
        return self.grid_steps_for_level(self.max_grid_levels())

    def sampling_weights(self, level: int) -> dict[str, float]:
        """Eq. 25: reallocate exploration to exploitation as grid refines."""
        lam_exl = self.lambda_exl * ((1 - self.c_expl) ** level)
        extra = self.lambda_exl - lam_exl
        return {
            "exl": lam_exl,
            "plus": self.lambda_plus + extra / 2,
            "minus": self.lambda_minus + extra / 2,
            "ter": self.lambda_ter,
        }

    def benchmark_passes(self, price: float) -> bool:
        return abs(price - self.target_price) <= self.target_tolerance


def b1_benchmark() -> CarlosConfig:
    """B1 benchmark preset: Table 2 contract + Table 6 hyperparameters."""
    return CarlosConfig(dev_mode=False)


# Backward compatibility alias
B1Config = CarlosConfig


@dataclass
class RLState:
    """Mutable state across Algorithm 1 loops."""

    grid_level: int = 0
    loop_count: int = 0
    lr: float = 1e-4
    prev_val_reward: float | None = None
