"""Experimental conditional-density API for emulator-based BackPop selection.

The classes and functions in this module are prototypes. They are not used by
the production BackPop scripts and are not validated for astrophysical
inference. The placeholder density model is intentionally simple so future work
can swap in a neural spline flow or density-ratio estimator behind the same
interface.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence

import numpy as np

DEFAULT_LAMBDA_NAMES = ("alpha_ce", "f_acc", "sigma_kick")
DEFAULT_OBSERVABLE_NAMES = ("mc", "q", "z_merge", "theta_latent")


@dataclass(frozen=True)
class CosmicInjectionCatalog:
    """Container for COSMIC-style emulator training data."""

    lambda_features: np.ndarray
    observables: np.ndarray
    lambda_names: tuple[str, ...] = DEFAULT_LAMBDA_NAMES
    observable_names: tuple[str, ...] = DEFAULT_OBSERVABLE_NAMES
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        lambda_features = np.asarray(self.lambda_features, dtype=float)
        observables = np.asarray(self.observables, dtype=float)
        if lambda_features.ndim != 2:
            raise ValueError("lambda_features must be a two-dimensional array")
        if observables.ndim != 2:
            raise ValueError("observables must be a two-dimensional array")
        if len(lambda_features) != len(observables):
            raise ValueError("lambda_features and observables must have the same row count")
        if lambda_features.shape[1] != len(self.lambda_names):
            raise ValueError("lambda_names length must match lambda_features columns")
        if observables.shape[1] != len(self.observable_names):
            raise ValueError("observable_names length must match observables columns")
        object.__setattr__(self, "lambda_features", lambda_features)
        object.__setattr__(self, "observables", observables)

    def save_npz(self, path: str | Path) -> None:
        """Write the catalog to a compact `.npz` file."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            lambda_features=self.lambda_features,
            observables=self.observables,
            lambda_names=np.asarray(self.lambda_names),
            observable_names=np.asarray(self.observable_names),
            metadata=json.dumps(dict(self.metadata or {})),
        )


class ConditionalDensityEmulator(Protocol):
    """Interface future density estimators should implement."""

    observable_names: tuple[str, ...]

    def sample(
        self,
        lambda_features: Sequence[float] | np.ndarray,
        n_samples: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Draw observables from p(observables | Lambda)."""


def featurize_hyperparameters(
    hyperparameters: Mapping[str, float],
    names: Sequence[str] = DEFAULT_LAMBDA_NAMES,
) -> np.ndarray:
    """Convert named hyperparameters into a stable numeric feature vector."""

    missing = [name for name in names if name not in hyperparameters]
    if missing:
        raise KeyError(f"Missing hyperparameters: {missing}")
    return np.asarray([hyperparameters[name] for name in names], dtype=float)


def load_cosmic_injection_catalog(path: str | Path) -> CosmicInjectionCatalog:
    """Load a COSMIC-style emulator catalog from `.npz`.

    Real COSMIC catalogs should be converted into this schema before training.
    """

    with np.load(path, allow_pickle=False) as data:
        metadata_raw = str(data["metadata"]) if "metadata" in data else "{}"
        return CosmicInjectionCatalog(
            lambda_features=data["lambda_features"],
            observables=data["observables"],
            lambda_names=tuple(str(x) for x in (data["lambda_names"] if "lambda_names" in data else DEFAULT_LAMBDA_NAMES)),
            observable_names=tuple(str(x) for x in (data["observable_names"] if "observable_names" in data else DEFAULT_OBSERVABLE_NAMES)),
            metadata=json.loads(metadata_raw),
        )


def generate_cosmic_injection_catalog(
    path: str | Path,
    n_simulations: int = 2_000,
    rng: np.random.Generator | None = None,
) -> CosmicInjectionCatalog:
    """Generate a fake COSMIC-like catalog for emulator API development.

    This is a deterministic toy stand-in, not a COSMIC runner. Replace this
    function with a wrapper around `cosmic-popsynth` for real training data.
    """

    rng = np.random.default_rng() if rng is None else rng
    alpha_ce = rng.uniform(0.2, 5.0, n_simulations)
    f_acc = rng.uniform(0.05, 0.95, n_simulations)
    sigma_kick = rng.uniform(20.0, 300.0, n_simulations)
    lambda_features = np.column_stack([alpha_ce, f_acc, sigma_kick])

    mc = 12.0 + 5.5 * np.log1p(alpha_ce) + 10.0 * f_acc + rng.normal(0.0, 2.0, n_simulations)
    q = np.clip(0.35 + 0.45 * f_acc - 0.0005 * sigma_kick + rng.normal(0.0, 0.08, n_simulations), 0.05, 1.0)
    z_merge = np.clip(0.05 + 0.04 * alpha_ce + 0.0015 * sigma_kick + rng.normal(0.0, 0.05, n_simulations), 0.0, None)
    theta_latent = rng.normal(np.log1p(alpha_ce) - f_acc, 0.3, n_simulations)
    observables = np.column_stack([mc, q, z_merge, theta_latent])

    catalog = CosmicInjectionCatalog(
        lambda_features=lambda_features,
        observables=observables,
        metadata={"warning": "fake COSMIC-like data for emulator prototyping only"},
    )
    catalog.save_npz(path)
    return catalog


@dataclass
class GaussianConditionalEmulator:
    """Linear-mean multivariate Gaussian placeholder emulator."""

    coefficients: np.ndarray
    residual_covariance: np.ndarray
    lambda_names: tuple[str, ...] = DEFAULT_LAMBDA_NAMES
    observable_names: tuple[str, ...] = DEFAULT_OBSERVABLE_NAMES

    @classmethod
    def fit(cls, catalog: CosmicInjectionCatalog, ridge: float = 1.0e-6) -> "GaussianConditionalEmulator":
        design = np.column_stack([np.ones(len(catalog.lambda_features)), catalog.lambda_features])
        coefficients, *_ = np.linalg.lstsq(design, catalog.observables, rcond=None)
        residuals = catalog.observables - design @ coefficients
        covariance = np.cov(residuals, rowvar=False) + ridge * np.eye(catalog.observables.shape[1])
        return cls(coefficients, covariance, catalog.lambda_names, catalog.observable_names)

    def sample(
        self,
        lambda_features: Sequence[float] | np.ndarray,
        n_samples: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        rng = np.random.default_rng() if rng is None else rng
        features = np.asarray(lambda_features, dtype=float)
        if features.shape != (len(self.lambda_names),):
            raise ValueError(f"Expected {len(self.lambda_names)} lambda features, got shape {features.shape}")
        mean = np.r_[1.0, features] @ self.coefficients
        draws = rng.multivariate_normal(mean, self.residual_covariance, size=n_samples)
        draws[:, self.observable_names.index("q")] = np.clip(draws[:, self.observable_names.index("q")], 0.0, 1.0)
        draws[:, self.observable_names.index("z_merge")] = np.clip(draws[:, self.observable_names.index("z_merge")], 0.0, None)
        return draws


def sample_predicted_mergers(
    emulator: ConditionalDensityEmulator,
    hyperparameters: Mapping[str, float],
    n_samples: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample emulator-predicted merger observables for named hyperparameters."""

    features = featurize_hyperparameters(hyperparameters)
    return emulator.sample(features, n_samples=n_samples, rng=rng)


def estimate_selection_alpha(
    emulator: ConditionalDensityEmulator,
    hyperparameters: Mapping[str, float],
    pdet: Callable[[np.ndarray], np.ndarray],
    n_samples: int = 10_000,
    rng: np.random.Generator | None = None,
) -> float:
    """Estimate detectable fraction from emulator samples and a pdet callable."""

    samples = sample_predicted_mergers(emulator, hyperparameters, n_samples, rng=rng)
    probabilities = np.asarray(pdet(samples), dtype=float)
    if probabilities.shape != (n_samples,):
        raise ValueError("pdet must return one probability per emulator sample")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("pdet probabilities must lie in [0, 1]")
    return float(np.mean(probabilities))
