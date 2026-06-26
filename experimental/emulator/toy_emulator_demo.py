"""Toy emulator workflow using fake COSMIC-like data.

Run from the repository root:

    python experimental/emulator/toy_emulator_demo.py --output-dir /tmp/backpop-emulator-demo

This script is experimental and is not a production inference workflow.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from .api import (
        GaussianConditionalEmulator,
        estimate_selection_alpha,
        generate_cosmic_injection_catalog,
        load_cosmic_injection_catalog,
        sample_predicted_mergers,
    )
except ImportError:  # Allows direct execution as a script from the repo root.
    from api import (  # type: ignore[no-redef]
        GaussianConditionalEmulator,
        estimate_selection_alpha,
        generate_cosmic_injection_catalog,
        load_cosmic_injection_catalog,
        sample_predicted_mergers,
    )


def toy_pdet(observables: np.ndarray) -> np.ndarray:
    """Smooth fake detection probability for API testing only."""

    mc = observables[:, 0]
    q = observables[:, 1]
    z_merge = observables[:, 2]
    mass_term = 1.0 / (1.0 + np.exp(-(mc - 22.0) / 4.0))
    redshift_term = np.exp(-1.5 * z_merge)
    mass_ratio_term = np.clip(q, 0.0, 1.0) ** 0.5
    return np.clip(mass_term * redshift_term * mass_ratio_term, 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("experimental/emulator/_toy_output"))
    parser.add_argument("--n-training", type=int, default=2_000)
    parser.add_argument("--n-selection-samples", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = args.output_dir / "fake_cosmic_catalog.npz"

    generate_cosmic_injection_catalog(catalog_path, n_simulations=args.n_training, rng=rng)
    catalog = load_cosmic_injection_catalog(catalog_path)
    emulator = GaussianConditionalEmulator.fit(catalog)

    target_lambda = {"alpha_ce": 1.5, "f_acc": 0.4, "sigma_kick": 120.0}
    predicted = sample_predicted_mergers(emulator, target_lambda, n_samples=5, rng=rng)
    alpha = estimate_selection_alpha(
        emulator,
        target_lambda,
        toy_pdet,
        n_samples=args.n_selection_samples,
        rng=rng,
    )

    print("WARNING: experimental emulator prototype; not production inference.")
    print(f"Wrote fake catalog: {catalog_path}")
    print(f"Catalog rows: {len(catalog.observables)}")
    print(f"Observable columns: {catalog.observable_names}")
    print("First five predicted mergers at target Lambda:")
    print(predicted)
    print(f"Toy selection alpha: {alpha:.6f}")


if __name__ == "__main__":
    main()
