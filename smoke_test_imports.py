#!/usr/bin/env python
"""Import BackPop's main modules and print dependency versions."""
from __future__ import annotations

import argparse
import importlib
from importlib import metadata

DEPENDENCIES = [
    "numpy", "scipy", "pandas", "astropy", "pesummary", "nautilus-sampler",
    "jax", "numpyro", "h5py", "matplotlib", "corner", "arviz", "cosmic-popsynth",
]
SAFE_MODULES = ["backpop_config", "cosmo_prior", "metadata_utils", "hierarchical_backpop_jax"]
COSMIC_MODULES = ["backpop", "run_backpop", "run_injections"]


def version(dist_name: str) -> str:
    try:
        return metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        return "not installed"


def import_one(name: str) -> None:
    importlib.import_module(name)
    print(f"import {name}: ok")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-cosmic", action="store_true", help="Skip modules that require a COSMIC installation.")
    args = parser.parse_args()

    print("Dependency versions:")
    for dep in DEPENDENCIES:
        print(f"  {dep}: {version(dep)}")

    print("\nModule imports:")
    for module in SAFE_MODULES:
        import_one(module)
    if args.skip_cosmic:
        print("COSMIC-dependent module imports: skipped")
    else:
        for module in COSMIC_MODULES:
            import_one(module)


if __name__ == "__main__":
    main()
