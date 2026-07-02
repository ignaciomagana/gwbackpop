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
SAFE_MODULES = [
    "gwbackpop.config",
    "gwbackpop.cosmology",
    "gwbackpop.metadata",
    "gwbackpop.inference.hierarchical",
]
COSMIC_MODULES = [
    "gwbackpop.evolution.cosmic",
    "gwbackpop.inference.single_event",
    "gwbackpop.selection.injections",
]


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
        from gwbackpop.evolution.cosmic import get_cosmic_capabilities

        caps = get_cosmic_capabilities()
        print("\nCOSMIC capabilities:")
        print(f"  cosmic-popsynth version: {caps['cosmic_popsynth_version']}")
        print(f"  cevars.alpha1 shape: {caps['cevars_alpha1_shape']}")
        print(f"  mtvars.acc_lim shape: {caps['mtvars_acc_lim_shape']}")
        print(f"  has se_flags: {caps['has_se_flags']}")
        print(f"  evolv2 docstring first line: {caps['evolv2_docstring_first_line']}")
        print(f"  selected evolv2 ABI convention: {caps['evolv2_call_convention'] or 'unknown until first call'}")
        print(f"  evolv2 return convention: {caps['evolv2_return_convention'] or 'unknown until first call'}")
        print(f"  kick_info shape used: {caps['evolv2_kick_info_shape'] or 'unknown until first call'}")


if __name__ == "__main__":
    main()
