# Experimental emulator BackPop prototype

**Status: experimental only. Do not use this path for production inference.**

This directory sketches an emulator-based hierarchical BackPop workflow that is intentionally separate from the production scripts. The prototype is a sandbox for replacing finite-COSMIC-catalog KDE selection estimates with a fast conditional density model for

```math
p(m_c, q, z_\mathrm{merge}, \theta_\mathrm{latent} \mid \Lambda),
```

where `\Lambda` denotes population hyperparameters or binary-physics controls and `\theta_\mathrm{latent}` denotes additional simulated latent quantities that future analyses may need for reweighting or diagnostics.

The current implementation is deliberately simple: it uses a conditional Gaussian placeholder, not a validated astrophysical model. The API is designed so a future neural spline flow, density-ratio estimator, or simulation-based inference model can replace the placeholder without changing downstream selection-code call sites.

## What is included

- `api.py` defines the experimental API:
  - generate a COSMIC-style injection catalog (`generate_cosmic_injection_catalog`),
  - load an injection catalog (`load_cosmic_injection_catalog`),
  - featurize hyperparameters (`featurize_hyperparameters`),
  - train a conditional density emulator (`GaussianConditionalEmulator.fit`),
  - sample predicted merger observables (`sample_predicted_mergers`),
  - estimate selection `alpha` from a user-supplied `pdet` function (`estimate_selection_alpha`).
- `toy_emulator_demo.py` runs the workflow on fake data when real COSMIC data are unavailable.

## Minimal workflow

```bash
python experimental/emulator/toy_emulator_demo.py --output-dir /tmp/backpop-emulator-demo
```

The toy script writes a fake catalog, fits the placeholder emulator, samples predicted mergers at a target `\Lambda`, and estimates

```math
\alpha(\Lambda) \approx \frac{1}{N}\sum_j p_\mathrm{det}(m_{c,j}, q_j, z_j).
```

## Catalog schema

The loader expects `.npz` files with at least these arrays:

| Field | Meaning |
| --- | --- |
| `lambda_features` | Two-dimensional array of hyperparameter or binary-physics features. |
| `observables` | Two-dimensional array of simulated outputs. Defaults to columns `(mc, q, z_merge, theta_latent)`. |

Optional arrays:

| Field | Meaning |
| --- | --- |
| `lambda_names` | Names for columns in `lambda_features`. |
| `observable_names` | Names for columns in `observables`. |
| `metadata` | JSON metadata string. |

## Production warnings

- This module is not imported by production BackPop entry points.
- The Gaussian placeholder is not a validated conditional density estimate.
- The toy `pdet` function is not a detector-selection model.
- Real analyses must track the COSMIC proposal density, failed simulations, merger cuts, cosmology, PE priors, and detector-frame/source-frame conventions.
- Selection estimates from this prototype should be treated as API tests and diagnostics only.

## Actions needed before package integration

1. Define a versioned COSMIC simulation schema with explicit proposal densities and failed-run accounting.
2. Generate training grids over the actual hierarchical hyperparameters `\Lambda` used by `hierarchical_backpop_jax.py`.
3. Replace the placeholder Gaussian with a validated conditional density model, such as a neural spline flow or density-ratio estimator.
4. Add calibration diagnostics: posterior predictive checks, coverage tests, held-out likelihoods, and comparisons against direct COSMIC KDE estimates.
5. Thread the emulator through a new, opt-in hierarchical-selection code path without changing existing production defaults.
6. Validate `\alpha(\Lambda)` against LVK/Farr found-injection estimates and direct `pdet` campaigns on shared benchmark catalogs.
7. Add CI tests that use tiny deterministic fake catalogs and separate long-running COSMIC integration tests behind an optional marker.
