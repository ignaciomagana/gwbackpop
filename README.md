# BackPop

BackPop connects gravitational-wave parameter-estimation results to binary-population physics.  It uses COSMIC binary evolution as a forward model for isolated binary black-hole (BBH) formation, compares the resulting merger observables to event-level gravitational-wave posteriors, and then combines multiple single-event results in a hierarchical population analysis.

The repository currently contains three main workflows. The installed console commands are the supported executable interface:

| Workflow | Main entry point | Purpose | Status |
|---|---|---|---|
| Single-event 2D mass-only inference | `gwbackpop-run-event` | Infer binary-evolution parameters for one GW event using a KDE likelihood in source-frame chirp mass and mass ratio, `(\mathcal{M}_c, q)`. | **Production-ready baseline**. This is the most mature single-event mode. |
| Single-event 3D mass-redshift inference | `gwbackpop-run-event --use_redshift_likelihood True` | Infer binary-evolution parameters for one GW event using `(\mathcal{M}_c, q, z_\mathrm{merger})` plus cosmological priors on formation redshift and metallicity. | **Production-ready if the PE samples and selection campaign are generated consistently**. Treat 2D/3D comparisons as model-dependent. |
| Hierarchical population inference | `gwbackpop-run-hierarchical` | Reweight per-event BackPop posteriors under a population hypermodel and optionally apply selection corrections. | **Production-ready for the LVK/Farr found-injection estimator, direct `pdet` estimator with validated injection catalogs, or explicit no-selection diagnostics**. |

Additional utilities:

- `gwbackpop-run-injections` builds COSMIC merger catalogs for selection corrections. It can either store a direct `P_det(m_1, m_2, z)` value from a pickled interpolator or store `pdet=nan` for the LVK/Farr workflow.
- `gwbackpop-plot` makes diagnostic figures from single-event output directories.
- `workflows/shell/run_hierarchical_2d.sh`, `workflows/shell/run_hierarchical_3d.sh`, and `workflows/slurm/run_catalog_gwtc3.slurm` are convenience wrappers for catalog-scale analyses.


## Repository layout

```text
src/gwbackpop/       installable package
workflows/           shell and SLURM workflow drivers
tests/               lightweight tests
experimental/        research prototypes not imported by production code
```

## What BackPop does

In short, BackPop supports single-event inference, COSMIC binary evolution, 2D mass-only likelihoods, 3D mass-redshift likelihoods, and hierarchical population inference.

BackPop samples **binary-evolution parameters** rather than just phenomenological BBH masses. A proposed parameter vector `\theta` is passed to COSMIC, COSMIC evolves the binary, and BackPop keeps proposals that form a merging BBH. The merger output is then compared to the gravitational-wave posterior for one event or reweighted across a catalog.

At a high level:

1. **Single-event inference**: for each GW event, BackPop samples a base prior `\pi_0(\theta)`, evolves each proposed binary with COSMIC, and evaluates a likelihood against the PE posterior KDE. The output is an event-specific posterior over `\theta` and a single-event evidence `Z_i`.
2. **COSMIC binary evolution**: `evolv2` in `gwbackpop.evolution.cosmic` maps ZAMS/binary-physics parameters to BBH merger masses and delay times. Non-mergers receive zero likelihood.
3. **2D mass-only likelihood**: the default single-event likelihood compares the COSMIC-predicted source-frame chirp mass and mass ratio to a KDE built from GW posterior samples.
4. **3D mass-redshift likelihood**: the redshift mode also predicts `z_\mathrm{merger}` from `z_\mathrm{form}` and the COSMIC delay time, and evaluates a KDE in `(\mathcal{M}_c, q, z_\mathrm{merger})`.
5. **Hierarchical population inference**: event posteriors are importance-reweighted from `\pi_0(\theta)` to a population density `p(\theta\mid\Lambda)`, where `\Lambda` are hyperparameters for common-envelope efficiency, accretion efficiency, and natal-kick distributions.

## Mathematical model

### Single-event likelihood

Let `d_i` be the strain data for event `i`, `x(\theta)` be the merger observables predicted by COSMIC, and `\pi_\mathrm{PE}(x)` be the prior used by the GW parameter-estimation run in the same observable coordinates. BackPop approximates the event likelihood using PE posterior samples:

```math
p(d_i \mid \theta)
\propto
\frac{p_\mathrm{PE}(x(\theta) \mid d_i)}{\pi_\mathrm{PE}(x(\theta))}.
```

In practice, `p_\mathrm{PE}(x\mid d_i)` is represented by a Gaussian KDE over PE posterior samples. BackPop then samples

```math
p(\theta \mid d_i)
\propto
p(d_i \mid \theta)\,\pi_0(\theta),
```

where `\pi_0(\theta)` is the single-event BackPop sampling prior configured by `get_backpop_config`.

#### 2D mode

The 2D mode uses

```math
x_\mathrm{2D}(\theta) = \left(\mathcal{M}_{c,\mathrm{src}}(\theta), q_\mathrm{src}(\theta)\right),
```

with

```math
\mathcal{M}_c = \frac{(m_1 m_2)^{3/5}}{(m_1+m_2)^{1/5}},
\qquad
q = \frac{m_2}{m_1}, \quad m_1 \ge m_2.
```

This mode is mass-only: it does not use the event redshift likelihood, and `logZ` is sampled with the configured flat single-event prior.

#### 3D mode

The 3D mode uses

```math
x_\mathrm{3D}(\theta) = \left(\mathcal{M}_{c,\mathrm{src}}(\theta), q_\mathrm{src}(\theta), z_\mathrm{merger}(\theta)\right).
```

It adds `z_form` as a sampled parameter, draws or weights it with an SFR-weighted comoving-volume prior, and uses a metallicity model `p(\log Z\mid z_\mathrm{form})`. COSMIC supplies the delay time `t_\mathrm{delay}`; `z_\mathrm{merger}` is obtained by subtracting this delay from the formation lookback time using the Planck15 cosmology implemented in `gwbackpop.cosmology`.

### Hierarchical likelihood

For a catalog of `N` events, BackPop uses posterior-sample importance reweighting. If event `i` was sampled under `\pi_0(\theta)`, then

```math
\mathcal{L}(\Lambda)
=
\prod_i
Z_i
\left\langle
\frac{p(\theta\mid\Lambda)}{\pi_0(\theta)}
\right\rangle_i
\alpha(\Lambda)^{-N}.
```

Here:

- `Z_i` is the single-event evidence saved by `gwbackpop-run-event` as `log_z.npy`.
- `\langle\cdot\rangle_i` is an average over posterior samples for event `i`.
- `p(\theta\mid\Lambda)` is the population model.
- `\alpha(\Lambda)` is the detectable fraction under that population model.
- If selection effects are disabled, BackPop sets the `\alpha(\Lambda)^{-N}` correction to 1. That is useful for debugging but generally biased for astrophysical inference.

The implemented population hypermodel varies these dimensions:

- `alpha_1`, `alpha_2`: truncated LogNormal population factors for common-envelope efficiency.
- `flim_1`, `flim_2`: Beta population factors for accretion-efficiency limits.
- `vk1`, `vk2`: truncated Maxwellian population factors for natal-kick speeds.

Other dimensions are treated as static under the current hypermodel and must remain consistent between event posteriors and selection injections.

## Selection effects

Selection effects enter through

```math
\alpha(\Lambda) = \int p_\mathrm{det}(\theta)\,p(\theta\mid\Lambda)\,d\theta.
```

BackPop has three relevant modes or estimators:

### 1. No selection correction (`none`)

Run `gwbackpop-run-hierarchical` without `--injections_path` and without `--lvk_found_path`. This is fast and useful for smoke tests, convergence debugging, and demonstrating the reweighting machinery. It is **diagnostic only**: because detected GW events are not an unbiased draw from the astrophysical population, no-selection population posteriors should not be overinterpreted.

### 2. Direct `pdet` estimator

`gwbackpop-run-injections --pdet_path /path/to/pdet_interpolator.pkl` evaluates a user-provided callable `P_det(m1_src, m2_src, z_merger)` for each COSMIC merger and stores the result in the injection `.npz`. This is useful for building and auditing direct detection-probability campaigns.

The JAX hierarchical driver now treats `--injections_path` without `--lvk_found_path` as `selection_mode="direct_pdet"`, provided the catalog contains finite `pdet` values. Build the catalog with:

```bash
gwbackpop-run-injections \
  --config_name lucky_strikes \
  --likelihood_mode 2D \
  --pdet_path /path/to/pdet_interpolator.pkl \
  --output_path injections/gwtc3_cosmic_mergers_with_pdet.npz \
  --n_inj 1000000 \
  --n_workers 64
```

Then run hierarchical selection with:

```bash
gwbackpop-run-hierarchical \
  --results_root results \
  --config_name lucky_strikes \
  --injections_path injections/gwtc3_cosmic_mergers_with_pdet.npz \
  --output_dir results/hierarchical/lucky_strikes/nuts/direct_pdet
```

The direct estimator has the schematic Monte Carlo form

```math
\alpha(\Lambda)
\approx
\frac{1}{N_\mathrm{inj}}\sum_j
P_\mathrm{det}(m_{1,j},m_{2,j},z_j)
\frac{p(\theta_j\mid\Lambda)}{q(\theta_j)},
```

where `q(\theta)` is the actual injection proposal density. The proposal density is not optional: changing how injections are drawn changes the denominator.

### 3. LVK/Farr found-injection estimator

The production selection workflow uses COSMIC merger catalogs plus an LVK found-injection HDF5 file. The code builds a kernel matrix between LVK found injections and COSMIC mergers in `(\log m_1, \log m_2, z)` and uses a Farr-style estimator for `\alpha(\Lambda)`.

Use this mode by passing both:

- `--injections_path`: COSMIC merger catalog from `gwbackpop-run-injections`.
- `--lvk_found_path`: LVK found-injection HDF5 containing source masses, redshift, and `sampling_pdf`.

This is the **recommended production hierarchical mode**, subject to the caveats below.

## 2D vs 3D distinction

The distinction is not just one extra KDE dimension; it changes the generative measure that must be used consistently throughout the analysis.

| Feature | 2D mode | 3D mode |
|---|---|---|
| Event KDE coordinates | `(\mathcal{M}_c, q)` | `(\mathcal{M}_c, q, z_\mathrm{merger})` |
| Redshift likelihood | Not used | Used through PE posterior KDE |
| `z_form` | Auxiliary for injection/redshift bookkeeping, not part of event likelihood | Sampled single-event parameter |
| `logZ` treatment | Flat configured prior | `p(\log Z \mid z_\mathrm{form})` metallicity prior |
| SFR prior | Not part of event likelihood | Included through `p(z_\mathrm{form})` |
| Selection campaign | Must be built with `--likelihood_mode 2D` | Must be built with `--likelihood_mode 3D` |

Do not mix 2D event posteriors with 3D injection campaigns, or vice versa, unless you are deliberately running a legacy diagnostic with `--allow_inconsistent_selection_model True` and are prepared to interpret the resulting bias.

## Expected input files

### PESummary GW posterior files

`gwbackpop-run-event --samples_path` expects a PESummary-compatible GW posterior file, usually HDF5. The code loads event posterior samples, builds a KDE in the requested coordinates, and uses the chosen approximant group, defaulting to `C01:Mixed`.

Important expectations:

- Mass samples must be interpretable as source-frame or detector-frame masses. Use `--mass_frame auto`, `source`, or `detector`.
- If detector-frame masses are used, redshift information must be available for conversion to source-frame masses.
- PE priors in the KDE coordinates must be understood well enough to justify the posterior-over-prior likelihood approximation.

### COSMIC injection `.npz`

`gwbackpop-run-injections` writes an `.npz` file containing the COSMIC merger catalog used for selection calculations. Important arrays and metadata include:

- `theta`: full sampled parameter vectors for merging COSMIC binaries.
- `params`: parameter names matching the columns of `theta`.
- `lower_bound`, `upper_bound`: base prior bounds.
- `m1_src`, `m2_src`, `z_merger`, `t_delay_myr`: merger observables.
- `pdet`: direct detection probability when `--pdet_path` is supplied; otherwise `nan` for LVK raw-injection mode.
- `log_q_proposal`: explicit proposal-density values for modern injection campaigns.
- `N_inj`, `N_merge`, `kick_proposal_sigma`, `likelihood_mode`, and run metadata.

### LVK found-injection HDF5

The LVK/Farr mode expects an HDF5 file with an `injections` group containing at least:

- `mass1_source`
- `mass2_source`
- `redshift`
- `sampling_pdf`

The code audits HDF5 metadata and assumes `sampling_pdf` is a density in `d(mass1_source) d(mass2_source) d(redshift)`. Because the kernel is evaluated in `(\log m_1, \log m_2, z)`, BackPop applies an `m_1 m_2` Jacobian. Use `--strict_lvk_sampling_pdf True` to fail rather than warn when the HDF5 metadata does not verify this coordinate convention.

## Example commands

The paths below are examples. Replace them with your PESummary files, pickled `P_det` interpolator, and LVK injection release.

### Single-event 2D run

```bash
gwbackpop-run-event \
  --samples_path /data/pe/GW150914_pesummary.h5 \
  --event_name GW150914 \
  --config_name lucky_strikes \
  --use_redshift_likelihood False \
  --approximant 'C01:Mixed' \
  --nlive 3000 \
  --neff 30000
```

### Single-event 3D run

```bash
gwbackpop-run-event \
  --samples_path /data/pe/GW190814_pesummary.h5 \
  --event_name GW190814 \
  --config_name lucky_strikes_zform \
  --use_redshift_likelihood True \
  --approximant 'C01:Mixed' \
  --nlive 3000 \
  --neff 30000
```

### Injection campaign

For LVK/Farr production selection, omit `--pdet_path` so the catalog stores raw COSMIC mergers and lets the LVK found-injection estimator handle detectability:

```bash
gwbackpop-run-injections \
  --config_name lucky_strikes_zform \
  --likelihood_mode 3D \
  --output_path injections/gwtc3_cosmic_mergers_3d.npz \
  --n_inj 1000000 \
  --n_workers 64
```

For a direct-`pdet` diagnostic campaign:

```bash
gwbackpop-run-injections \
  --config_name lucky_strikes \
  --likelihood_mode 2D \
  --pdet_path /data/selection/pdet_interpolator.pkl \
  --output_path injections/gwtc3_cosmic_mergers_with_pdet.npz \
  --n_inj 1000000 \
  --n_workers 64
```

### Hierarchical run with no selection correction

This is a diagnostic run only:

```bash
gwbackpop-run-hierarchical \
  --results_root results \
  --config_name lucky_strikes \
  --output_dir results/hierarchical/lucky_strikes/nuts/no_selection \
  --n_samples 5000 \
  --num_warmup 500 \
  --num_samples 1000 \
  --num_chains 4
```

### Hierarchical direct-`pdet` run

Direct-`pdet` hierarchical selection is implemented in `gwbackpop-run-hierarchical` when an injection catalog with finite `pdet` is provided and `--lvk_found_path` is omitted:

```bash
gwbackpop-run-hierarchical \
  --results_root results \
  --config_name lucky_strikes \
  --injections_path injections/gwtc3_cosmic_mergers_with_pdet.npz \
  --output_dir results/hierarchical/lucky_strikes/nuts/direct_pdet
```

This mode uses the `pdet` stored in the COSMIC merger catalog and normalizes `\alpha(\Lambda)` by `N_inj`, the total number of proposed COSMIC draws, not `N_merge`, the number of stored mergers. If `pdet` is `nan`, either provide `--lvk_found_path` for LVK/Farr mode or rebuild the catalog with `--pdet_path`. 2D events need 2D injection metadata; 3D events need 3D injection metadata.

### Hierarchical LVK/Farr run

```bash
gwbackpop-run-hierarchical \
  --results_root results \
  --config_name lucky_strikes_zform \
  --injections_path injections/gwtc3_cosmic_mergers_3d.npz \
  --lvk_found_path injections/endo3_bbhpop-LIGO-T2100113-v12.hdf5 \
  --output_dir results/hierarchical/lucky_strikes_zform/nuts/lvk_farr \
  --n_samples 10000 \
  --num_warmup 500 \
  --num_samples 1000 \
  --num_chains 4 \
  --lvk_n_found_max 5000 \
  --strict_lvk_sampling_pdf False
```

## Outputs

### Single-event output

`gwbackpop-run-event` writes to `results/<event_name>/<config_name>/`:

- `points.npy`: posterior samples in BackPop parameter space.
- `log_w.npy`: log importance weights.
- `log_l.npy`: log likelihood values.
- `log_z.npy`: log evidence used by the hierarchical analysis.
- `blobs.npy`: COSMIC evolution tracks for saved samples.
- `metadata.npz`: event name, parameter names, bounds, package versions, and likelihood-mode metadata.

### Hierarchical output

`gwbackpop-run-hierarchical` writes to the requested `--output_dir`:

- `samples.npz`: per-chain NumPyro samples.
- `points.npy`: flattened hyperparameter posterior samples.
- `summary.csv`: posterior means, quantiles, R-hat, and effective sample sizes.
- `metadata.npz`: run settings, event list, selection-mode metadata, and timing.
- Optional diagnostic plots when plotting dependencies are installed.

## Known caveats

### Selection-effect consistency

Selection injections must be generated under the same generative measure as the event posteriors. In particular, 2D event posteriors should use 2D injection metadata and 3D event posteriors should use 3D injection metadata. The JAX hierarchical driver checks this metadata and fails by default on inconsistent selection models.

### Proposal density

Injection campaigns are not always drawn from the same distribution as the target population. For example, kick speeds are drawn from a truncated Maxwellian proposal rather than a uniform box, and kick directions are drawn isotropically in COSMIC coordinates. Correct selection estimates require the ratio `p(\theta\mid\Lambda)/q(\theta)`, not just `p(\theta\mid\Lambda)`.

### PE prior assumptions

The single-event likelihood uses a KDE approximation to the PE posterior divided by the PE prior in the chosen coordinates. If the PE prior is not flat in those coordinates, or if the conversion between detector-frame and source-frame masses is mishandled, the resulting likelihood can be biased. Users should audit the PESummary file, approximant group, mass frame, and redshift handling for each event.

### LVK `sampling_pdf` coordinate assumptions

The LVK/Farr estimator assumes `sampling_pdf` is expressed per `d(mass1_source) d(mass2_source) d(redshift)`, while BackPop evaluates kernels in `(\log m_1, \log m_2, z)` and applies the corresponding `m_1 m_2` Jacobian. Ambiguous HDF5 metadata triggers warnings by default. Use strict mode for production audits.

### Finite support and truncation

BackPop uses finite prior bounds for several event and injection parameters and truncated population densities for some hypermodel factors. KDE tails, support gates, and truncation choices can affect evidence values and hierarchical weights. The default single-event support gate is `none`, which leaves tail behavior to the KDE; `hard` and `soft` gates are explicit diagnostics and should be justified before use in production.

### Diagnostic modes are not science results

No-selection hierarchical runs, direct-`pdet` runs with invalid or inconsistent pdet catalogs, small injection campaigns, heavily subsampled LVK found-injection matrices, and intentionally inconsistent 2D/3D combinations are diagnostics. They can be valuable for debugging but should not be presented as final astrophysical population constraints.

## Environment setup and lightweight tests

This repository includes a `pyproject.toml` for reproducible Python dependency setup.  A typical fresh checkout can be prepared with:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
```

The core Python dependencies are `numpy`, `scipy`, `pandas`, `astropy`, `pesummary`, `nautilus-sampler`, `jax`, `numpyro`, `h5py`, `matplotlib`, `corner`, and `arviz`.

COSMIC is separated from the default install because it can require compiled binary-evolution components and platform-specific setup.  If your platform supports the PyPI package, try:

```bash
python -m pip install -e '.[cosmic]'
```

If that does not work, follow the COSMIC installation instructions for your system and then rerun the smoke test without `--skip-cosmic`.

Fast local checks that do not require COSMIC are:

```bash
gwbackpop-smoke-test --skip-cosmic
pytest tests/test_lightweight_infrastructure.py tests/test_truncated_population_densities.py tests/test_hierarchical_toy_recovery.py tests/test_selection_model_consistency.py tests/test_default_hyperparams.py
```

Injection catalogs written by `gwbackpop-run-injections` keep their full scientific
arrays in the requested catalog NPZ and write metadata to a separate
`*_metadata.npz`/JSON sidecar.  Two-dimensional catalogs record `z_form` as an
auxiliary proposal draw: when explicit `log_q_proposal` weights are present,
hierarchical selection accounting cancels this auxiliary SFR proposal factor
rather than leaving a spurious `1/q(z_form)` term.  Three-dimensional catalogs
draw and evaluate `P(logZ | z_form)` on the exact finite `logZ` support of the
active BackPop config.

After installing COSMIC, run the full smoke import check with:

```bash
gwbackpop-smoke-test
```

Heavy tests and production workflows that evolve COSMIC binaries or consume GW/LVK data products are intentionally kept separate from the GitHub Actions lightweight test job.
