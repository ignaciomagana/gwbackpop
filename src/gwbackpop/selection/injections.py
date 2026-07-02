"""
Selection injection catalog generation
--------------------------------------
Pre-compute the injection campaign needed for selection effects in the
hierarchical BackPop analysis.

For each injection we:
  1. Draw (m1, q, logtb, logZ, α1, α2, flim1, flim2, vk*, angles) from π₀ (flat)
  2. Draw z_form from the SFR-weighted comoving volume prior (Madau-Dickinson)
  3. Draw logZ from P(logZ | z_form) — truncated Normal (Andrews+2021 Eq. 8)
     This concentrates injections where stars actually form, which is the
     natural proposal for z_form and avoids wasting COSMIC calls at z>10.
  4. Run COSMIC → BBH merger or not
  5. If merger: compute z_merger from t_delay + z_form
  6. Evaluate P_det(m1_src, m2_src, z_merger) via the user-supplied interpolator
  7. Store full parameter vector + merger outputs + pdet

The weight ratio p(θ_j | Λ_pop) / q(θ_j) for each merging injection is:
  - α1, α2:    log-Normal / Uniform[0.1, 20]
  - flim1,2:   Beta(a,b)  / Uniform[0, 1]
  - vk1, vk2:  Maxwell    / truncated Maxwell proposal on [0, 500]
  - phi1,phi2: flat event prior / isotropic-direction proposal density
  - z_form:    1.0  (drew from SFR prior = population prior for z_form)
  - logZ:      1.0  (drew from P(logZ|z_form) = population prior for logZ)
  - m1, q, logtb, theta, omega: 1.0  (flat in both population and proposal)

This means InjectionCampaign.log_weight_ratio only touches the 6 binary
physics dimensions, identical to EventPosterior.log_weight_ratio.

Usage
-----
  gwbackpop-run-injections \\
      --pdet_path     /path/to/pdet_interpolator.pkl \\
      --output_path   injections/gwtc3_injections.npz \\
      --n_inj         1000000 \\
      --n_workers     64

  # Estimate merger fraction first (fast, N=10000):
  gwbackpop-run-injections \\
      --pdet_path  /path/to/pdet_interpolator.pkl \\
      --output_path injections/test.npz \\
      --n_inj      10000 --n_workers 8 --dry_run True

Output (NPZ file):
  theta           (N_merge, N_params)   full parameter vectors of merging injections
  m1_src          (N_merge,)            source-frame merger masses
  m2_src          (N_merge,)
  z_merger        (N_merge,)
  t_delay_myr     (N_merge,)
  pdet            (N_merge,)            P_det(m1_src, m2_src, z_merger) ∈ [0,1]
  params          list of parameter names (same order as theta columns)
  lower_bound     (N_params,)           π₀ bounds for weight ratio computation
  upper_bound     (N_params,)
  N_inj           scalar                total draws (for normalisation)
  N_merge         scalar                merging BBH count
  N_workers       scalar
  wall_time_s     scalar
"""

from __future__ import annotations

import os
import sys
import time
import pickle
import warnings
import traceback
from collections import Counter
import numpy as np
from argparse import ArgumentParser, ArgumentTypeError
from multiprocessing import Pool, cpu_count
from scipy.stats import maxwell, beta as beta_dist, lognorm
from scipy.special import logsumexp
from gwbackpop.metadata import base_runtime_metadata, get_package_versions, save_metadata
from gwbackpop.evolution.cosmic_capabilities import (
    inspect_cosmic_capabilities,
    require_supported_cosmic_for_independent_alpha_flim,
)
from gwbackpop.population.constants import POP_PARAM_NAMES

# ---------------------------------------------------------------------------
# Default parameter space is loaded from gwbackpop.config.get_backpop_config().
# These module globals are initialised to the default config for import-time
# helpers/tests, and overwritten in worker processes by _worker_init().
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_NAME = "lucky_strikes"


def _load_config(config_name: str):
    from gwbackpop.config import get_backpop_config

    lower_bound, upper_bound, params_in, fixed_params = get_backpop_config(config_name)
    return (
        np.asarray(lower_bound, dtype=np.float64),
        np.asarray(upper_bound, dtype=np.float64),
        list(params_in),
        dict(fixed_params),
    )


LOWER, UPPER, PARAMS, FIXED_PARAMS = _load_config(DEFAULT_CONFIG_NAME)

# Kick velocity injection proposal: Maxwellian(KICK_PROPOSAL_SIGMA km/s)
# The uniform prior [0, 500] has f_merge ~ 0 because large kicks disrupt all binaries.
# A Maxwellian concentrates draws where mergers actually occur while maintaining
# full support over [0, 500].  The weight ratio in InjectionCampaign / LVKInjectionCampaign
# divides by this proposal density rather than 1/500.
# Choice: σ=50 km/s gives median kick ~63 km/s.  BH kicks from fallback
# are typically O(10-100) km/s for massive progenitors (Fryer+2012, Mandel+2016).
KICK_PROPOSAL_SIGMA: float = 50.0   # km/s  — Maxwellian scale parameter
PROPOSAL_NAME = "zams_uniform_truncated_maxwell_isotropic_kick_aux_zform"
LIKELIHOOD_MODE = "2D"
PROPOSAL_VERSION = "3"
COORDINATE_SYSTEM = "backpop_zams_log10Z_zform_source_merger"

# logZ drawn from P(logZ|z_form); bounds used only for rejection safeguard
LOGZ_LO = np.log10(1e-4)
LOGZ_HI = np.log10(0.03)

# z_form drawn from SFR prior; upper limit where SFR is negligible
ZFORM_MAX = 20.0
DEBUG_FAILURES = False
MAX_DEBUG_EXAMPLES = 5
PROPOSAL_MODE = "broad"
BROAD_MIXTURE_FRACTION = 1.0
HYPERPOSTERIOR_COMPONENTS: np.ndarray | None = None
MIXTURE_COMPONENT_COUNT = 0
HYPERPOSTERIOR_PATH: str | None = None


# ---------------------------------------------------------------------------
# Worker function (runs in subprocess — must be importable at module level)
# ---------------------------------------------------------------------------

def _worker_init(pdet_path: str | None, config_name: str = DEFAULT_CONFIG_NAME, likelihood_mode: str = "2D", pdet_callable=None, debug_failures: bool = False, proposal_mode: str = "broad", broad_mixture_fraction: float = 1.0, hyperposterior_components=None, hyperposterior_path: str | None = None) -> None:
    """Load P_det interpolator once per worker process.
    If pdet_path is None, P_det evaluation is skipped and pdet=nan is stored.
    Use this mode when building the COSMIC merger catalog for LVKInjectionCampaign.
    """
    global _PDET, LOWER, UPPER, PARAMS, FIXED_PARAMS, LIKELIHOOD_MODE, DEBUG_FAILURES, PROPOSAL_MODE, BROAD_MIXTURE_FRACTION, HYPERPOSTERIOR_COMPONENTS, MIXTURE_COMPONENT_COUNT, HYPERPOSTERIOR_PATH
    LOWER, UPPER, PARAMS, FIXED_PARAMS = _load_config(config_name)
    require_supported_cosmic_for_independent_alpha_flim(
        config_name=config_name,
        params=PARAMS,
    )
    LIKELIHOOD_MODE = str(likelihood_mode).upper()
    DEBUG_FAILURES = bool(debug_failures)
    PROPOSAL_MODE = str(proposal_mode)
    BROAD_MIXTURE_FRACTION = float(broad_mixture_fraction)
    HYPERPOSTERIOR_COMPONENTS = None if hyperposterior_components is None else np.asarray(hyperposterior_components, dtype=np.float64)
    MIXTURE_COMPONENT_COUNT = 0 if HYPERPOSTERIOR_COMPONENTS is None else int(len(HYPERPOSTERIOR_COMPONENTS))
    HYPERPOSTERIOR_PATH = hyperposterior_path
    if pdet_callable is not None:
        _PDET = pdet_callable
        return
    if pdet_path is None:
        _PDET = None
        return
    with open(pdet_path, 'rb') as f:
        _PDET = pickle.load(f)


def _failure(reason: str, **extra) -> dict | None:
    if not DEBUG_FAILURES:
        return None
    out = {"ok": False, "reason": reason}
    out.update(extra)
    return out


def load_hyperposterior_samples(path: str) -> np.ndarray:
    """Load posterior samples in POP_PARAM_NAMES order from a no-selection run."""
    data = np.load(path, allow_pickle=True)
    if "samples" in data.files:
        arr = np.asarray(data["samples"], dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] == len(POP_PARAM_NAMES):
            return arr
    cols = []
    missing = []
    for name in POP_PARAM_NAMES:
        if name in data.files:
            cols.append(np.ravel(np.asarray(data[name], dtype=np.float64)))
        else:
            missing.append(name)
    if missing:
        raise ValueError(f"Hyperposterior file is missing POP_PARAM_NAMES entries: {missing}")
    n = min(len(c) for c in cols)
    arr = np.column_stack([c[:n] for c in cols])
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError("Hyperposterior samples must be non-empty and finite")
    return arr


def _subsample_hyperposterior(samples: np.ndarray, count: int, seed: int = 271828) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float64)
    if count <= 0 or len(samples) <= count:
        return samples.copy()
    rng = np.random.default_rng(seed)
    return samples[rng.choice(len(samples), size=count, replace=False)]


def _draw_truncated_lognormal(rng: np.random.Generator, mu: float, sig: float, lo: float, hi: float) -> float:
    c0 = lognorm.cdf(lo, s=sig, scale=np.exp(mu))
    c1 = lognorm.cdf(hi, s=sig, scale=np.exp(mu))
    return float(lognorm.ppf(rng.uniform(c0, c1), s=sig, scale=np.exp(mu)))


def _truncated_lognormal_logpdf(x, mu: float, sig: float, lo: float, hi: float):
    norm = lognorm.cdf(hi, s=sig, scale=np.exp(mu)) - lognorm.cdf(lo, s=sig, scale=np.exp(mu))
    lp = lognorm.logpdf(x, s=sig, scale=np.exp(mu)) - np.log(norm)
    return np.where((lo <= x) & (x <= hi) & (norm > 0), lp, -np.inf)


def _component_logpdf_theta(theta: np.ndarray, component: np.ndarray) -> float:
    hp = dict(zip(POP_PARAM_NAMES, np.asarray(component, dtype=np.float64)))
    logp = 0.0
    for name, mu, sig in (("alpha_1", hp["mu_logalpha1"], hp["sig_logalpha1"]),
                          ("alpha_2", hp["mu_logalpha2"], hp["sig_logalpha2"])):
        if name in PARAMS:
            i = PARAMS.index(name)
            logp += float(_truncated_lognormal_logpdf(theta[i], mu, sig, LOWER[i], UPPER[i]))
    for name, a, b in (("flim_1", hp["a_f1"], hp["b_f1"]), ("flim_2", hp["a_f2"], hp["b_f2"])):
        if name in PARAMS:
            i = PARAMS.index(name)
            logp += float(beta_dist.logpdf(theta[i], a, b))
    for name, sig in (("vk1", hp["sigma_v1"]), ("vk2", hp["sigma_v2"])):
        if name in PARAMS:
            i = PARAMS.index(name)
            logp += float(_truncated_maxwell_logpdf(theta[i], sig, LOWER[i], UPPER[i]))
    return float(logp)


def compute_adaptive_log_q_proposal(theta: np.ndarray, z_form: float, log10_Z: float, eps: float, components: np.ndarray) -> float:
    """Exact broad/adaptive-mixture proposal density for a stored injection."""
    broad = compute_log_q_proposal(theta, z_form, log10_Z, KICK_PROPOSAL_SIGMA)
    if components is None or len(components) == 0:
        return float(broad)
    comp = np.array([_component_logpdf_theta(theta, c) for c in components], dtype=np.float64)
    # Adaptive components retain the same proposal factors for non-varied
    # dimensions as the broad campaign, replacing only alpha/flim/vk factors.
    for name in ("alpha_1", "alpha_2", "flim_1", "flim_2", "vk1", "vk2"):
        if name in PARAMS:
            i = PARAMS.index(name)
            if name in ("vk1", "vk2"):
                broad -= float(_truncated_maxwell_logpdf(theta[i], KICK_PROPOSAL_SIGMA, LOWER[i], UPPER[i]))
            elif name in ("phi1", "phi2"):
                pass
            else:
                broad -= float(-np.log(UPPER[i] - LOWER[i]))
    adaptive = broad + logsumexp(comp) - np.log(len(comp))
    terms = []
    if eps > 0:
        terms.append(np.log(eps) + compute_log_q_proposal(theta, z_form, log10_Z, KICK_PROPOSAL_SIGMA))
    if eps < 1:
        terms.append(np.log1p(-eps) + adaptive)
    return float(logsumexp(terms))


def _draw_theta_adaptive(rng: np.random.Generator) -> np.ndarray:
    theta = rng.uniform(LOWER, UPPER)
    if HYPERPOSTERIOR_COMPONENTS is None or rng.uniform() < BROAD_MIXTURE_FRACTION:
        for kick_name in ("vk1", "vk2"):
            if kick_name in PARAMS:
                i = PARAMS.index(kick_name)
                theta[i] = _draw_truncated_maxwell(rng, KICK_PROPOSAL_SIGMA, LOWER[i], UPPER[i])
        return theta
    hp = dict(zip(POP_PARAM_NAMES, HYPERPOSTERIOR_COMPONENTS[rng.integers(len(HYPERPOSTERIOR_COMPONENTS))]))
    for name, mu, sig in (("alpha_1", hp["mu_logalpha1"], hp["sig_logalpha1"]),
                          ("alpha_2", hp["mu_logalpha2"], hp["sig_logalpha2"])):
        if name in PARAMS:
            i = PARAMS.index(name); theta[i] = _draw_truncated_lognormal(rng, mu, sig, LOWER[i], UPPER[i])
    for name, a, b in (("flim_1", hp["a_f1"], hp["b_f1"]), ("flim_2", hp["a_f2"], hp["b_f2"])):
        if name in PARAMS:
            theta[PARAMS.index(name)] = float(rng.beta(a, b))
    for name, sig in (("vk1", hp["sigma_v1"]), ("vk2", hp["sigma_v2"])):
        if name in PARAMS:
            i = PARAMS.index(name); theta[i] = _draw_truncated_maxwell(rng, sig, LOWER[i], UPPER[i])
    return theta


def _run_one(seed: int) -> dict | None:
    """Draw one injection, run COSMIC, return result dict or None.

    Returns None if binary doesn't form a merging BBH — these are discarded
    (contribute P_det = 0 to α regardless of Λ_pop).
    """
    rng = np.random.default_rng(seed)

    # ---- Draw binary physics + ZAMS ----
    # Non-kick parameters: flat prior π₀ (uniform over LOWER, UPPER)
    # Kick speeds: Maxwellian(KICK_PROPOSAL_SIGMA) truncated to [0, 500].
    # Kick directions: isotropic in COSMIC convention.  COSMIC expects
    # natal_kick_array=[vk, phi, theta, omega, rand_seed], where phi is the
    # co-latitude/elevation-like polar angle in [-90, 90] deg and theta is
    # the azimuth in [0, 360] deg.  Isotropy therefore requires sin(phi), not
    # phi itself, to be uniform; theta and omega are uniform.
    # Rationale: uniform [0,500] km/s gives f_merge ≈ 0 because most kicks
    # disrupt the binary.  Maxwellian concentrates where mergers occur
    # while maintaining support over the full [0,500] range.
    theta = _draw_theta_adaptive(rng) if PROPOSAL_MODE == "adaptive_hyperposterior" else rng.uniform(LOWER, UPPER)
    params_dict = dict(zip(PARAMS, theta))

    # Overwrite vk1 and vk2 with exactly truncated Maxwellian draws.
    # Do not clip: clipping would create an atom at the upper bound.
    for kick_name in ('vk1', 'vk2'):
        if PROPOSAL_MODE == "adaptive_hyperposterior":
            continue
        if kick_name in PARAMS:
            i_kick = PARAMS.index(kick_name)
            params_dict[kick_name] = _draw_truncated_maxwell(
                rng, KICK_PROPOSAL_SIGMA, LOWER[i_kick], UPPER[i_kick]
            )
            # Reflect truncation into theta for storage.
            theta[i_kick] = params_dict[kick_name]

    # Overwrite kick angles with isotropic directions in COSMIC convention.
    for suffix in ("1", "2"):
        _draw_and_store_kick_direction(rng, params_dict, theta, suffix)

    # ---- Draw auxiliary formation redshift and metallicity ----
    # 2D selection uses z_form only as a COSMIC/redshift auxiliary variable and
    # draws logZ from the same flat base measure as the 2D event likelihood.
    # 3D selection uses the physical SFR and P(logZ | z_form) priors.
    z_form = _draw_z_form(rng)
    if z_form is None:
        return _failure("invalid_z_form")

    if LIKELIHOOD_MODE == "2D":
        log10_Z = float(rng.uniform(LOGZ_LO, LOGZ_HI))
    elif LIKELIHOOD_MODE == "3D":
        log10_Z = _draw_logZ_given_z(rng, z_form)
        if log10_Z is None:
            return _failure("invalid_logZ")
    else:
        raise ValueError(f"Unknown LIKELIHOOD_MODE={LIKELIHOOD_MODE!r}")

    params_dict['logZ'] = log10_Z
    if 'logZ' in PARAMS:
        theta[PARAMS.index('logZ')] = log10_Z
    if 'z_form' in PARAMS:
        theta[PARAMS.index('z_form')] = z_form

    if PROPOSAL_MODE == "adaptive_hyperposterior":
        log_q_proposal = compute_adaptive_log_q_proposal(
            theta, z_form, log10_Z, BROAD_MIXTURE_FRACTION, HYPERPOSTERIOR_COMPONENTS
        )
    else:
        log_q_proposal = compute_log_q_proposal(theta, z_form, log10_Z, KICK_PROPOSAL_SIGMA)
    if not np.isfinite(log_q_proposal):
        return _failure("invalid_log_q_proposal")

    # ---- Run COSMIC ----
    try:
        from gwbackpop.evolution.cosmic import evolv2
        final_state, bpp_raw, _ = evolv2(
            params_dict, ['mass_1', 'mass_2'], fixed_params=FIXED_PARAMS
        )
    except Exception as exc:
        return _failure(
            "cosmic_exception",
            message=repr(exc),
            traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        )

    if final_state is None:
        return _failure("no_bbh_merger")   # no BBH merger within Hubble time

    m1_src = float(final_state['mass_1'])
    m2_src = float(final_state['mass_2'])
    if m1_src < m2_src:
        m1_src, m2_src = m2_src, m1_src

    # ---- Delay time → z_merger ----
    from gwbackpop.cosmology import z_merger_from_t_delay
    import pandas as pd
    from gwbackpop.evolution.cosmic import COLS_KEEP

    # Extract t_delay from bpp
    bpp = pd.DataFrame(bpp_raw, columns=COLS_KEEP)
    merger_rows = bpp.loc[
        (bpp.kstar_1 == 14) & (bpp.kstar_2 == 14) & (bpp.evol_type == 3)
    ]
    if len(merger_rows) == 0:
        return _failure("no_bbh_merger")
    t_delay = float(merger_rows['tphys'].iloc[0])   # Myr

    z_merger = z_merger_from_t_delay(z_form, t_delay)
    if z_merger is None:
        return _failure("invalid_z_merger")   # merges in future or before formation

    # ---- P_det(m1_src, m2_src, z_merger) ----
    # If _PDET is None (LVK raw-injection mode), store nan — pdet is evaluated
    # later by LVKInjectionCampaign using the found injection set directly.
    if _PDET is None:
        pdet = np.nan
    else:
        try:
            pdet = float(_PDET(m1_src, m2_src, z_merger))
            pdet = float(np.clip(pdet, 0.0, 1.0))
        except Exception as exc:
            return _failure("pdet_exception", message=repr(exc))

    return dict(
        theta      = theta,
        logZ       = log10_Z,
        z_form     = z_form,
        m1_src     = m1_src,
        m2_src     = m2_src,
        z_merger   = z_merger,
        t_delay    = t_delay,
        pdet       = pdet,
        log_q_proposal = log_q_proposal,
        ok=True,
    )


def _draw_truncated_maxwell(
    rng: np.random.Generator,
    scale: float,
    lower: float,
    upper: float,
) -> float:
    """Draw exactly from Maxwell(scale) conditioned on lower <= vk <= upper.

    Uses inverse-CDF sampling between the two truncation CDF values, avoiding
    both clipping (which creates a point mass at the boundary) and rejection
    failures for narrow or far-tail intervals.
    """
    cdf_lower = maxwell.cdf(lower, scale=scale)
    cdf_upper = maxwell.cdf(upper, scale=scale)
    if not cdf_upper > cdf_lower:
        raise ValueError("Invalid truncated Maxwell interval or scale")
    u = rng.uniform(cdf_lower, cdf_upper)
    return float(maxwell.ppf(u, scale=scale))


def _truncated_maxwell_logpdf(x: float | np.ndarray, scale: float, lower: float, upper: float) -> float | np.ndarray:
    """Log density of Maxwell(scale) truncated to [lower, upper]."""
    norm = maxwell.cdf(upper, scale=scale) - maxwell.cdf(lower, scale=scale)
    logpdf = maxwell.logpdf(x, scale=scale) - np.log(norm)
    return np.where((lower <= x) & (x <= upper) & (norm > 0.0), logpdf, -np.inf)



def _draw_and_store_kick_direction(
    rng: np.random.Generator,
    params_dict: dict,
    theta: np.ndarray,
    suffix: str,
) -> None:
    """Draw one isotropic COSMIC kick direction and store sampled params.

    COSMIC's explicit natal-kick columns are [vk, phi, theta, omega, rand_seed],
    with phi valid on [-90, 90] deg and theta valid on [0, 360] deg.  Uniform
    phi is not isotropic; an isotropic direction has sin(phi) ~ Uniform[-1, 1].
    The orbital phase omega is not part of the direction and remains uniform.
    """
    phi_name = f"phi{suffix}"
    theta_name = f"theta{suffix}"
    omega_name = f"omega{suffix}"

    if phi_name in PARAMS:
        i_phi = PARAMS.index(phi_name)
        sin_phi_lo = np.sin(np.deg2rad(LOWER[i_phi]))
        sin_phi_hi = np.sin(np.deg2rad(UPPER[i_phi]))
        sin_phi = rng.uniform(sin_phi_lo, sin_phi_hi)
        phi = float(np.rad2deg(np.arcsin(sin_phi)))
        params_dict[phi_name] = phi
        theta[i_phi] = phi

    # The COSMIC azimuth is named ``theta`` in natal_kick_array.  It is uniform
    # for an isotropic direction, so the initial box draw is already correct.
    for name in (theta_name, omega_name):
        if name in PARAMS:
            params_dict[name] = float(theta[PARAMS.index(name)])


def _isotropic_phi_logpdf(phi_deg: float | np.ndarray, lower: float, upper: float) -> float | np.ndarray:
    """Log PDF for phi when sin(phi) is uniform over the configured bounds."""
    phi = np.asarray(phi_deg, dtype=np.float64)
    sin_lo = np.sin(np.deg2rad(lower))
    sin_hi = np.sin(np.deg2rad(upper))
    norm = sin_hi - sin_lo
    density = (np.pi / 180.0) * np.cos(np.deg2rad(phi)) / norm
    out = np.where((lower <= phi) & (phi <= upper) & (density > 0.0), np.log(density), -np.inf)
    return float(out) if out.ndim == 0 else out


def sample_kick_directions_for_diagnostic(
    rng: np.random.Generator,
    n: int,
    phi_bounds: tuple[float, float] = (-90.0, 90.0),
    theta_bounds: tuple[float, float] = (0.0, 360.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Draw diagnostic COSMIC kick angles from the same isotropic law as injections."""
    sin_phi = rng.uniform(
        np.sin(np.deg2rad(phi_bounds[0])),
        np.sin(np.deg2rad(phi_bounds[1])),
        size=n,
    )
    phi = np.rad2deg(np.arcsin(sin_phi))
    # Use a randomized midpoint grid rather than plain Monte Carlo for the
    # azimuthal diagnostic.  This samples the same uniform law but keeps
    # lightweight tests stable by avoiding rare variance excursions.
    theta_unit = (np.arange(n, dtype=np.float64) + 0.5) / n
    rng.shuffle(theta_unit)
    theta = theta_bounds[0] + theta_unit * (theta_bounds[1] - theta_bounds[0])
    return phi, theta

def _log_q_z_form(z_form: float) -> float:
    """Log density of the SFR proposal actually sampled on [0, ZFORM_MAX]."""
    from gwbackpop.cosmology import _prior_weight_grid, _zgrid

    if z_form < 0.0 or z_form > ZFORM_MAX:
        return -np.inf
    mask = _zgrid <= ZFORM_MAX
    z_norm_grid = np.concatenate([_zgrid[mask], np.array([ZFORM_MAX])])
    w_norm_grid = np.concatenate([
        _prior_weight_grid[mask],
        np.array([np.interp(ZFORM_MAX, _zgrid, _prior_weight_grid)]),
    ])
    norm = np.trapz(w_norm_grid, z_norm_grid)
    weight = np.interp(z_form, _zgrid, _prior_weight_grid, left=0.0, right=0.0)
    return float(np.log(weight) - np.log(norm)) if weight > 0.0 and norm > 0.0 else -np.inf


def compute_log_q_proposal(
    theta: np.ndarray,
    z_form: float,
    log10_Z: float,
    kick_sigma: float = KICK_PROPOSAL_SIGMA,
) -> float:
    """Log density of the full injection proposal used by this campaign."""
    from gwbackpop.cosmology import log_prior_logZ_given_z_on_support

    theta = np.asarray(theta, dtype=np.float64)
    log_q = 0.0
    for i, name in enumerate(PARAMS):
        if not (LOWER[i] <= theta[i] <= UPPER[i]):
            return -np.inf
        if name in ("vk1", "vk2"):
            log_q += float(_truncated_maxwell_logpdf(theta[i], kick_sigma, LOWER[i], UPPER[i]))
        elif name in ("phi1", "phi2"):
            log_q += float(_isotropic_phi_logpdf(theta[i], LOWER[i], UPPER[i]))
        elif name == "logZ" and LIKELIHOOD_MODE == "3D":
            # 3D logZ is proposed from P(logZ | z_form), not uniformly.
            continue
        elif name == "z_form" and LIKELIHOOD_MODE == "3D":
            # 3D z_form is proposed from the SFR prior below, not uniformly.
            continue
        else:
            log_q += float(-np.log(UPPER[i] - LOWER[i]))
    log_q += float(_log_q_z_form(z_form))
    if LIKELIHOOD_MODE == "3D":
        log_q += float(log_prior_logZ_given_z_on_support(log10_Z, z_form, LOGZ_LO, LOGZ_HI))
    return float(log_q)


def _draw_z_form(rng: np.random.Generator, max_tries: int = 200) -> float | None:
    """Draw z_form from P(z_form) ∝ (dV_c/dz) ψ(z) / (1+z) via rejection sampling.

    Uses a Uniform[0, ZFORM_MAX] proposal with the prior as the accept/reject weight.
    The normalised prior peaks near z~2 with max value computable from the table.
    """
    from gwbackpop.cosmology import _prior_weight_grid, _zgrid, _prior_norm

    # Maximum of the (unnormalised) prior weight — precomputed from table
    log_max = float(np.log(_prior_weight_grid.max()))

    for _ in range(max_tries):
        z_try = rng.uniform(0.0, ZFORM_MAX)
        log_p = float(
            np.interp(z_try, _zgrid, np.log(np.clip(_prior_weight_grid, 1e-300, None)))
        )
        log_u = np.log(rng.uniform(0, 1))
        if log_u <= log_p - log_max:
            return float(z_try)
    return None


def _draw_logZ_given_z(rng: np.random.Generator, z_form: float,
                        max_tries: int = 50) -> float | None:
    """Draw logZ from P(logZ | z_form), normalized on active config support."""
    del max_tries
    from gwbackpop.cosmology import draw_logZ_given_z_on_support

    return draw_logZ_given_z_on_support(rng, z_form, LOGZ_LO, LOGZ_HI)


# ---------------------------------------------------------------------------
# Campaign runner
# ---------------------------------------------------------------------------

def resolve_pdet_callable(
    pdet_mode: str = "auto",
    pdet_path: str | None = None,
    *,
    snr_pdet_method: str = "orientation_monte_carlo",
    snr_threshold: float = 10.0,
    snr_ref: float = 20.0,
    snr_mc_ref: float = 26.1,
    snr_d_ref_mpc: float = 1000.0,
    snr_sensitivity_scale: float = 1.0,
    snr_logistic_width: float = 1.0,
    snr_orientation_seed: int = 1234,
    snr_n_orientation: int = 200000,
):
    """Resolve injection pdet mode into a callable/path plus metadata."""
    mode = str(pdet_mode).lower()
    if mode == "auto":
        mode = "pickle" if pdet_path else "none"
    if mode not in {"none", "pickle", "snr_proxy"}:
        raise ValueError("pdet_mode must be one of auto, none, pickle, snr_proxy")
    metadata = {"pdet_mode": mode}
    if mode == "none":
        return None, None, metadata
    if mode == "pickle":
        if not pdet_path:
            raise ValueError("pdet_mode='pickle' requires --pdet_path")
        return pdet_path, None, metadata
    if pdet_path:
        warnings.warn("pdet_mode='snr_proxy' ignores --pdet_path", RuntimeWarning)
    from gwbackpop.selection.snr_pdet import make_snr_proxy_pdet_callable
    callable_ = make_snr_proxy_pdet_callable(
        method=snr_pdet_method, rho_threshold=snr_threshold, rho_ref=snr_ref,
        mc_ref=snr_mc_ref, d_ref_mpc=snr_d_ref_mpc,
        sensitivity_scale=snr_sensitivity_scale, logistic_width=snr_logistic_width,
        seed=snr_orientation_seed, n_orientation=snr_n_orientation,
    )
    metadata.update(
        pdet_model_name="semi_analytic_snr_proxy",
        snr_pdet_method=snr_pdet_method, snr_threshold=float(snr_threshold),
        snr_ref=float(snr_ref), snr_mc_ref=float(snr_mc_ref),
        snr_d_ref_mpc=float(snr_d_ref_mpc),
        snr_sensitivity_scale=float(snr_sensitivity_scale),
        snr_logistic_width=float(snr_logistic_width),
        snr_orientation_seed=int(snr_orientation_seed),
        snr_n_orientation=int(snr_n_orientation),
        pdet_is_diagnostic=True, pdet_production_ready=False,
    )
    return None, callable_, metadata


def run_campaign(
    pdet_path: str | None,
    output_path: str,
    n_inj: int,
    n_workers: int,
    chunk_size: int = 500,
    config_name: str | None = None,
    likelihood_mode: str = "2D",
    pdet_mode: str = "auto",
    snr_pdet_method: str = "orientation_monte_carlo",
    snr_threshold: float = 10.0,
    snr_ref: float = 20.0,
    snr_mc_ref: float = 26.1,
    snr_d_ref_mpc: float = 1000.0,
    snr_sensitivity_scale: float = 1.0,
    snr_logistic_width: float = 1.0,
    snr_orientation_seed: int = 1234,
    snr_n_orientation: int = 200000,
    debug_failures: bool = False,
    proposal_mode: str = "broad",
    hyperposterior_path: str | None = None,
    broad_mixture_fraction: float = 0.1,
    mixture_component_count: int = 512,
) -> None:
    """Run the full injection campaign and save results.

    Parameters
    ----------
    pdet_path : str
        Path to pickled P_det interpolator callable: f(m1_src, m2_src, z) → [0,1].
    output_path : str
        Output .npz path.
    n_inj : int
        Total number of draws from the prior (including non-merging).
    n_workers : int
        Parallel worker processes.
    chunk_size : int
        Seeds per Pool.map call — balances overhead vs memory.
    """
    config_name = config_name or DEFAULT_CONFIG_NAME
    global LOWER, UPPER, PARAMS, FIXED_PARAMS, LIKELIHOOD_MODE, DEBUG_FAILURES, PROPOSAL_MODE, BROAD_MIXTURE_FRACTION, HYPERPOSTERIOR_COMPONENTS, MIXTURE_COMPONENT_COUNT, HYPERPOSTERIOR_PATH
    LOWER, UPPER, PARAMS, FIXED_PARAMS = _load_config(config_name)
    require_supported_cosmic_for_independent_alpha_flim(
        config_name=config_name,
        params=PARAMS,
    )
    LIKELIHOOD_MODE = str(likelihood_mode).upper()
    DEBUG_FAILURES = bool(debug_failures)
    if LIKELIHOOD_MODE not in {"2D", "3D"}:
        raise ValueError("likelihood_mode must be 2D or 3D")
    PROPOSAL_MODE = str(proposal_mode)
    if PROPOSAL_MODE not in {"broad", "adaptive_hyperposterior"}:
        raise ValueError("proposal_mode must be broad or adaptive_hyperposterior")
    BROAD_MIXTURE_FRACTION = float(broad_mixture_fraction)
    if not (0.0 <= BROAD_MIXTURE_FRACTION <= 1.0):
        raise ValueError("broad_mixture_fraction must be in [0, 1]")
    HYPERPOSTERIOR_PATH = hyperposterior_path
    HYPERPOSTERIOR_COMPONENTS = None
    MIXTURE_COMPONENT_COUNT = 0
    if PROPOSAL_MODE == "adaptive_hyperposterior":
        if not hyperposterior_path:
            raise ValueError("adaptive_hyperposterior requires hyperposterior_path")
        HYPERPOSTERIOR_COMPONENTS = _subsample_hyperposterior(
            load_hyperposterior_samples(hyperposterior_path), int(mixture_component_count)
        )
        MIXTURE_COMPONENT_COUNT = int(len(HYPERPOSTERIOR_COMPONENTS))

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    pdet_path_resolved, pdet_callable, pdet_metadata = resolve_pdet_callable(
        pdet_mode, pdet_path, snr_pdet_method=snr_pdet_method,
        snr_threshold=snr_threshold, snr_ref=snr_ref, snr_mc_ref=snr_mc_ref,
        snr_d_ref_mpc=snr_d_ref_mpc, snr_sensitivity_scale=snr_sensitivity_scale,
        snr_logistic_width=snr_logistic_width, snr_orientation_seed=snr_orientation_seed,
        snr_n_orientation=snr_n_orientation,
    )
    start = time.time()

    # Seeds: deterministic, reproducible, one per injection
    seeds = np.arange(n_inj, dtype=np.int64)

    results = []
    failures = Counter()
    failure_examples = []
    n_done  = 0

    print(f"[injections] n_inj={n_inj:,}  n_workers={n_workers}")
    print(
        f"[injections] WARNING: proposal_version={PROPOSAL_VERSION} uses "
        "truncated Maxwellian kick speeds and isotropic COSMIC kick directions; "
        "do not combine its log_q_proposal with older proposal versions."
    )
    print(f"[injections] pdet_mode: {pdet_metadata['pdet_mode']}")
    print(f"[injections] proposal_mode: {PROPOSAL_MODE}")
    print(f"[injections] pdet: {pdet_path_resolved}")
    print(f"[injections] output: {output_path}")
    if "jax" in sys.modules:
        print("[injections] WARNING: jax is already imported before multiprocessing fork; "
              "JAX is multithreaded and may emit os.fork RuntimeWarning.")
    print()

    with Pool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(pdet_path_resolved, config_name, LIKELIHOOD_MODE, pdet_callable, debug_failures,
                  PROPOSAL_MODE, BROAD_MIXTURE_FRACTION, HYPERPOSTERIOR_COMPONENTS, HYPERPOSTERIOR_PATH),
    ) as pool:
        for i in range(0, n_inj, chunk_size):
            batch = seeds[i : i + chunk_size]
            batch_results = pool.map(_run_one, batch)
            for r in batch_results:
                if r is None:
                    continue
                if r.get("ok", True):
                    results.append(r)
                else:
                    reason = str(r.get("reason", "unknown_failure"))
                    failures[reason] += 1
                    if len(failure_examples) < MAX_DEBUG_EXAMPLES:
                        failure_examples.append(r)
            n_done += len(batch)

            if n_done % max(chunk_size * 20, 10_000) == 0 or n_done == n_inj:
                n_merge = len(results)
                frac    = n_merge / n_done
                elapsed = time.time() - start
                rate    = n_done / elapsed
                eta     = (n_inj - n_done) / rate if rate > 0 else 0
                pdet_mean = np.mean([r['pdet'] for r in results]) if results else 0
                print(
                    f"  {n_done:>9,}/{n_inj:,}  "
                    f"N_merge={n_merge:>6,}  f_merge={frac:.4f}  "
                    f"<Pdet>={pdet_mean:.3f}  "
                    f"elapsed={elapsed/60:.1f}min  ETA={eta/60:.1f}min"
                )

    if debug_failures:
        print()
        print("[injections] Failure histogram:")
        if failures:
            for reason, count in failures.most_common():
                print(f"  {reason:>24s}: {count:,}")
        else:
            print("  (no rejected injections recorded)")
        for idx, failure in enumerate(failure_examples, start=1):
            print(f"[injections] Failure example {idx}: {failure.get('reason')}")
            if failure.get("message"):
                print(f"  message: {failure['message']}")
            if failure.get("traceback"):
                print(failure["traceback"].rstrip())

    n_merge = len(results)
    if n_merge == 0:
        raise RuntimeError(
            "No merging BBHs found in injection campaign. "
            "Check COSMIC flags and prior bounds."
        )

    # ---- Pack into arrays ----
    theta      = np.array([r['theta']    for r in results])   # (N_merge, N_params)
    logZ_arr   = np.array([r['logZ']     for r in results])
    z_form_arr = np.array([r['z_form']   for r in results])
    m1_src     = np.array([r['m1_src']   for r in results])
    m2_src     = np.array([r['m2_src']   for r in results])
    z_merger   = np.array([r['z_merger'] for r in results])
    t_delay    = np.array([r['t_delay']  for r in results])
    pdet       = np.array([r['pdet']     for r in results])
    log_q_proposal = np.array([r['log_q_proposal'] for r in results])

    elapsed = time.time() - start

    # ---- Summary statistics ----
    print()
    print(f"[injections] Done in {elapsed/3600:.2f} hr")
    print(f"  N_inj          = {n_inj:,}")
    print(f"  N_merge        = {n_merge:,}")
    print(f"  f_merge        = {n_merge/n_inj:.4f}")
    using_pdet = not np.all(np.isnan(pdet))
    if using_pdet:
        print(f"  <P_det>        = {pdet.mean():.4f}")
        print(f"  N_eff_detected = {(pdet.sum()**2 / (pdet**2).sum()):.0f}")
        if HYPERPOSTERIOR_COMPONENTS is not None:
            from gwbackpop.inference.hierarchical import compute_direct_pdet_selection_weight_diagnostics
            cosmic_diag = dict(
                theta=theta, params=PARAMS, lo=LOWER, hi=UPPER, kick_sigma=KICK_PROPOSAL_SIGMA,
                log_q_proposal=log_q_proposal, log_pop_static=np.zeros(len(theta)),
                pdet=pdet, N_inj=n_inj, N_merge=n_merge,
            )
            selection_diag = compute_direct_pdet_selection_weight_diagnostics(
                np.median(HYPERPOSTERIOR_COMPONENTS, axis=0), "no_selection_posterior_median", cosmic_diag
            )
            print(f"  pdet-positive  = {selection_diag['positive_pdet_count']:,}")
            print(f"  sel ESS@median = {selection_diag['injection_ess']:.1f}")
            print(f"  top1 alpha frac= {selection_diag['top_1pct_alpha_fraction']:.4f}")
            print(f"  top0.1 alpha   = {selection_diag['top_0p1pct_alpha_fraction']:.4f}")
        else:
            selection_diag = {}
    else:
        print("  P_det: not evaluated (LVK raw-injection mode — pdet=nan stored)")
        selection_diag = {}
    print(f"  m1_src range   = [{m1_src.min():.1f}, {m1_src.max():.1f}] M_sun")
    print(f"  z_merger range = [{z_merger.min():.3f}, {z_merger.max():.3f}]")

    proposal_distribution_description = (
        "Uniform over non-kick BackPop box parameters; vk1/vk2 from exactly "
        f"truncated Maxwell(scale={KICK_PROPOSAL_SIGMA} km/s) over configured bounds; "
        "COSMIC kick directions isotropic via uniform sin(phi), uniform theta, "
        "uniform omega; z_form from SFR-weighted comoving-volume prior on "
        f"[0,{ZFORM_MAX}]; logZ uniform for 2D mode and P(logZ|z_form) for 3D mode."
    )
    cosmic_capabilities = inspect_cosmic_capabilities()
    metadata = dict(
        **base_runtime_metadata("."),
        package_versions=get_package_versions(["numpy", "scipy", "astropy", "cosmic"]),
        cosmic_capabilities=cosmic_capabilities,
        cosmic_popsynth_version=cosmic_capabilities["cosmic_popsynth_version"],
        supports_independent_alpha=cosmic_capabilities["supports_independent_alpha"],
        supports_independent_flim=cosmic_capabilities["supports_independent_flim"],
        supports_cosmic410_evolv2_signature=cosmic_capabilities["supports_cosmic410_evolv2_signature"],
        supported_for_independent_alpha_flim=cosmic_capabilities["supported_for_independent_alpha_flim"],
        config_name=config_name,
        proposal_version=PROPOSAL_VERSION,
        proposal_name=PROPOSAL_NAME,
        proposal_mode=PROPOSAL_MODE,
        broad_mixture_fraction=float(BROAD_MIXTURE_FRACTION),
        hyperposterior_path=HYPERPOSTERIOR_PATH,
        mixture_component_count=int(MIXTURE_COMPONENT_COUNT),
        proposal_metadata=(
            "adaptive_hyperposterior draws alpha/flim/vk from a no-selection "
            "hyperposterior mixture and evaluates exact broad/adaptive log_q via logsumexp; "
            "all other proposal factors follow the selected 2D/3D injection mode."
            if PROPOSAL_MODE == "adaptive_hyperposterior" else proposal_distribution_description
        ),
        adaptive_selection_diagnostics=selection_diag,
        proposal_distribution_description=proposal_distribution_description,
        log_q_proposal_available=bool(np.all(np.isfinite(log_q_proposal))),
        fixed_parameters=FIXED_PARAMS,
        n_total_injections=int(n_inj),
        n_merging_injections=int(n_merge),
        random_seed_convention="Deterministic one-to-one seeds: numpy default_rng(seed=i) for injection draw i in [0, n_inj).",
        likelihood_mode=LIKELIHOOD_MODE,
        uses_z_form=bool(LIKELIHOOD_MODE == "3D"),
        uses_aux_z_form=True,
        aux_z_form_proposal="sfr_weighted_comoving_volume",
        aux_z_form_distribution="log_q_proposal_includes_sfr_prior_density",
        uses_sfr_prior=bool(LIKELIHOOD_MODE == "3D"),
        uses_logZ_given_z_prior=bool(LIKELIHOOD_MODE == "3D"),
        logZ_support=[float(LOGZ_LO), float(LOGZ_HI)],
        coordinate_system=COORDINATE_SYSTEM,
        pdet_path=pdet_path_resolved,
        **pdet_metadata,
        n_workers=int(n_workers),
        chunk_size=int(chunk_size),
        wall_time_s=float(elapsed),
    )

    # ---- Save ----
    np.savez(
        output_path,
        theta        = theta,
        logZ         = logZ_arr,
        z_form       = z_form_arr,
        m1_src       = m1_src,
        m2_src       = m2_src,
        z_merger     = z_merger,
        t_delay_myr  = t_delay,
        pdet         = pdet,
        log_q_proposal = log_q_proposal,
        params       = PARAMS,
        lower_bound  = LOWER,
        upper_bound  = UPPER,
        fixed_params = np.array(FIXED_PARAMS, dtype=object),
        N_inj                = np.array([n_inj]),
        N_merge              = np.array([n_merge]),
        N_workers            = np.array([n_workers]),
        kick_proposal_sigma  = np.array([KICK_PROPOSAL_SIGMA]),
        proposal_name         = np.array([PROPOSAL_NAME]),
        proposal_mode         = np.array([PROPOSAL_MODE]),
        broad_mixture_fraction = np.array([BROAD_MIXTURE_FRACTION]),
        hyperposterior_path   = np.array([HYPERPOSTERIOR_PATH or ""]),
        mixture_component_count = np.array([MIXTURE_COMPONENT_COUNT]),
        proposal_version      = np.array([PROPOSAL_VERSION]),
        coordinate_system     = np.array([COORDINATE_SYSTEM]),
        config_name           = np.array([config_name]),
        likelihood_mode        = np.array([LIKELIHOOD_MODE]),
        uses_z_form           = np.array([LIKELIHOOD_MODE == "3D"]),
        uses_aux_z_form       = np.array([True]),
        aux_z_form_proposal   = np.array(["sfr_weighted_comoving_volume"]),
        aux_z_form_distribution = np.array(["log_q_proposal_includes_sfr_prior_density"]),
        uses_sfr_prior        = np.array([LIKELIHOOD_MODE == "3D"]),
        uses_logZ_given_z_prior = np.array([LIKELIHOOD_MODE == "3D"]),
        wall_time_s          = np.array([elapsed]),
        pdet_mode           = np.array([pdet_metadata["pdet_mode"]]),
        pdet_model_name      = np.array([pdet_metadata.get("pdet_model_name", "")]),
        snr_pdet_method      = np.array([pdet_metadata.get("snr_pdet_method", "")]),
        snr_threshold        = np.array([pdet_metadata.get("snr_threshold", np.nan)]),
        snr_ref              = np.array([pdet_metadata.get("snr_ref", np.nan)]),
        snr_mc_ref           = np.array([pdet_metadata.get("snr_mc_ref", np.nan)]),
        snr_d_ref_mpc        = np.array([pdet_metadata.get("snr_d_ref_mpc", np.nan)]),
        snr_sensitivity_scale = np.array([pdet_metadata.get("snr_sensitivity_scale", np.nan)]),
        snr_logistic_width   = np.array([pdet_metadata.get("snr_logistic_width", np.nan)]),
        snr_orientation_seed = np.array([pdet_metadata.get("snr_orientation_seed", -1)]),
        snr_n_orientation    = np.array([pdet_metadata.get("snr_n_orientation", -1)]),
        pdet_is_diagnostic   = np.array([pdet_metadata.get("pdet_is_diagnostic", False)]),
        pdet_production_ready = np.array([pdet_metadata.get("pdet_production_ready", pdet_metadata["pdet_mode"] == "pickle")]),
        cosmic_popsynth_version = np.array([cosmic_capabilities["cosmic_popsynth_version"] or ""]),
        supports_independent_alpha = np.array([cosmic_capabilities["supports_independent_alpha"]]),
        supports_independent_flim = np.array([cosmic_capabilities["supports_independent_flim"]]),
        supports_cosmic410_evolv2_signature = np.array([cosmic_capabilities["supports_cosmic410_evolv2_signature"]]),
        supported_for_independent_alpha_flim = np.array([cosmic_capabilities["supported_for_independent_alpha_flim"]]),
        metadata             = np.array(metadata, dtype=object),
    )
    catalog_path = os.fspath(output_path)
    from pathlib import Path
    metadata_path = Path(catalog_path).with_name(Path(catalog_path).stem + "_metadata.npz")
    save_metadata(metadata_path, metadata, overwrite_existing_npz=True)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _str_to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).lower()
    if lowered in ("true", "t", "1", "yes", "y"):
        return True
    if lowered in ("false", "f", "0", "no", "n"):
        return False
    raise ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def parse_args():
    p = ArgumentParser(description="BackPop injection campaign for selection effects.")
    p.add_argument("--pdet_path",   default=None,
                   help="Path to pickled P_det(m1_src, m2_src, z) interpolator.")
    p.add_argument("--pdet_mode", choices=["auto", "none", "pickle", "snr_proxy"], default="auto",
                   help="P_det source: auto uses pickle when --pdet_path is set, else none; snr_proxy is diagnostic only.")
    p.add_argument("--snr_pdet_method", choices=["hard_threshold", "orientation_monte_carlo", "logistic"], default="orientation_monte_carlo")
    p.add_argument("--snr_threshold", type=float, default=10.0)
    p.add_argument("--snr_ref", type=float, default=20.0)
    p.add_argument("--snr_mc_ref", type=float, default=26.1)
    p.add_argument("--snr_d_ref_mpc", type=float, default=1000.0)
    p.add_argument("--snr_sensitivity_scale", type=float, default=1.0)
    p.add_argument("--snr_logistic_width", type=float, default=1.0)
    p.add_argument("--snr_orientation_seed", type=int, default=1234)
    p.add_argument("--snr_n_orientation", type=int, default=200000)
    p.add_argument("--output_path", required=True,
                   help="Output NPZ file path.")
    p.add_argument("--n_inj",       type=int, default=1_000_000,
                   help="Total injection draws (default 1e6).")
    p.add_argument("--n_workers",   type=int, default=None,
                   help="Worker processes (default: all available CPUs).")
    p.add_argument("--chunk_size",  type=int, default=500,
                   help="Seeds per pool.map call (default 500).")
    p.add_argument("--dry_run",     type=str, default='False',
                   help="If True, run n_inj=10000 to estimate merger fraction only.")
    p.add_argument("--debug_failures", type=_str_to_bool, default=False,
                   help="If True, record and print rejection reasons and COSMIC exception tracebacks.")
    p.add_argument("--config_name", default=DEFAULT_CONFIG_NAME,
                   help="BackPop config name whose params/bounds are used for injections (default: lucky_strikes).")
    p.add_argument("--likelihood_mode", choices=["2D", "3D"], default="2D",
                   help="Selection base measure metadata/proposal: 2D uses flat logZ and no population z_form; 3D uses SFR z_form and P(logZ|z_form).")
    p.add_argument("--proposal_mode", choices=["broad", "adaptive_hyperposterior"], default="broad",
                   help="Injection proposal: original broad proposal or mixture adapted to a no-selection hyperposterior.")
    p.add_argument("--hyperposterior_path", default=None,
                   help="No-selection samples.npz containing POP_PARAM_NAMES samples for adaptive_hyperposterior.")
    p.add_argument("--broad_mixture_fraction", type=float, default=0.1,
                   help="Mixture weight for the original broad proposal in adaptive_hyperposterior mode.")
    p.add_argument("--mixture_component_count", type=int, default=512,
                   help="Number of hyperposterior components to use when evaluating the adaptive mixture density.")
    return p.parse_args()


def main():
    opts = parse_args()
    dry  = opts.dry_run.lower() in ('true', 't', '1', 'yes')
    debug_failures = bool(opts.debug_failures)
    n    = 10_000 if dry else opts.n_inj
    nw   = opts.n_workers or cpu_count()

    if dry:
        print("[injections] DRY RUN — estimating merger fraction with 10,000 draws")

    run_campaign(
        pdet_path   = opts.pdet_path,
        output_path = opts.output_path,
        n_inj       = n,
        n_workers   = nw,
        chunk_size  = opts.chunk_size,
        config_name  = opts.config_name,
        likelihood_mode = opts.likelihood_mode,
        pdet_mode = opts.pdet_mode,
        snr_pdet_method = opts.snr_pdet_method,
        snr_threshold = opts.snr_threshold,
        snr_ref = opts.snr_ref,
        snr_mc_ref = opts.snr_mc_ref,
        snr_d_ref_mpc = opts.snr_d_ref_mpc,
        snr_sensitivity_scale = opts.snr_sensitivity_scale,
        snr_logistic_width = opts.snr_logistic_width,
        snr_orientation_seed = opts.snr_orientation_seed,
        snr_n_orientation = opts.snr_n_orientation,
        debug_failures = debug_failures,
        proposal_mode = opts.proposal_mode,
        hyperposterior_path = opts.hyperposterior_path,
        broad_mixture_fraction = opts.broad_mixture_fraction,
        mixture_component_count = opts.mixture_component_count,
    )


if __name__ == "__main__":
    main()
