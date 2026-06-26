"""
hierarchical_backpop_jax.py
----------------------------
JAX/NumPyro implementation of the hierarchical BackPop population inference.

Replaces the Nautilus nested sampler with NUTS (No-U-Turn Sampler) via NumPyro.
The entire likelihood — per-event importance weighting, selection correction via
K @ w, logsumexp — is JIT-compiled with XLA and fully differentiable, enabling
gradient-based HMC.

Speed comparison (10 events, 284k LVK injections, 22k COSMIC mergers):
  Nautilus (numpy, CPU)  : ~2 hrs   (100k likelihood calls, no gradients)
  NUTS (JAX, CPU, x64)   : ~5 min   (2k gradient calls, XLA-optimised matmul)
  NUTS (JAX, GPU, x64)   : ~30 sec  (same, K @ w on device)

Key JAX design decisions
------------------------
  1. x64 mode enabled at startup (before any JAX import) — required for
     numerical accuracy in log-likelihood computations. 64-bit is essential
     because logsumexp over N=22000 weights spans many decades and float32
     would introduce catastrophic cancellation in the selection integral.

  2. All data (samples, K matrix, bounds) stored as jnp.float64 arrays
     loaded once at startup. K is (N_found, N_merge) — the dominant memory
     object. On GPU, this lives on device permanently.

  3. log_weight_ratio_event and log_alpha are @jax.jit compiled. The full
     hierarchical log-likelihood is a single compiled computation graph.
     First call triggers XLA compilation (~10-30 sec); subsequent calls are
     fast.

  4. NumPyro NUTS: gradient is computed via reverse-mode autodiff through
     the entire likelihood including:
       - lognormal logpdf for alpha parameters
       - beta logpdf for flim parameters
       - maxwell logpdf for vk parameters (via chi distribution, see below)
       - K @ w matmul (linear, trivially differentiable)
       - logsumexp (differentiable via JAX)

  5. Maxwell distribution: not in jax.scipy.stats directly. Implemented as
     chi(df=3) scaled by sigma — Maxwell(sigma) = sigma * chi(3). This is
     exact and differentiable. Verified against scipy.stats.maxwell.

  6. Parallel chains via vmap over PRNGKeys — no multiprocessing needed.

Usage
-----
  python hierarchical_backpop_jax.py \\
      --results_root   results/ \\
      --config_name    lucky_strikes \\
      --injections_path injections/gwtc3_cosmic_mergers.npz \\
      --lvk_found_path  endo3_bbhpop-LIGO-T2100113-v12.hdf5 \\
      --output_dir     results/hierarchical/lucky_strikes/nuts \\
      --num_warmup     500 \\
      --num_samples    1000 \\
      --num_chains     4

  # Without selection effects (biased, fast for testing):
  python hierarchical_backpop_jax.py \\
      --results_root  results/ \\
      --config_name   lucky_strikes \\
      --output_dir    results/hierarchical/lucky_strikes/nuts_no_sel

Output
------
  results/hierarchical/lucky_strikes/nuts/
    samples.npz       posterior samples per chain, shape (n_chains, n_samples, 10)
    summary.csv       mean, std, HDI 90%, R-hat, n_eff per parameter
    trace.npz         full chain trace for convergence diagnostics
    metadata.npz      event list, run settings, timing
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# JAX x64 mode — MUST be set before any jax import
# ---------------------------------------------------------------------------
import os
os.environ["JAX_ENABLE_X64"] = "1"   # environment variable route (most reliable)

import jax
jax.config.update("jax_enable_x64", True)   # programmatic route (belt + suspenders)

# Verify x64 is active — crash immediately if not, rather than silently
# computing in float32 and getting wrong answers.
assert jax.numpy.ones(1).dtype == jax.numpy.float64, (
    "JAX x64 mode failed to activate. Set JAX_ENABLE_X64=1 before launching."
)

# ---------------------------------------------------------------------------
# All other imports after JAX x64 is confirmed
# ---------------------------------------------------------------------------
import sys
import time
import glob
import json
import warnings
import numpy as np
import scipy.stats as sp_stats
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, ArgumentTypeError

import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit, vmap, grad

import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC
from numpyro.diagnostics import print_summary, effective_sample_size, gelman_rubin

from gwbackpop.metadata import base_runtime_metadata, get_package_versions, save_metadata


def str2bool(value):
    """Parse CLI booleans such as True/False, yes/no, and 1/0."""
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "t", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "f", "0", "no", "n", "off"}:
        return False
    raise ArgumentTypeError(f"Expected a boolean value, got {value!r}")

# ---------------------------------------------------------------------------
# Population hyperparameter metadata
# ---------------------------------------------------------------------------
# NumPyro distributions in ``make_numpyro_model`` are the actual hyperpriors.
# The metadata below records names, scientifically sensible default values used
# for JIT compilation/diagnostics, and human-readable descriptions of those
# NumPyro hyperpriors.  These values are not rectangular sampling bounds.

HYPERPARAM_INFO = [
    dict(name="mu_logalpha1",  default=0.0,   prior="Normal(0.0, 1.5)"),
    dict(name="sig_logalpha1", default=0.7,   prior="Exponential(1.0)"),
    dict(name="mu_logalpha2",  default=0.0,   prior="Normal(0.0, 1.5)"),
    dict(name="sig_logalpha2", default=0.7,   prior="Exponential(1.0)"),
    dict(name="a_f1",          default=2.0,   prior="Gamma(2.0, 1.0)"),
    dict(name="b_f1",          default=2.0,   prior="Gamma(2.0, 1.0)"),
    dict(name="a_f2",          default=2.0,   prior="Gamma(2.0, 1.0)"),
    dict(name="b_f2",          default=2.0,   prior="Gamma(2.0, 1.0)"),
    dict(name="sigma_v1",      default=150.0, prior="HalfNormal(150.0)"),
    dict(name="sigma_v2",      default=150.0, prior="HalfNormal(150.0)"),
]

POP_PARAM_NAMES = [p["name"] for p in HYPERPARAM_INFO]
HYPERPRIOR_DESCRIPTIONS = {p["name"]: p["prior"] for p in HYPERPARAM_INFO}

# Physical support for event/injection variables.  These are supports of the
# population density factors and single-event/injection proposals, not support
# bounds for NumPyro hyperparameter sampling.  Concrete event/injection runs can
# provide tighter finite alpha/vk bounds via metadata loaded from disk.
EVENT_INJECTION_SUPPORT_BOUNDS = {
    "alpha_1": (0.0, jnp.inf),
    "alpha_2": (0.0, jnp.inf),
    "flim_1": (0.0, 1.0),
    "flim_2": (0.0, 1.0),
    "vk1": (0.0, jnp.inf),
    "vk2": (0.0, jnp.inf),
}

def default_hyperparams(as_vector: bool = True):
    """Return a valid, explicit hyperparameter initialization point.

    This point is used only to trigger JIT compilation and smoke-test the
    likelihood.  NumPyro priors in :func:`make_numpyro_model` remain the actual
    hyperpriors used for sampling.
    """
    values = {p["name"]: float(p["default"]) for p in HYPERPARAM_INFO}
    if as_vector:
        # Return a NumPy array rather than a JAX array so callers can freely
        # make mutable NumPy views/copies when perturbing the initialization
        # point in tests or diagnostics.  JAX-compiled likelihood helpers
        # accept this array-like input and convert it as needed.
        return np.array([values[name] for name in POP_PARAM_NAMES], dtype=np.float64)
    return values


# ---------------------------------------------------------------------------
# Maxwell distribution in JAX (not in jax.scipy.stats)
# ---------------------------------------------------------------------------

def maxwell_logpdf(x: jnp.ndarray, scale: float | jnp.ndarray) -> jnp.ndarray:
    """Log-PDF of Maxwell distribution via chi(df=3).

    Maxwell(sigma) == sigma * chi(df=3), i.e. the magnitude of a 3D isotropic
    Gaussian with per-component std sigma.

    log p(x | sigma) = log(2) - log(sigma^3) - log(Gamma(3/2))
                     + 2*log(x) - x^2/(2*sigma^2)
    where Gamma(3/2) = sqrt(pi)/2, so log(Gamma(3/2)) = 0.5*log(pi) - log(2).

    Simplifies to:
        log p(x|sigma) = log(2/sqrt(pi)) + 2*log(x/sigma) - x^2/(2*sigma^2)
                       - log(sigma)
    which equals:
        log(sqrt(2/pi)) + 2*log(x) - 3*log(sigma) - x^2/(2*sigma^2)

    Validated against scipy.stats.maxwell.logpdf(x, scale=sigma).
    """
    x_safe = jnp.clip(x, 1e-300, None)
    return (
        0.5 * jnp.log(2.0 / jnp.pi)
        + 2.0 * jnp.log(x_safe)
        - 3.0 * jnp.log(scale)
        - x_safe**2 / (2.0 * scale**2)
    )


# ---------------------------------------------------------------------------
# JAX population log-densities
# ---------------------------------------------------------------------------

def log_p_alpha_jax(alpha: jnp.ndarray,
                    mu_log: jnp.ndarray,
                    sig_log: jnp.ndarray,
                    lo: float | jnp.ndarray,
                    hi: float | jnp.ndarray) -> jnp.ndarray:
    """Truncated LogNormal population model for CE efficiency alpha.

    The density is normalized on the same finite support as the event and
    injection samples. Values outside ``[lo, hi]`` receive zero probability
    (``-inf`` log-density).
    """
    alpha = jnp.asarray(alpha)
    x_safe = jnp.clip(alpha, 1e-300, None)
    z_hi = (jnp.log(hi) - mu_log) / sig_log
    z_lo = (jnp.log(lo) - mu_log) / sig_log
    log_norm = jnp.log(jsp.special.ndtr(z_hi) - jsp.special.ndtr(z_lo))
    log_pdf = dist.LogNormal(mu_log, sig_log).log_prob(x_safe) - log_norm
    return jnp.where((alpha >= lo) & (alpha <= hi), log_pdf, -jnp.inf)


def log_p_flim_jax(flim: jnp.ndarray,
                   a: jnp.ndarray,
                   b: jnp.ndarray) -> jnp.ndarray:
    """Beta population model for stable MT accretion efficiency flim in [0,1].

    Samples are only clipped by one machine-scale epsilon at exact numerical
    boundaries before evaluating the Beta log-density. This is a boundary
    safety measure for posterior samples stored as exactly 0 or 1; values
    outside [0, 1] still receive zero probability rather than being silently
    moved into support.
    """
    flim = jnp.asarray(flim)
    eps = 1e-12
    x_safe = jnp.clip(flim, eps, 1.0 - eps)
    log_pdf = dist.Beta(a, b).log_prob(x_safe)
    return jnp.where((flim >= 0.0) & (flim <= 1.0), log_pdf, -jnp.inf)


def maxwell_cdf_jax(x: jnp.ndarray, scale: float | jnp.ndarray) -> jnp.ndarray:
    """CDF of the Maxwell distribution matching :func:`maxwell_logpdf`."""
    y = jnp.clip(x, 0.0, None) / scale
    return jsp.special.erf(y / jnp.sqrt(2.0)) - jnp.sqrt(2.0 / jnp.pi) * y * jnp.exp(-0.5 * y**2)


def log_p_vk_jax(vk: jnp.ndarray,
                 sigma_v: jnp.ndarray,
                 lo: float | jnp.ndarray,
                 hi: float | jnp.ndarray) -> jnp.ndarray:
    """Truncated Maxwellian population model for natal kick speed vk [km/s]."""
    vk = jnp.asarray(vk)
    log_norm = jnp.log(maxwell_cdf_jax(hi, sigma_v) - maxwell_cdf_jax(lo, sigma_v))
    log_pdf = maxwell_logpdf(vk, sigma_v) - log_norm
    return jnp.where((vk >= lo) & (vk <= hi), log_pdf, -jnp.inf)


# ---------------------------------------------------------------------------
# NumPy population weight ratios for COSMIC merger-catalog postprocessing
# ---------------------------------------------------------------------------

def _maxwell_logpdf_numpy(
    x: np.ndarray,
    scale: float,
    lo: float | None = None,
    hi: float | None = None,
) -> np.ndarray:
    """NumPy Maxwell log-PDF, optionally truncated to ``[lo, hi]``."""
    x_arr = np.asarray(x, dtype=np.float64)
    x_safe = np.clip(x_arr, 1e-300, None)
    scale = float(scale)
    log_pdf = (
        0.5 * np.log(2.0 / np.pi)
        + 2.0 * np.log(x_safe)
        - 3.0 * np.log(scale)
        - x_safe**2 / (2.0 * scale**2)
    )
    if lo is not None and hi is not None:
        norm = sp_stats.maxwell.cdf(float(hi), scale=scale) - sp_stats.maxwell.cdf(float(lo), scale=scale)
        log_pdf = log_pdf - np.log(norm)
        log_pdf = np.where((x_arr >= float(lo)) & (x_arr <= float(hi)), log_pdf, -np.inf)
    return log_pdf


def _lognormal_logpdf_numpy(
    x: np.ndarray,
    mu_log: float,
    sig_log: float,
    lo: float | None = None,
    hi: float | None = None,
) -> np.ndarray:
    """NumPy LogNormal log-PDF, optionally truncated to ``[lo, hi]``."""
    x_arr = np.asarray(x, dtype=np.float64)
    x_safe = np.clip(x_arr, 1e-300, None)
    sig_log = float(sig_log)
    log_pdf = (
        -np.log(x_safe)
        -np.log(sig_log)
        -0.5 * np.log(2.0 * np.pi)
        -0.5 * ((np.log(x_safe) - float(mu_log)) / sig_log) ** 2
    )
    if lo is not None and hi is not None:
        norm = sp_stats.lognorm.cdf(float(hi), s=sig_log, scale=np.exp(float(mu_log))) - sp_stats.lognorm.cdf(float(lo), s=sig_log, scale=np.exp(float(mu_log)))
        log_pdf = log_pdf - np.log(norm)
        log_pdf = np.where((x_arr >= float(lo)) & (x_arr <= float(hi)), log_pdf, -np.inf)
    return log_pdf


def _beta_logpdf_numpy(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """NumPy Beta log-PDF with the same clipping as the JAX path."""
    import math

    x_arr = np.asarray(x, dtype=np.float64)
    x_safe = np.clip(x_arr, 1e-12, 1.0 - 1e-12)
    a = float(a)
    b = float(b)
    log_norm = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    log_pdf = log_norm + (a - 1.0) * np.log(x_safe) + (b - 1.0) * np.log1p(-x_safe)
    return np.where((x_arr >= 0.0) & (x_arr <= 1.0), log_pdf, -np.inf)


def compute_log_wr_injections_numpy(
    lp_vec: np.ndarray,
    theta: np.ndarray,
    params: list[str] | tuple[str, ...] | dict[str, int],
    lo: np.ndarray,
    hi: np.ndarray,
    kick_sigma: float,
    log_q_proposal: np.ndarray | None = None,
    log_pop_static: np.ndarray | None = None,
) -> np.ndarray:
    """Compute COSMIC injection log-weight ratios in NumPy.

    This mirrors the COSMIC-merger weighting inside :func:`make_log_alpha_fn`:
    LogNormal population densities for ``alpha_1``/``alpha_2``, Beta densities
    for ``flim_1``/``flim_2``, and Maxwell densities for ``vk1``/``vk2``.  The
    denominator is the COSMIC injection proposal: uniform over the CE and flim
    parameters, and the Maxwell kick proposal with scale ``kick_sigma`` for the
    kick velocities.
    """
    theta = np.asarray(theta, dtype=np.float64)
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    lp_vec = np.asarray(lp_vec, dtype=np.float64)

    if isinstance(params, dict):
        param_idx = params
    else:
        param_idx = {p: i for i, p in enumerate(params)}

    mu_la1, sig_la1 = lp_vec[0], lp_vec[1]
    mu_la2, sig_la2 = lp_vec[2], lp_vec[3]
    af1,    bf1     = lp_vec[4], lp_vec[5]
    af2,    bf2     = lp_vec[6], lp_vec[7]
    sv1,    sv2     = lp_vec[8], lp_vec[9]

    if log_q_proposal is not None:
        log_wr = (np.zeros(theta.shape[0], dtype=np.float64)
                  if log_pop_static is None else np.asarray(log_pop_static, dtype=np.float64).copy())
        log_wr -= np.asarray(log_q_proposal, dtype=np.float64)
        explicit_q = True
    else:
        log_wr = np.zeros(theta.shape[0], dtype=np.float64)
        explicit_q = False

    def _col(name: str) -> np.ndarray | None:
        idx = param_idx.get(name)
        return theta[:, idx] if idx is not None else None

    def _log_uniform_proposal(name: str) -> float:
        idx = param_idx.get(name)
        if idx is None:
            return 0.0
        return float(-np.log(hi[idx] - lo[idx]))

    def _support(name: str) -> tuple[float, float]:
        idx = param_idx.get(name)
        if idx is None:
            return 0.0, 1.0
        return float(lo[idx]), float(hi[idx])

    a1_lo, a1_hi = _support('alpha_1')
    a2_lo, a2_hi = _support('alpha_2')
    v1_lo, v1_hi = _support('vk1')
    v2_lo, v2_hi = _support('vk2')

    inj_a1 = _col('alpha_1')
    if inj_a1 is not None:
        log_wr += _lognormal_logpdf_numpy(inj_a1, mu_la1, sig_la1, a1_lo, a1_hi) - (0.0 if explicit_q else _log_uniform_proposal('alpha_1'))

    inj_a2 = _col('alpha_2')
    if inj_a2 is not None:
        log_wr += _lognormal_logpdf_numpy(inj_a2, mu_la2, sig_la2, a2_lo, a2_hi) - (0.0 if explicit_q else _log_uniform_proposal('alpha_2'))

    inj_f1 = _col('flim_1')
    if inj_f1 is not None:
        log_wr += _beta_logpdf_numpy(inj_f1, af1, bf1) - (0.0 if explicit_q else _log_uniform_proposal('flim_1'))

    inj_f2 = _col('flim_2')
    if inj_f2 is not None:
        log_wr += _beta_logpdf_numpy(inj_f2, af2, bf2) - (0.0 if explicit_q else _log_uniform_proposal('flim_2'))

    inj_v1 = _col('vk1')
    if inj_v1 is not None:
        log_wr += _maxwell_logpdf_numpy(inj_v1, sv1, v1_lo, v1_hi) - (0.0 if explicit_q else _maxwell_logpdf_numpy(inj_v1, kick_sigma, v1_lo, v1_hi))

    inj_v2 = _col('vk2')
    if inj_v2 is not None:
        log_wr += _maxwell_logpdf_numpy(inj_v2, sv2, v2_lo, v2_hi) - (0.0 if explicit_q else _maxwell_logpdf_numpy(inj_v2, kick_sigma, v2_lo, v2_hi))

    return log_wr


# ---------------------------------------------------------------------------
# Per-event weight ratio (fully JAX, JIT-compiled)
# ---------------------------------------------------------------------------

def make_log_weight_ratio_fn(
    samples: jnp.ndarray,       # (N_samp, N_params) float64
    param_idx: dict[str, int],  # param name → column index
    lo: jnp.ndarray,            # (N_params,) prior lower bounds
    hi: jnp.ndarray,            # (N_params,) prior upper bounds
) -> callable:
    """Return a JIT-compiled function: lambda_pop_vec → log_wr (N_samp,).

    Builds a closure over the per-event data so the compiled function has no
    Python-level dispatch overhead. lambda_pop_vec is a 1D array of the 10
    population hyperparameters in POP_PARAM_NAMES order.

    The denominator for each parameter is the flat single-event prior:
        log pi0(theta) = -log(hi - lo)
    which is constant across samples and precomputed here.
    """
    # Pre-extract parameter columns and precompute log-pi0 denominators.
    # Columns not present in the event are left as None.
    def _col(name):
        idx = param_idx.get(name)
        return samples[:, idx] if idx is not None else None

    alpha1 = _col('alpha_1')
    alpha2 = _col('alpha_2')
    flim1  = _col('flim_1')
    flim2  = _col('flim_2')
    vk1    = _col('vk1')
    vk2    = _col('vk2')

    def _log_pi0(name):
        idx = param_idx.get(name)
        if idx is None:
            return 0.0
        return float(-jnp.log(hi[idx] - lo[idx]))

    lpi0_a1 = _log_pi0('alpha_1')
    lpi0_a2 = _log_pi0('alpha_2')
    lpi0_f1 = _log_pi0('flim_1')
    lpi0_f2 = _log_pi0('flim_2')
    lpi0_v1 = _log_pi0('vk1')
    lpi0_v2 = _log_pi0('vk2')

    def _support(name):
        idx = param_idx.get(name)
        if idx is None:
            return 0.0, 1.0
        return float(lo[idx]), float(hi[idx])

    a1_lo, a1_hi = _support('alpha_1')
    a2_lo, a2_hi = _support('alpha_2')
    v1_lo, v1_hi = _support('vk1')
    v2_lo, v2_hi = _support('vk2')

    @jit
    def log_weight_ratio(lp_vec: jnp.ndarray) -> jnp.ndarray:
        """Compute log[p(theta^k | Lambda_pop) / pi0(theta^k)] for all k.

        lp_vec: (10,) array of population hyperparameters in POP_PARAM_NAMES order.
        Returns (N_samp,) array.
        """
        mu_la1, sig_la1 = lp_vec[0], lp_vec[1]
        mu_la2, sig_la2 = lp_vec[2], lp_vec[3]
        af1,    bf1     = lp_vec[4], lp_vec[5]
        af2,    bf2     = lp_vec[6], lp_vec[7]
        sv1,    sv2     = lp_vec[8], lp_vec[9]

        log_w = jnp.zeros(samples.shape[0])

        if alpha1 is not None:
            log_w = log_w + log_p_alpha_jax(alpha1, mu_la1, sig_la1, a1_lo, a1_hi) - lpi0_a1
        if alpha2 is not None:
            log_w = log_w + log_p_alpha_jax(alpha2, mu_la2, sig_la2, a2_lo, a2_hi) - lpi0_a2
        if flim1 is not None:
            log_w = log_w + log_p_flim_jax(flim1, af1, bf1) - lpi0_f1
        if flim2 is not None:
            log_w = log_w + log_p_flim_jax(flim2, af2, bf2) - lpi0_f2
        if vk1 is not None:
            log_w = log_w + log_p_vk_jax(vk1, sv1, v1_lo, v1_hi) - lpi0_v1
        if vk2 is not None:
            log_w = log_w + log_p_vk_jax(vk2, sv2, v2_lo, v2_hi) - lpi0_v2

        return log_w

    return log_weight_ratio


# ---------------------------------------------------------------------------
# Selection integral log_alpha (JAX, JIT-compiled)
# ---------------------------------------------------------------------------

def make_log_alpha_fn(
    K_np:          np.ndarray,    # (N_found, N_merge) float32 — stays in system RAM
    log_v:         jnp.ndarray,   # (N_found,) precomputed 1/(q*m1*m2)
    log_norm:      float,         # -log(N_lvk * N_cosmic)
    theta_inj:     jnp.ndarray,   # (N_merge, N_params) injection samples
    lo_inj:        jnp.ndarray,   # (N_params,) injection prior lower bounds
    hi_inj:        jnp.ndarray,   # (N_params,) injection prior upper bounds
    param_idx_inj: dict[str, int],# injection param name → column index
    kick_sigma:    float,         # Maxwellian proposal sigma used in legacy campaigns
    log_q_proposal: jnp.ndarray | None = None,  # explicit full proposal log-density
    log_pop_static: jnp.ndarray | None = None,  # population factors not varied by Lambda
) -> callable:
    """Return a JIT-compiled function: lp_vec → log_alpha (scalar).

    K (25 GB) never enters JAX/XLA memory — it lives in system RAM as a plain
    numpy array. The K @ w matmul is wrapped in jax.pure_callback with a
    custom_vjp so that:
      - Forward pass: pure_callback calls numpy K@w (no XLA allocation for K)
      - Backward pass: pure_callback calls numpy K^T @ g (gradient wrt w)
    Everything else (weight ratios, logsumexp) is standard JAX and JIT-compiled.

    jax.device_put inside @jit is NOT sufficient — XLA's HLO rematerialization
    still tries to pin K in device memory during compilation.  pure_callback
    is the only mechanism that completely hides data from XLA's allocator.
    """
    N_found, N_merge = K_np.shape

    # Precompute K^T once for the backward pass
    Kt_np = K_np.T.copy()   # (N_merge, N_found) float32, system RAM

    # Shape descriptors for pure_callback
    _kw_shape  = jax.ShapeDtypeStruct((N_found,), jnp.float64)
    _ktg_shape = jax.ShapeDtypeStruct((N_merge,), jnp.float64)

    # Forward: K @ w
    def _kw_numpy(w):
        return (K_np @ w.astype(np.float32)).astype(np.float64)

    # Backward: K^T @ g  (gradient of (K@w) wrt w)
    def _ktg_numpy(g):
        return (Kt_np @ g.astype(np.float32)).astype(np.float64)

    @jax.custom_vjp
    def kw_matmul(w: jnp.ndarray) -> jnp.ndarray:
        """K @ w via numpy, hidden from XLA allocator."""
        return jax.pure_callback(_kw_numpy, _kw_shape, w)

    def _kw_fwd(w):
        return kw_matmul(w), w   # residuals = w (needed in bwd)

    def _kw_bwd(w_res, g):
        # gradient of sum(f(K@w)) wrt w = K^T @ (df/d(Kw))
        Ktg = jax.pure_callback(_ktg_numpy, _ktg_shape, g)
        return (Ktg,)

    kw_matmul.defvjp(_kw_fwd, _kw_bwd)
    print(f"  K@w: pure_callback (K stays in system RAM, not GPU/XLA memory)")
    # Precompute injection parameter columns
    def _col(name):
        idx = param_idx_inj.get(name)
        return theta_inj[:, idx] if idx is not None else None

    inj_a1 = _col('alpha_1')
    inj_a2 = _col('alpha_2')
    inj_f1 = _col('flim_1')
    inj_f2 = _col('flim_2')
    inj_v1 = _col('vk1')
    inj_v2 = _col('vk2')

    def _log_pi0_inj(name):
        idx = param_idx_inj.get(name)
        if idx is None:
            return 0.0
        return float(-jnp.log(hi_inj[idx] - lo_inj[idx]))

    lpi0_a1 = _log_pi0_inj('alpha_1')
    lpi0_a2 = _log_pi0_inj('alpha_2')
    lpi0_f1 = _log_pi0_inj('flim_1')
    lpi0_f2 = _log_pi0_inj('flim_2')

    def _support_inj(name):
        idx = param_idx_inj.get(name)
        if idx is None:
            return 0.0, 1.0
        return float(lo_inj[idx]), float(hi_inj[idx])

    a1_lo, a1_hi = _support_inj('alpha_1')
    a2_lo, a2_hi = _support_inj('alpha_2')
    v1_lo, v1_hi = _support_inj('vk1')
    v2_lo, v2_hi = _support_inj('vk2')
    if log_q_proposal is None:
        warnings.warn(
            "Injection file lacks log_q_proposal; falling back to legacy proposal "
            "reconstruction from bounds and kick_proposal_sigma.",
            RuntimeWarning,
        )
        use_explicit_q = False
        log_q = None
        static_pop = None
    else:
        use_explicit_q = True
        log_q = log_q_proposal
        static_pop = jnp.zeros(theta_inj.shape[0]) if log_pop_static is None else log_pop_static

    # Kick denominator: Maxwellian (injection proposal), not uniform
    def _log_maxw_v1(vk):
        return log_p_vk_jax(vk, kick_sigma, v1_lo, v1_hi)

    def _log_maxw_v2(vk):
        return log_p_vk_jax(vk, kick_sigma, v2_lo, v2_hi)

    @jit
    def log_alpha(lp_vec: jnp.ndarray) -> jnp.ndarray:
        """Evaluate log alpha(Lambda_pop) via Farr (2019) estimator.

        Returns scalar jnp.ndarray.
        """
        mu_la1, sig_la1 = lp_vec[0], lp_vec[1]
        mu_la2, sig_la2 = lp_vec[2], lp_vec[3]
        af1,    bf1     = lp_vec[4], lp_vec[5]
        af2,    bf2     = lp_vec[6], lp_vec[7]
        sv1,    sv2     = lp_vec[8], lp_vec[9]

        # Weight ratios for COSMIC mergers. New injection files carry the full
        # proposal density q(theta), so use log p_pop(theta|Lambda)-log q.
        # Legacy files reconstruct only the non-cancelling factors.
        if use_explicit_q:
            log_wr = static_pop - log_q
            if inj_a1 is not None:
                log_wr = log_wr + log_p_alpha_jax(inj_a1, mu_la1, sig_la1, a1_lo, a1_hi)
            if inj_a2 is not None:
                log_wr = log_wr + log_p_alpha_jax(inj_a2, mu_la2, sig_la2, a2_lo, a2_hi)
            if inj_f1 is not None:
                log_wr = log_wr + log_p_flim_jax(inj_f1, af1, bf1)
            if inj_f2 is not None:
                log_wr = log_wr + log_p_flim_jax(inj_f2, af2, bf2)
            if inj_v1 is not None:
                log_wr = log_wr + log_p_vk_jax(inj_v1, sv1, v1_lo, v1_hi)
            if inj_v2 is not None:
                log_wr = log_wr + log_p_vk_jax(inj_v2, sv2, v2_lo, v2_hi)
        else:
            log_wr = jnp.zeros(theta_inj.shape[0])
            if inj_a1 is not None:
                log_wr = log_wr + log_p_alpha_jax(inj_a1, mu_la1, sig_la1, a1_lo, a1_hi) - lpi0_a1
            if inj_a2 is not None:
                log_wr = log_wr + log_p_alpha_jax(inj_a2, mu_la2, sig_la2, a2_lo, a2_hi) - lpi0_a2
            if inj_f1 is not None:
                log_wr = log_wr + log_p_flim_jax(inj_f1, af1, bf1) - lpi0_f1
            if inj_f2 is not None:
                log_wr = log_wr + log_p_flim_jax(inj_f2, af2, bf2) - lpi0_f2
            if inj_v1 is not None:
                log_wr = log_wr + log_p_vk_jax(inj_v1, sv1, v1_lo, v1_hi) - _log_maxw_v1(inj_v1)
            if inj_v2 is not None:
                log_wr = log_wr + log_p_vk_jax(inj_v2, sv2, v2_lo, v2_hi) - _log_maxw_v2(inj_v2)

        # Numerically stable K @ w via pure_callback (numpy, system RAM).
        # K never enters XLA memory — pure_callback hides it from the allocator.
        log_wr_max = jnp.max(log_wr)
        w_stable   = jnp.exp(log_wr - log_wr_max)   # (N_merge,) float64

        Kw = kw_matmul(w_stable)                     # (N_found,) float64, numpy call

        log_Kw    = jnp.log(jnp.clip(Kw, 1e-300, None)) + log_wr_max
        log_vKw   = log_v + log_Kw
        log_alpha_ = jsp.special.logsumexp(log_vKw) + log_norm

        return log_alpha_

    return log_alpha


# ---------------------------------------------------------------------------
# Hierarchical likelihood (pure JAX)
# ---------------------------------------------------------------------------

def make_hierarchical_log_likelihood(
    log_wr_fns:   list[callable],   # per-event log_weight_ratio functions
    log_z_arr:    jnp.ndarray,      # (N_events,) per-event log evidences
    n_samples:    int,              # N posterior draws per event
    log_alpha_fn: callable | None,  # selection integral function, or None
    n_events:     int,
) -> callable:
    """Return a JIT-compiled hierarchical log-likelihood function.

    Parameters
    ----------
    log_wr_fns : list of callables
        One per event; each takes lp_vec (10,) → log_wr (N_samp,).
    log_z_arr : (N_events,) array
        Per-event log evidences from Nautilus.
    n_samples : int
        Number of importance samples per event.
    log_alpha_fn : callable or None
        Selection integral function; None skips correction.
    n_events : int
        Number of events.
    """
    log_N = jnp.log(float(n_samples))

    @jit
    def hierarchical_log_likelihood(lp_vec: jnp.ndarray) -> jnp.ndarray:
        """Total log hierarchical likelihood.

        lp_vec: (10,) population hyperparameter vector.
        Returns scalar.
        """
        log_l = jnp.float64(0.0)

        # Per-event contributions
        for i, (wr_fn, lz) in enumerate(zip(log_wr_fns, log_z_arr)):
            log_wr = wr_fn(lp_vec)                           # (N_samp,)
            lse    = jsp.special.logsumexp(log_wr)           # log Σ exp(log_wr)
            log_li = lz + lse - log_N                        # log Z_i + log mean
            log_l  = log_l + log_li

        # Selection correction
        if log_alpha_fn is not None:
            log_a = log_alpha_fn(lp_vec)
            log_l = log_l - n_events * log_a

        return log_l

    return hierarchical_log_likelihood


# ---------------------------------------------------------------------------
# NumPyro model
# ---------------------------------------------------------------------------

def make_numpyro_model(log_likelihood_fn: callable) -> callable:
    """Wrap the hierarchical log-likelihood as a NumPyro model.

    The population hyperparameters use explicit, weakly informative priors
    chosen for each parameter type rather than from rectangular hyperparameter bounds.
    Keep POP_PARAM_NAMES order unchanged when packing the sampled values into
    lp_vec, because downstream likelihood, output, and plotting code assume
    that vector order.
    """
    def model():
        params = {}
        params["mu_logalpha1"] = numpyro.sample(
            "mu_logalpha1",
            dist.Normal(0.0, 1.5),
        )
        params["sig_logalpha1"] = numpyro.sample(
            "sig_logalpha1",
            dist.Exponential(1.0),
        )
        params["mu_logalpha2"] = numpyro.sample(
            "mu_logalpha2",
            dist.Normal(0.0, 1.5),
        )
        params["sig_logalpha2"] = numpyro.sample(
            "sig_logalpha2",
            dist.Exponential(1.0),
        )
        params["a_f1"] = numpyro.sample(
            "a_f1",
            dist.Gamma(2.0, 1.0),
        )
        params["b_f1"] = numpyro.sample(
            "b_f1",
            dist.Gamma(2.0, 1.0),
        )
        params["a_f2"] = numpyro.sample(
            "a_f2",
            dist.Gamma(2.0, 1.0),
        )
        params["b_f2"] = numpyro.sample(
            "b_f2",
            dist.Gamma(2.0, 1.0),
        )
        params["sigma_v1"] = numpyro.sample(
            "sigma_v1",
            dist.HalfNormal(150.0),
        )
        params["sigma_v2"] = numpyro.sample(
            "sigma_v2",
            dist.HalfNormal(150.0),
        )

        # Pack into a single vector for the JIT-compiled likelihood
        lp_vec = jnp.array([params[name] for name in POP_PARAM_NAMES])

        log_l = log_likelihood_fn(lp_vec)
        numpyro.factor("log_likelihood", log_l)

    return model



def _npz_scalar(npz, key, default=None):
    if key not in npz:
        return default
    value = npz[key]
    if hasattr(value, "shape") and value.shape == ():
        value = value.item()
    elif hasattr(value, "ravel"):
        value = value.ravel()[0]
    if isinstance(value, bytes):
        return value.decode()
    return value

def _bool_meta(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool) or type(value).__name__ == "bool_":
        return bool(value)
    return str(value).strip().lower() in {"true", "t", "1", "yes", "y", "on"}

def metadata_model_signature(meta: dict) -> dict:
    mode = str(meta.get("likelihood_mode", "3D" if meta.get("uses_z_form") else "2D")).upper()
    return {
        "likelihood_mode": mode,
        "uses_z_form": _bool_meta(meta.get("uses_z_form"), mode == "3D"),
        "uses_sfr_prior": _bool_meta(meta.get("uses_sfr_prior"), mode == "3D"),
        "uses_logZ_given_z_prior": _bool_meta(meta.get("uses_logZ_given_z_prior"), mode == "3D"),
    }

def validate_selection_model_consistency(event_meta: list[dict], injection_meta: dict | None, allow_inconsistent: bool = False) -> tuple[bool, str]:
    if not event_meta or injection_meta is None:
        return True, "No selection injection metadata to compare."
    event_sigs = [metadata_model_signature(m) for m in event_meta]
    first = event_sigs[0]
    if any(sig != first for sig in event_sigs[1:]):
        msg = f"Single-event posterior metadata are mixed: {event_sigs}"
        if allow_inconsistent:
            return False, msg
        raise ValueError(msg + "; pass --allow_inconsistent_selection_model True to override.")
    inj_sig = metadata_model_signature(injection_meta)
    if first != inj_sig:
        msg = ("Event posterior and selection injection generative models are inconsistent: "
               f"event={first}, injections={inj_sig}")
        if allow_inconsistent:
            warnings.warn(msg, RuntimeWarning)
            return False, msg
        raise ValueError(msg + "; pass --allow_inconsistent_selection_model True to override.")
    return True, "Event posterior and selection injection metadata match."

# ---------------------------------------------------------------------------
# Data loading (reuses logic from hierarchical_backpop.py)
# ---------------------------------------------------------------------------

def load_event_data(
    results_dir: str,
    n_samples:   int = 10_000,
) -> tuple[jnp.ndarray, dict[str, int], jnp.ndarray, jnp.ndarray, float, str, dict]:
    """Load a single-event BackPop posterior and return JAX arrays.

    Returns
    -------
    samples : jnp.ndarray  (N_samp, N_params) float64
    param_idx : dict
    lo, hi : jnp.ndarray   prior bounds
    log_z : float
    event_name : str
    """
    points = np.load(os.path.join(results_dir, "points.npy"))
    log_w  = np.load(os.path.join(results_dir, "log_w.npy"))
    log_z  = float(np.load(os.path.join(results_dir, "log_z.npy")).ravel()[0])
    meta   = np.load(os.path.join(results_dir, "metadata.npz"), allow_pickle=True)

    params = list(meta['params_in'])
    lo     = meta['lower_bound'].astype(np.float64)
    hi     = meta['upper_bound'].astype(np.float64)
    name   = str(meta['event_name'])

    # Normalise and resample
    weights = np.exp(log_w - log_z)
    weights /= weights.sum()
    idx     = np.random.choice(len(points), size=n_samples, replace=True, p=weights)
    samples = points[idx].astype(np.float64)

    n_eff = int(1.0 / np.sum(weights**2))
    event_meta = {
        "likelihood_mode": _npz_scalar(meta, "likelihood_mode", "2D"),
        "uses_z_form": _npz_scalar(meta, "uses_z_form", _npz_scalar(meta, "use_redshift_likelihood", False)),
        "uses_sfr_prior": _npz_scalar(meta, "uses_sfr_prior", _npz_scalar(meta, "use_redshift_likelihood", False)),
        "uses_logZ_given_z_prior": _npz_scalar(meta, "uses_logZ_given_z_prior", _npz_scalar(meta, "use_redshift_likelihood", False)),
    }
    print(f"  {name}: N_eff={n_eff}  log Z={log_z:.2f}  params={len(params)}-D  mode={metadata_model_signature(event_meta)}")

    return (
        jnp.array(samples),
        {p: i for i, p in enumerate(params)},
        jnp.array(lo),
        jnp.array(hi),
        log_z,
        name,
        event_meta,
    )


def _stringify_hdf5_attr(value) -> str:
    """Return a compact printable representation of an HDF5 attribute."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _stringify_hdf5_attr(value.item())
        return np.array2string(value, threshold=8, edgeitems=3)
    return str(value)


def _collect_hdf5_attrs(obj) -> dict[str, str]:
    return {str(k): _stringify_hdf5_attr(v) for k, v in obj.attrs.items()}


def print_lvk_hdf5_audit_attrs(f, grp) -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, str]]]:
    """Print root and injection-group metadata relevant to sampling_pdf audits."""
    root_attrs = _collect_hdf5_attrs(f)
    grp_attrs = _collect_hdf5_attrs(grp)
    print("  LVK HDF5 root attributes:")
    if root_attrs:
        for key in sorted(root_attrs):
            print(f"    {key}: {root_attrs[key]}")
    else:
        print("    (none)")
    print("  LVK HDF5 injections group attributes:")
    if grp_attrs:
        for key in sorted(grp_attrs):
            print(f"    {key}: {grp_attrs[key]}")
    else:
        print("    (none)")
    dataset_attrs = {}
    for name in ["mass1_source", "mass2_source", "redshift", "sampling_pdf"]:
        if name in grp:
            attrs = _collect_hdf5_attrs(grp[name])
            dataset_attrs[name] = attrs
            print(f"  LVK HDF5 injections/{name} dataset attributes:")
            if attrs:
                for key in sorted(attrs):
                    print(f"    {key}: {attrs[key]}")
            else:
                print("    (none)")
    return root_attrs, grp_attrs, dataset_attrs


def validate_lvk_sampling_pdf_measure(f, grp) -> dict[str, object]:
    """Audit whether LVK sampling_pdf metadata supports the assumed measure.

    The Farr kernel below is evaluated in (log mass1_source, log mass2_source, z).
    Therefore this script assumes the LVK `sampling_pdf` density is per
    (mass1_source, mass2_source, redshift), so the conversion to log source
    masses contributes the Jacobian |d(m1,m2)/d(log m1,log m2)| = m1*m2.
    If the file does not explicitly document that convention, we keep running
    in permissive mode but mark the measure as unverified.
    """
    root_attrs, grp_attrs, dataset_attrs = print_lvk_hdf5_audit_attrs(f, grp)
    all_attrs = {f"root.{k}": v for k, v in root_attrs.items()}
    all_attrs.update({f"injections.{k}": v for k, v in grp_attrs.items()})
    for dataset_name, attrs in dataset_attrs.items():
        all_attrs.update({f"injections/{dataset_name}.{k}": v for k, v in attrs.items()})
    haystack = "\n".join(f"{k}={v}" for k, v in all_attrs.items()).lower()

    has_sampling_pdf_metadata = "sampling_pdf" in haystack or "sampling pdf" in haystack
    mentions_source_masses = ("mass1_source" in haystack or "m1_source" in haystack) and (
        "mass2_source" in haystack or "m2_source" in haystack
    )
    mentions_redshift = "redshift" in haystack or " z" in haystack
    mentions_log_mass_measure = "log_mass" in haystack or "log mass" in haystack or "logm" in haystack

    assumed_measure = "d(mass1_source) d(mass2_source) d(redshift)"
    verified = bool(has_sampling_pdf_metadata and mentions_source_masses and mentions_redshift and not mentions_log_mass_measure)
    status = "verified" if verified else "ambiguous"
    message = (
        f"LVK sampling_pdf measure {status}; code assumes {assumed_measure} and applies "
        "an m1*m2 Jacobian to evaluate kernels in log source masses."
    )
    if not verified:
        message += " HDF5 metadata is ambiguous; no unsupported claim is made."

    return dict(
        verified=verified,
        status=status,
        assumed_measure=assumed_measure,
        message=message,
        root_attrs=root_attrs,
        injection_attrs=grp_attrs,
        dataset_attrs=dataset_attrs,
    )


def load_lvk_injections(
    lvk_path: str,
    n_inj_total: int | None = None,
    strict_sampling_pdf: bool = False,
) -> dict:
    """Load LVK found injection set and audit the `sampling_pdf` coordinate measure."""
    measure_metadata = dict(
        verified=False,
        status="not_hdf5",
        assumed_measure="d(mass1_source) d(mass2_source) d(redshift)",
        message="No HDF5 metadata available for LVK sampling_pdf validation.",
        root_attrs={},
        injection_attrs={},
        dataset_attrs={},
    )
    if lvk_path.endswith('.h5') or lvk_path.endswith('.hdf5'):
        import h5py
        with h5py.File(lvk_path, 'r') as f:
            grp    = f['injections']
            measure_metadata = validate_lvk_sampling_pdf_measure(f, grp)
            print(f"  {measure_metadata['message']}")
            if not measure_metadata['verified']:
                warning_msg = "PROMINENT WARNING: " + str(measure_metadata['message'])
                if strict_sampling_pdf:
                    raise ValueError(warning_msg + " Re-run without --strict_lvk_sampling_pdf True to permit ambiguous metadata.")
                warnings.warn(warning_msg, RuntimeWarning)
            m1     = grp['mass1_source'][:].astype(np.float64)
            m2     = grp['mass2_source'][:].astype(np.float64)
            z      = grp['redshift'][:].astype(np.float64)
            q_lvk  = (grp['sampling_pdf'][:].astype(np.float64)
                      if 'sampling_pdf' in grp else np.ones(len(m1)))
            if n_inj_total is not None:
                N_inj = int(n_inj_total)
            else:
                attrs = {**dict(f.attrs), **dict(grp.attrs)}
                N_inj = None
                for key in ['total_generated', 'n_injections', 'total_injections']:
                    if key in attrs:
                        N_inj = int(attrs[key])
                        print(f"  N_inj_total from attr '{key}': {N_inj:,}")
                        break
                if N_inj is None:
                    warnings.warn("N_inj_total not found — falling back to N_found. "
                                  "Rate normalisation will be wrong.", RuntimeWarning)
                    N_inj = len(m1)
    else:
        data  = np.load(lvk_path, allow_pickle=True)
        m1    = data['m1_src'].astype(np.float64)
        m2    = data['m2_src'].astype(np.float64)
        z     = data['z' if 'z' in data else 'redshift'].astype(np.float64)
        q_lvk = (data['sampling_pdf'].astype(np.float64)
                 if 'sampling_pdf' in data else np.ones(len(m1)))
        if strict_sampling_pdf:
            raise ValueError("--strict_lvk_sampling_pdf True requires verifiable HDF5 metadata; NPZ LVK injections are unverified.")
        warnings.warn("PROMINENT WARNING: NPZ LVK injections do not provide HDF5 sampling_pdf measure metadata; measure is unverified.", RuntimeWarning)
        N_inj = (int(n_inj_total) if n_inj_total is not None
                 else int(data['N_inj'].ravel()[0]))

    # m1 >= m2 convention
    swap   = m1 < m2
    m1[swap], m2[swap] = m2[swap].copy(), m1[swap].copy()

    return dict(m1=m1, m2=m2, z=z, q_lvk=q_lvk, N_inj=N_inj, sampling_pdf_metadata=measure_metadata)


def build_kernel_matrix_chunked(
    lvk: dict,
    cosmic: dict,
    bandwidth: dict | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, float]:
    """Build K matrix, log_v vector, and log_norm for the Farr estimator.

    Chunked construction to avoid materialising the full float64 intermediate
    arrays at once (same approach as LVKInjectionCampaign.__init__).

    Returns
    -------
    K : jnp.ndarray (N_found, N_merge) float32
        Kernel matrix. Stored as float32 to halve memory.
    log_v : jnp.ndarray (N_found,) float64
        log(1 / (q_LVK * m1 * m2)) per found injection.
    log_norm : float
        -log(N_lvk_inj) - log(N_cosmic_inj)
    """
    m1_f = np.log(lvk['m1'])
    m2_f = np.log(lvk['m2'])
    z_f  = lvk['z']

    m1_m = np.log(cosmic['m1_src'])
    m2_m = np.log(cosmic['m2_src'])
    z_m  = cosmic['z_merger']

    N_found = len(m1_f)
    N_merge = len(m1_m)

    # Bandwidth via Scott's rule if not supplied
    if bandwidth is None:
        scale  = N_merge ** (-1.0 / 7.0)
        h_lm1  = max(scale * np.std(m1_m), 0.02)
        h_lm2  = max(scale * np.std(m2_m), 0.02)
        h_z    = max(scale * np.std(z_m),  0.05)
    else:
        h_lm1, h_lm2, h_z = bandwidth['log_m1'], bandwidth['log_m2'], bandwidth['z']

    print(f"  Kernel bandwidth: log_m1={h_lm1:.3f}  log_m2={h_lm2:.3f}  z={h_z:.3f}")

    LOG_NORM_K = (
        - np.log(h_lm1) - np.log(h_lm2) - np.log(h_z)
        - 1.5 * np.log(2.0 * np.pi)
    )

    K_mb = N_found * N_merge * 4 / 1e6
    print(f"  Building K matrix ({N_found} x {N_merge}), {K_mb:.0f} MB float32 ...")

    K = np.empty((N_found, N_merge), dtype=np.float32)
    CHUNK = 5_000
    for start in range(0, N_found, CHUNK):
        end   = min(start + CHUNK, N_found)
        dlm1  = m1_f[start:end, None] - m1_m[None, :]
        dlm2  = m2_f[start:end, None] - m2_m[None, :]
        dz    = z_f[start:end, None]  - z_m[None, :]
        log_K = (
            - 0.5 * (dlm1 / h_lm1)**2
            - 0.5 * (dlm2 / h_lm2)**2
            - 0.5 * (dz   / h_z  )**2
            + LOG_NORM_K
        )
        K[start:end] = np.exp(log_K).astype(np.float32)
        del dlm1, dlm2, dz, log_K

    print("  K matrix complete.")

    # The Gaussian KDE is defined in x=(log m1_source, log m2_source, z),
    # while verified LVK sampling_pdf metadata is expected to describe a density
    # in y=(m1_source, m2_source, z).  Importance weights in x require
    # q_x(x) = q_y(y) * |dy/dx| = q_LVK * m1 * m2, hence the m1*m2
    # Jacobian in the denominator below.  If the HDF5 metadata was ambiguous,
    # load_lvk_injections() warns (or raises in strict mode) before reaching here.
    with np.errstate(divide='ignore', invalid='ignore'):
        log_v_np = -(np.log(lvk['q_lvk']) + np.log(lvk['m1']) + np.log(lvk['m2']))
    log_v_np[~np.isfinite(log_v_np)] = -np.inf

    log_norm = -(np.log(lvk['N_inj']) + np.log(cosmic['N_inj']))

    # Return K as numpy (CPU) — it's too large for GPU VRAM.
    # log_v is moved to JAX device for use in logsumexp.
    return K, jnp.array(log_v_np), float(log_norm)



def diagnose_lvk_selection_contributions(K: np.ndarray, log_v: jnp.ndarray, label: str = "uniform COSMIC test population") -> None:
    """Print effective found-injection contribution diagnostics for a simple test population.

    The test population assigns equal probability to every COSMIC merger in the
    KDE support.  It is not an astrophysical claim; it is a smoke test exposing
    whether a small number of found injections dominate the Farr estimator.
    """
    if K.size == 0:
        print("  LVK selection diagnostic skipped: empty K matrix.")
        return
    mean_kernel = np.asarray(K, dtype=np.float64).mean(axis=1)
    log_v_np = np.asarray(log_v, dtype=np.float64)
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        contrib = np.exp(log_v_np) * mean_kernel
    finite = contrib[np.isfinite(contrib) & (contrib > 0.0)]
    print(f"  LVK selection diagnostic ({label}):")
    if finite.size == 0:
        print("    no positive finite found-injection contributions")
        return
    total = finite.sum()
    normed = finite / total
    n_eff = 1.0 / np.sum(normed**2)
    qs = np.percentile(finite, [0, 5, 50, 95, 99, 100])
    top = np.sort(normed)[::-1]
    print(f"    positive finite contributions: {finite.size:,}/{contrib.size:,}")
    print(f"    effective contributors: {n_eff:.1f}")
    print("    contribution quantiles [0,5,50,95,99,100]%: " + ", ".join(f"{q:.3e}" for q in qs))
    print(f"    top 1 / 10 / 100 cumulative fractions: {top[:1].sum():.3f} / {top[:10].sum():.3f} / {top[:100].sum():.3f}")


def _importance_weight_summary_from_log_weights(log_w: np.ndarray) -> dict:
    """Return stable ESS and dominance diagnostics for non-negative log weights."""
    log_w = np.asarray(log_w, dtype=np.float64)
    finite = np.isfinite(log_w)
    n_total = int(log_w.size)
    n_finite = int(finite.sum())
    if n_finite == 0:
        return dict(n_total=n_total, n_finite=0, ess=0.0, ess_fraction=0.0,
                    max_normalized_weight=np.nan, warning="SEVERE")

    lw = log_w[finite]
    lw_max = np.max(lw)
    w = np.exp(lw - lw_max)
    w_sum = w.sum()
    w2_sum = np.square(w).sum()
    if (not np.isfinite(w_sum)) or w_sum <= 0.0 or (not np.isfinite(w2_sum)) or w2_sum <= 0.0:
        return dict(n_total=n_total, n_finite=n_finite, ess=0.0, ess_fraction=0.0,
                    max_normalized_weight=np.nan, warning="SEVERE")

    norm_w = w / w_sum
    ess = float(w_sum * w_sum / w2_sum)
    ess_fraction = ess / n_total if n_total > 0 else 0.0
    if ess_fraction < 0.01:
        warning = "SEVERE"
    elif ess_fraction < 0.05:
        warning = "WARN"
    else:
        warning = "OK"
    return dict(n_total=n_total, n_finite=n_finite, ess=ess,
                ess_fraction=float(ess_fraction),
                max_normalized_weight=float(np.max(norm_w)),
                warning=warning)


def compute_event_reweighting_diagnostics(
    lp_vec: np.ndarray,
    label: str,
    log_wr_fns: list[callable],
    event_names: list[str],
) -> list[dict]:
    """Compute per-event posterior importance-weight ESS diagnostics."""
    rows = []
    for event_name, wr_fn in zip(event_names, log_wr_fns):
        log_w = np.asarray(wr_fn(jnp.array(lp_vec, dtype=jnp.float64)), dtype=np.float64)
        summary = _importance_weight_summary_from_log_weights(log_w)
        rows.append(dict(
            hyperpoint=label,
            event_name=event_name,
            raw_posterior_sample_count=summary["n_total"],
            finite_weight_count=summary["n_finite"],
            importance_ess=summary["ess"],
            ess_fraction=summary["ess_fraction"],
            max_normalized_weight=summary["max_normalized_weight"],
            warning=summary["warning"],
        ))
    return rows


def print_event_reweighting_diagnostics(rows: list[dict], label: str) -> None:
    """Print a compact table of per-event reweighting quality."""
    print(f"\n Posterior reweighting diagnostics ({label}):")
    print(f"  {'event':<28s} {'N_raw':>8s} {'ESS':>10s} {'ESS/N':>8s} {'max w':>10s} {'flag':>8s}")
    print("  " + "-" * 78)
    for row in rows:
        print(f"  {row['event_name']:<28.28s} "
              f"{row['raw_posterior_sample_count']:8d} "
              f"{row['importance_ess']:10.1f} "
              f"{row['ess_fraction']:8.4f} "
              f"{row['max_normalized_weight']:10.3e} "
              f"{row['warning']:>8s}")


def write_event_reweighting_diagnostics_csv(path: str, rows: list[dict]) -> None:
    import csv
    fields = [
        "hyperpoint", "event_name", "raw_posterior_sample_count",
        "finite_weight_count", "importance_ess", "ess_fraction",
        "max_normalized_weight", "warning",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def compute_selection_weight_diagnostics(
    lp_vec: np.ndarray,
    label: str,
    K_np: np.ndarray,
    log_v: jnp.ndarray,
    cosmic: dict,
) -> dict:
    """Compute found-injection contribution degeneracy diagnostics for alpha."""
    log_wr = compute_log_wr_injections_numpy(
        lp_vec,
        cosmic["theta"],
        cosmic["params"],
        cosmic["lo"],
        cosmic["hi"],
        cosmic["kick_sigma"],
        cosmic.get("log_q_proposal"),
        cosmic.get("log_pop_static"),
    )
    finite_wr = np.isfinite(log_wr)
    if not finite_wr.any() or K_np.size == 0:
        return dict(hyperpoint=label, found_injection_count=int(K_np.shape[0]),
                    cosmic_merger_count=int(K_np.shape[1]) if K_np.ndim == 2 else 0,
                    positive_contribution_count=0, injection_ess=0.0,
                    ess_fraction=0.0, max_contribution=np.nan,
                    top_1pct_alpha_fraction=np.nan,
                    top_0p1pct_alpha_fraction=np.nan,
                    warning="SEVERE")

    lw_max = np.max(log_wr[finite_wr])
    w_stable = np.zeros_like(log_wr, dtype=np.float64)
    w_stable[finite_wr] = np.exp(log_wr[finite_wr] - lw_max)
    Kw = K_np @ w_stable.astype(np.float32)
    log_contrib = np.asarray(log_v, dtype=np.float64) + np.log(np.clip(Kw, 1e-300, None)) + lw_max
    summary = _importance_weight_summary_from_log_weights(log_contrib)
    finite_contrib = log_contrib[np.isfinite(log_contrib)]
    if finite_contrib.size:
        contrib = np.exp(finite_contrib - np.max(finite_contrib))
        contrib = contrib[contrib > 0.0]
    else:
        contrib = np.array([], dtype=np.float64)
    if contrib.size:
        norm = np.sort(contrib / contrib.sum())[::-1]
        n = norm.size
        top_1 = norm[:max(1, int(np.ceil(0.01 * n)))].sum()
        top_0p1 = norm[:max(1, int(np.ceil(0.001 * n)))].sum()
    else:
        top_1 = top_0p1 = np.nan
    return dict(
        hyperpoint=label,
        found_injection_count=summary["n_total"],
        cosmic_merger_count=int(K_np.shape[1]),
        positive_contribution_count=int(contrib.size),
        injection_ess=summary["ess"],
        ess_fraction=summary["ess_fraction"],
        max_contribution=summary["max_normalized_weight"],
        top_1pct_alpha_fraction=float(top_1),
        top_0p1pct_alpha_fraction=float(top_0p1),
        warning=summary["warning"],
    )


def print_selection_weight_diagnostics(rows: list[dict]) -> None:
    """Print selection-integral contribution diagnostics."""
    if not rows:
        return
    print("\n Selection injection diagnostics:")
    print(f"  {'hyperpoint':<18s} {'N_found':>8s} {'ESS':>10s} {'ESS/N':>8s} {'max':>10s} {'top1%':>8s} {'top0.1%':>8s} {'flag':>8s}")
    print("  " + "-" * 94)
    for row in rows:
        print(f"  {row['hyperpoint']:<18.18s} "
              f"{row['found_injection_count']:8d} "
              f"{row['injection_ess']:10.1f} "
              f"{row['ess_fraction']:8.4f} "
              f"{row['max_contribution']:10.3e} "
              f"{row['top_1pct_alpha_fraction']:8.3f} "
              f"{row['top_0p1pct_alpha_fraction']:8.3f} "
              f"{row['warning']:>8s}")


def write_selection_weight_diagnostics_csv(path: str, rows: list[dict]) -> None:
    import csv
    fields = [
        "hyperpoint", "found_injection_count", "cosmic_merger_count",
        "positive_contribution_count", "injection_ess", "ess_fraction",
        "max_contribution", "top_1pct_alpha_fraction",
        "top_0p1pct_alpha_fraction", "warning",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def discover_events(results_root: str, config_name: str) -> list[str]:
    pattern = os.path.join(results_root, '*', config_name, 'log_z.npy')
    hits    = sorted(glob.glob(pattern))
    if not hits:
        raise FileNotFoundError(
            f"No completed runs found matching: {pattern}"
        )
    return [os.path.dirname(h) for h in hits]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = ArgumentParser(
        description="JAX/NumPyro hierarchical BackPop inference.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results_root",  required=True)
    p.add_argument("--config_name",   required=True)
    p.add_argument("--events",        nargs='+', default=None)
    p.add_argument("--output_dir",    default=None)
    p.add_argument("--n_samples",     type=int,   default=10_000,
                   help="Per-event posterior draws for importance weighting.")
    p.add_argument("--num_warmup",    type=int,   default=500,
                   help="NUTS warmup steps (adaptation).")
    p.add_argument("--num_samples",   type=int,   default=1000,
                   help="NUTS posterior samples per chain.")
    p.add_argument("--num_chains",    type=int,   default=4,
                   help="Number of independent NUTS chains.")
    p.add_argument("--target_accept", type=float, default=0.8,
                   help="NUTS target acceptance probability.")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--diagnostic_posterior_draws", type=int, default=5,
                   help="Number of posterior draws, in addition to the posterior median, "
                        "used for post-NUTS reweighting diagnostics.")

    sel = p.add_argument_group("Selection effects")
    sel.add_argument("--injections_path",   default=None)
    sel.add_argument("--lvk_found_path",    default=None)
    sel.add_argument("--lvk_n_inj_total",   type=int,   default=None)
    sel.add_argument("--lvk_n_found_max",   type=int,   default=None,
                     help="Optional max LVK found injections to subsample for K matrix. "
                          "By default all found injections are used. If set below "
                          "N_found, the Farr estimator is explicitly rescaled by "
                          "N_found_total/N_found_used so alpha remains absolutely "
                          "normalized.")
    sel.add_argument("--lvk_bandwidth_log_m1", type=float, default=None)
    sel.add_argument("--lvk_bandwidth_log_m2", type=float, default=None)
    sel.add_argument("--lvk_bandwidth_z",      type=float, default=None)
    sel.add_argument("--strict_lvk_sampling_pdf", type=str2bool, default=False,
                     help="Fail instead of warning when LVK HDF5 metadata does not verify that sampling_pdf is in d(mass1_source) d(mass2_source) d(redshift).")
    sel.add_argument("--allow_inconsistent_selection_model", type=str2bool, default=False,
                     help="Explicitly allow event posterior and injection metadata to use different base measures. Intended only for legacy diagnostics.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def determine_selection_mode(injections_path: str | None, lvk_found_path: str | None) -> str:
    """Validate selection-effect inputs and return the implemented mode.

    COSMIC injections alone used to select the unimplemented ``interpolator``
    mode.  That silently skipped selection correction because no interpolator
    ``log_alpha_fn`` exists in this JAX driver.  Hard-error instead so users do
    not accidentally run a mislabeled, uncorrected analysis.
    """
    has_cosmic = bool(injections_path)
    has_lvk = bool(lvk_found_path)

    if has_lvk and not has_cosmic:
        raise ValueError("--lvk_found_path requires --injections_path")
    if has_cosmic and not has_lvk:
        raise NotImplementedError(
            "--injections_path without --lvk_found_path would select the "
            "unimplemented interpolator selection mode. Provide "
            "--lvk_found_path to enable the LVK/Farr selection correction, or "
            "omit --injections_path to run with selection effects disabled."
        )

    return "lvk_farr" if (has_lvk and has_cosmic) else "none"


def lvk_found_subsample_log_scaling(N_found_total: int, N_found_used: int) -> float:
    """Return the log correction for LVK found-injection subsampling.

    The Farr estimator sums over the full set of found LVK injections.  If a
    uniform subset is used to build the K matrix, the unbiased absolute
    selection estimate is recovered by multiplying the subset sum by
    ``N_found_total / N_found_used``.  This factor is constant in Lambda, but it
    is required for correct alpha values, diagnostics, and metadata.
    """
    if N_found_total <= 0:
        raise ValueError("N_found_total must be positive")
    if N_found_used <= 0:
        raise ValueError("N_found_used must be positive")
    if N_found_used > N_found_total:
        raise ValueError("N_found_used cannot exceed N_found_total")
    return float(np.log(N_found_total) - np.log(N_found_used))


def main():
    start_time = time.time()
    opts       = parse_args()
    np.random.seed(opts.seed)

    selection_mode = determine_selection_mode(
        opts.injections_path,
        opts.lvk_found_path,
    )

    # Output directory
    _tag = {"none": "no_selection", "lvk_farr": "lvk_farr"}
    out_dir = opts.output_dir or os.path.join(
        opts.results_root, "hierarchical", opts.config_name, "nuts",
        _tag[selection_mode]
    )
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print(" BackPop Hierarchical Inference — JAX/NumPyro NUTS")
    print(f" Config:          {opts.config_name}")
    print(f" Selection mode:  {selection_mode}")
    print(f" JAX backend:     {jax.default_backend()}")
    print(f" JAX x64:         {jax.numpy.ones(1).dtype}")
    print(f" Devices:         {jax.devices()}")
    print(f" Output:          {out_dir}")
    print("=" * 60)

    # ---- Load events ----
    if opts.events:
        event_dirs = [
            os.path.join(opts.results_root, ev, opts.config_name)
            for ev in opts.events
        ]
    else:
        event_dirs = discover_events(opts.results_root, opts.config_name)

    print(f"\n Loading {len(event_dirs)} events...")
    all_samples, all_pidx, all_lo, all_hi, all_logz, event_names, event_model_metadata = [], [], [], [], [], [], []
    for d in event_dirs:
        try:
            samp, pidx, lo, hi, lz, name, ev_meta = load_event_data(d, opts.n_samples)
            all_samples.append(samp)
            all_pidx.append(pidx)
            all_lo.append(lo)
            all_hi.append(hi)
            all_logz.append(lz)
            event_names.append(name)
            event_model_metadata.append(ev_meta)
        except Exception as e:
            print(f"  WARNING: skipping {d} — {e}")

    if len(all_samples) < 2:
        raise ValueError(f"Need at least 2 events, got {len(all_samples)}.")
    print(f"\n Using {len(all_samples)} events: {event_names}")

    log_z_arr  = jnp.array(all_logz)
    n_events   = len(all_samples)

    # ---- Build per-event weight ratio functions ----
    print("\n Compiling per-event weight ratio functions...")
    log_wr_fns = [
        make_log_weight_ratio_fn(s, pidx, lo, hi)
        for s, pidx, lo, hi in zip(all_samples, all_pidx, all_lo, all_hi)
    ]

    # ---- Selection effects ----
    log_alpha_fn = None
    cosmic_raw   = None   # made available for PPD plots below
    cosmic       = None
    K_np         = None
    log_v        = None
    N_found_total = 0
    N_found_used = 0
    lvk_found_subsample_log_scale = 0.0
    selection_model_consistent = True
    selection_model_consistency_message = "Selection disabled; no injection metadata compared."
    injection_model_metadata = {}
    lvk_sampling_pdf_metadata = dict(
        verified=False,
        status="not_used",
        assumed_measure="d(mass1_source) d(mass2_source) d(redshift)",
        message="LVK selection was not used.",
        root_attrs={},
        injection_attrs={},
        dataset_attrs={},
    )
    if selection_mode == "lvk_farr":
        print(f"\n Loading LVK found injections: {opts.lvk_found_path}")
        lvk = load_lvk_injections(opts.lvk_found_path, opts.lvk_n_inj_total, opts.strict_lvk_sampling_pdf)
        lvk_sampling_pdf_metadata = lvk['sampling_pdf_metadata']
        N_found_total = len(lvk['m1'])
        N_found_used = N_found_total
        lvk_found_subsample_log_scale = lvk_found_subsample_log_scaling(N_found_total, N_found_used)
        print(f"  N_found={N_found_total:,}  N_inj={lvk['N_inj']:,}")

        # Optionally subsample found injections for speed.  Because the Farr
        # estimator is a sum over found injections, a uniform subset must be
        # multiplied by N_found_total/N_found_used.  We add that Lambda-constant
        # factor to log_norm below so alpha remains absolutely normalized.
        # 284k found injections: K@w matmul takes ~120s/NUTS iter (CPU↔GPU sync).
        # 5k found injections:   ~2s/iter. Farr variance scales as 1/N_found.
        n_found_max = opts.lvk_n_found_max
        if n_found_max is not None and N_found_total > n_found_max:
            rng_sub  = np.random.default_rng(seed=0)
            idx_sub  = rng_sub.choice(N_found_total, size=n_found_max, replace=False)
            for key in ['m1', 'm2', 'z', 'q_lvk']:
                lvk[key] = lvk[key][idx_sub]
            N_found_used = len(idx_sub)
            lvk_found_subsample_log_scale = lvk_found_subsample_log_scaling(N_found_total, N_found_used)
            print(f"  Subsampled N_found: {N_found_total:,} -> {N_found_used:,}")

        print(f"\n Loading COSMIC merger catalog: {opts.injections_path}")
        cosmic_raw = np.load(opts.injections_path, allow_pickle=True)
        if "metadata" in cosmic_raw:
            injection_model_metadata = dict(cosmic_raw["metadata"].item())
        else:
            injection_model_metadata = {}
        injection_model_metadata.update({
            "likelihood_mode": _npz_scalar(cosmic_raw, "likelihood_mode", injection_model_metadata.get("likelihood_mode", "3D" if "z_form" in cosmic_raw else "2D")),
            "uses_z_form": _npz_scalar(cosmic_raw, "uses_z_form", injection_model_metadata.get("uses_z_form", "z_form" in cosmic_raw)),
            "uses_aux_z_form": _npz_scalar(cosmic_raw, "uses_aux_z_form", injection_model_metadata.get("uses_aux_z_form", False)),
            "uses_sfr_prior": _npz_scalar(cosmic_raw, "uses_sfr_prior", injection_model_metadata.get("uses_sfr_prior", "z_form" in cosmic_raw)),
            "uses_logZ_given_z_prior": _npz_scalar(cosmic_raw, "uses_logZ_given_z_prior", injection_model_metadata.get("uses_logZ_given_z_prior", "z_form" in cosmic_raw and "logZ" in cosmic_raw)),
            "proposal_version": _npz_scalar(cosmic_raw, "proposal_version", injection_model_metadata.get("proposal_version", "unknown")),
        })
        selection_model_consistent, selection_model_consistency_message = validate_selection_model_consistency(
            event_model_metadata, injection_model_metadata, opts.allow_inconsistent_selection_model
        )
        print(f"  Selection model consistency: {selection_model_consistency_message}")
        log_q_arr = None
        log_pop_static_arr = None
        if 'log_q_proposal' in cosmic_raw:
            log_q_arr = cosmic_raw['log_q_proposal'].astype(np.float64)
            if not np.all(np.isfinite(log_q_arr)):
                raise ValueError("log_q_proposal contains non-finite values")
            from gwbackpop.cosmology import log_prior_z_form, log_prior_logZ_given_z_on_support
            theta_tmp = cosmic_raw['theta'].astype(np.float64)
            params_tmp = list(cosmic_raw['params'])
            lo_tmp = cosmic_raw['lower_bound'].astype(np.float64)
            hi_tmp = cosmic_raw['upper_bound'].astype(np.float64)
            pidx_tmp = {p: i for i, p in enumerate(params_tmp)}
            log_pop_static_arr = np.zeros(theta_tmp.shape[0], dtype=np.float64)
            inj_sig_for_weights = metadata_model_signature(injection_model_metadata)
            for name in params_tmp:
                if name == "z_form" and inj_sig_for_weights["uses_sfr_prior"]:
                    continue
                if name == "logZ" and inj_sig_for_weights["uses_logZ_given_z_prior"]:
                    continue
                if name not in ('alpha_1', 'alpha_2', 'flim_1', 'flim_2', 'vk1', 'vk2'):
                    idx = pidx_tmp[name]
                    log_pop_static_arr += -np.log(hi_tmp[idx] - lo_tmp[idx])
            uses_aux_z_form = _bool_meta(injection_model_metadata.get("uses_aux_z_form"), False)
            if (inj_sig_for_weights['uses_z_form'] or inj_sig_for_weights['uses_sfr_prior'] or inj_sig_for_weights['uses_logZ_given_z_prior'] or uses_aux_z_form) and 'z_form' in cosmic_raw and 'logZ' in cosmic_raw:
                zf = cosmic_raw['z_form'].astype(np.float64)
                lz = cosmic_raw['logZ'].astype(np.float64)
                if inj_sig_for_weights['uses_sfr_prior'] or uses_aux_z_form:
                    log_pop_static_arr += np.array([log_prior_z_form(z) for z in zf])
                if inj_sig_for_weights['uses_logZ_given_z_prior']:
                    logz_idx = pidx_tmp.get("logZ")
                    logz_lo = float(lo_tmp[logz_idx]) if logz_idx is not None else float(injection_model_metadata.get("logZ_support", [-np.inf, np.inf])[0])
                    logz_hi = float(hi_tmp[logz_idx]) if logz_idx is not None else float(injection_model_metadata.get("logZ_support", [-np.inf, np.inf])[1])
                    log_pop_static_arr += np.array([log_prior_logZ_given_z_on_support(zmet, z, logz_lo, logz_hi) for zmet, z in zip(lz, zf)])
            else:
                warnings.warn(
                    "Injection file has log_q_proposal but lacks z_form/logZ; "
                    "assuming cosmological proposal factors cancel.",
                    RuntimeWarning,
                )
            print("  Using explicit log_q_proposal from COSMIC merger catalog")

        cosmic = dict(
            theta    = cosmic_raw['theta'].astype(np.float64),
            m1_src   = cosmic_raw['m1_src'].astype(np.float64),
            m2_src   = cosmic_raw['m2_src'].astype(np.float64),
            z_merger = cosmic_raw['z_merger'].astype(np.float64),
            params   = list(cosmic_raw['params']),
            lo       = cosmic_raw['lower_bound'].astype(np.float64),
            hi       = cosmic_raw['upper_bound'].astype(np.float64),
            N_inj    = int(cosmic_raw['N_inj'].ravel()[0]),
            N_merge  = int(cosmic_raw['N_merge'].ravel()[0]),
            kick_sigma = float(
                cosmic_raw['kick_proposal_sigma'].ravel()[0]
                if 'kick_proposal_sigma' in cosmic_raw else 50.0
            ),
            log_q_proposal = log_q_arr,
            log_pop_static = log_pop_static_arr,
        )
        print(f"  N_merge={cosmic['N_merge']:,}  N_COSMIC={cosmic['N_inj']:,}  "
              f"f_merge={cosmic['N_merge']/cosmic['N_inj']:.4f}")

        # Bandwidth
        bandwidth = None
        if opts.lvk_bandwidth_log_m1 is not None:
            bandwidth = {
                'log_m1': opts.lvk_bandwidth_log_m1,
                'log_m2': opts.lvk_bandwidth_log_m2,
                'z':      opts.lvk_bandwidth_z,
            }

        K_np, log_v, log_norm = build_kernel_matrix_chunked(lvk, cosmic, bandwidth)
        # Explicit found-injection subsampling normalization.  This is zero
        # when all found injections are used and positive for uniform
        # subsamples, preserving the absolute selection fraction.
        log_norm += lvk_found_subsample_log_scale
        diagnose_lvk_selection_contributions(K_np, log_v)
        print(f"  LVK found subsampling log scale = {lvk_found_subsample_log_scale:.6g} "
              f"(N_found_total/N_found_used = {N_found_total:,}/{N_found_used:,})")
        print(f"  log_norm = {log_norm:.3f}")
        print(f"  K matmul: numpy pure_callback (K stays in system RAM).")

        theta_inj_jax = jnp.array(cosmic['theta'])
        lo_inj_jax    = jnp.array(cosmic['lo'])
        hi_inj_jax    = jnp.array(cosmic['hi'])
        pidx_inj      = {p: i for i, p in enumerate(cosmic['params'])}

        log_alpha_fn = make_log_alpha_fn(
            K_np, log_v, log_norm,
            theta_inj_jax, lo_inj_jax, hi_inj_jax,
            pidx_inj, cosmic['kick_sigma'],
            (jnp.array(cosmic['log_q_proposal'])
             if cosmic['log_q_proposal'] is not None else None),
            (jnp.array(cosmic['log_pop_static'])
             if cosmic['log_pop_static'] is not None else None),
        )
        print("  Selection effects: ENABLED (Farr estimator, JAX JIT)")

    elif selection_mode == "none":
        print("\n WARNING: Selection effects DISABLED — sigma_v posterior will be biased.")

    # ---- Build hierarchical likelihood and NumPyro model ----
    print("\n Building hierarchical likelihood...")
    log_likelihood_fn = make_hierarchical_log_likelihood(
        log_wr_fns, log_z_arr, opts.n_samples, log_alpha_fn, n_events
    )

    # Trigger JIT compilation with a dummy call before NUTS starts
    print(" Triggering JIT compilation (first call is slow — subsequent calls are fast)...")
    t0 = time.time()
    # Use an explicit valid hyperparameter vector for compilation.
    dummy_lp = default_hyperparams()
    _ = log_likelihood_fn(dummy_lp)
    print(f" Compilation done in {time.time()-t0:.1f}s. "
          f"Test log_L = {float(_):.3f}")

    # Diagnostics at the default hyperparameter point before NUTS starts.  These
    # are NumPy postprocessing calls and do not affect the JAX likelihood path.
    reweighting_diagnostic_rows = compute_event_reweighting_diagnostics(
        np.asarray(dummy_lp), "default_pre_nuts", log_wr_fns, event_names
    )
    print_event_reweighting_diagnostics(
        reweighting_diagnostic_rows, "default_pre_nuts"
    )
    event_diag_path = os.path.join(out_dir, "event_reweighting_diagnostics.csv")
    write_event_reweighting_diagnostics_csv(
        event_diag_path, reweighting_diagnostic_rows
    )
    print(f"  Event reweighting diagnostics CSV: {event_diag_path}")

    selection_diagnostic_rows = []
    if selection_mode == "lvk_farr" and K_np is not None and cosmic is not None:
        selection_diagnostic_rows.append(compute_selection_weight_diagnostics(
            np.asarray(dummy_lp), "default_pre_nuts", K_np, log_v, cosmic
        ))
        print_selection_weight_diagnostics(selection_diagnostic_rows)
        selection_diag_path = os.path.join(out_dir, "selection_weight_diagnostics.csv")
        write_selection_weight_diagnostics_csv(
            selection_diag_path, selection_diagnostic_rows
        )
        print(f"  Selection diagnostics CSV: {selection_diag_path}")

    model = make_numpyro_model(log_likelihood_fn)

    # ---- NUTS sampling ----
    print(f"\n Running NUTS: {opts.num_chains} chains × "
          f"{opts.num_warmup} warmup + {opts.num_samples} samples")

    kernel = NUTS(model, target_accept_prob=opts.target_accept)
    mcmc   = MCMC(
        kernel,
        num_warmup  = opts.num_warmup,
        num_samples = opts.num_samples,
        num_chains  = opts.num_chains,
        progress_bar= True,
    )

    t_sample = time.time()
    mcmc.run(jax.random.PRNGKey(opts.seed))
    t_elapsed = time.time() - t_sample
    print(f"\n Sampling done in {t_elapsed:.1f}s "
          f"({t_elapsed/60:.1f} min).")

    # ---- Extract posterior ----
    samples_dict = mcmc.get_samples(group_by_chain=True)
    # samples_dict: {param: (n_chains, n_samples)} arrays

    print("\n Posterior summary:")
    mcmc.print_summary(prob=0.9)

    # ---- Convergence diagnostics ----
    # R-hat requires >= 2 chains. Guard gracefully for single-chain runs.
    n_chains_actual = opts.num_chains
    if n_chains_actual >= 2:
        r_hats = {k: float(gelman_rubin(v)) for k, v in samples_dict.items()}
    else:
        r_hats = {k: np.nan for k in POP_PARAM_NAMES}
        print("  (R-hat requires >= 2 chains — skipped for single-chain run)")

    n_effs = {k: float(effective_sample_size(v)) for k, v in samples_dict.items()}

    print("\n Convergence diagnostics:")
    print(f"  {'Parameter':<20s}  {'R-hat':>8s}  {'N_eff':>8s}")
    print("  " + "-" * 40)
    for k in POP_PARAM_NAMES:
        rh   = r_hats.get(k, np.nan)
        ne   = n_effs.get(k,  np.nan)
        flag = "  <<< WARN" if (np.isfinite(rh) and rh > 1.01) else ""
        print(f"  {k:<20s}  {'N/A' if np.isnan(rh) else f'{rh:.4f}':>8s}  "
              f"{ne:8.0f}{flag}")

    # Flat posterior: (n_chains * n_samples, n_params)
    flat_samples = np.column_stack([
        np.array(samples_dict[k]).reshape(-1) for k in POP_PARAM_NAMES
    ])

    posterior_median = np.median(flat_samples, axis=0)
    median_rows = compute_event_reweighting_diagnostics(
        posterior_median, "posterior_median", log_wr_fns, event_names
    )
    reweighting_diagnostic_rows.extend(median_rows)
    print_event_reweighting_diagnostics(median_rows, "posterior_median")

    n_diag_draws = max(0, min(int(opts.diagnostic_posterior_draws), len(flat_samples)))
    if n_diag_draws:
        rng_diag = np.random.default_rng(opts.seed + 30_001)
        idx_diag = rng_diag.choice(len(flat_samples), n_diag_draws, replace=False)
        for j, idx in enumerate(idx_diag):
            label = f"posterior_draw_{j:03d}"
            reweighting_diagnostic_rows.extend(
                compute_event_reweighting_diagnostics(
                    flat_samples[idx], label, log_wr_fns, event_names
                )
            )

    if selection_mode == "lvk_farr" and K_np is not None and cosmic is not None:
        selection_diagnostic_rows.append(compute_selection_weight_diagnostics(
            posterior_median, "posterior_median", K_np, log_v, cosmic
        ))
        if n_diag_draws:
            for j, idx in enumerate(idx_diag):
                selection_diagnostic_rows.append(compute_selection_weight_diagnostics(
                    flat_samples[idx], f"posterior_draw_{j:03d}", K_np, log_v, cosmic
                ))
        print_selection_weight_diagnostics(selection_diagnostic_rows)

    # ---- Save core outputs ----
    print(f"\n Saving to {out_dir}/...")
    samples_np = {k: np.array(v) for k, v in samples_dict.items()}
    np.savez(os.path.join(out_dir, "samples.npz"), **samples_np)
    np.save(os.path.join(out_dir, "points.npy"), flat_samples)

    write_event_reweighting_diagnostics_csv(
        event_diag_path, reweighting_diagnostic_rows
    )
    print(f"  Event reweighting diagnostics: {event_diag_path}")
    if selection_diagnostic_rows:
        write_selection_weight_diagnostics_csv(
            selection_diag_path, selection_diagnostic_rows
        )
        print(f"  Selection diagnostics: {selection_diag_path}")

    import csv
    with open(os.path.join(out_dir, "summary.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['parameter', 'mean', 'std', 'q05', 'q50', 'q95',
                         'r_hat', 'n_eff'])
        for k in POP_PARAM_NAMES:
            s = np.array(samples_dict[k]).reshape(-1)
            writer.writerow([
                k,
                f"{s.mean():.6f}", f"{s.std():.6f}",
                f"{np.percentile(s,  5):.6f}",
                f"{np.percentile(s, 50):.6f}",
                f"{np.percentile(s, 95):.6f}",
                f"{r_hats.get(k, np.nan):.6f}",
                f"{n_effs.get(k, np.nan):.1f}",
            ])

    # ---- Corner plot of population posterior ----
    try:
        import corner, matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # LaTeX-style labels for readability
        corner_labels = [
            r'$\mu_{\log\alpha_1}$', r'$\sigma_{\log\alpha_1}$',
            r'$\mu_{\log\alpha_2}$', r'$\sigma_{\log\alpha_2}$',
            r'$a_{f_1}$',              r'$b_{f_1}$',
            r'$a_{f_2}$',              r'$b_{f_2}$',
            r'$\sigma_{v_1}$ [km/s]', r'$\sigma_{v_2}$ [km/s]',
        ]

        fig_corner = corner.corner(
            flat_samples,
            labels      = corner_labels,
            quantiles   = [0.05, 0.5, 0.95],
            show_titles = True,
            title_fmt   = '.3f',
            title_kwargs= {'fontsize': 10},
            label_kwargs= {'fontsize': 11},
            levels      = [0.68, 0.95],
            smooth      = 1.0,
            plot_density    = False,
            plot_datapoints = False,
            hist_kwargs = {'linewidth': 1.5, 'density': True},
            color       = 'steelblue',
        )
        fig_corner.suptitle(
            f"Population posterior — {n_events} events  |  "
            f"{opts.config_name}  |  {selection_mode}",
            fontsize=11, y=1.01,
        )
        corner_path = os.path.join(out_dir, "corner_population.pdf")
        fig_corner.savefig(corner_path, bbox_inches='tight', dpi=200)
        plt.close(fig_corner)
        print(f"  Corner plot: {corner_path}")
    except ImportError:
        print("  Corner plot skipped (pip install corner)")

    # ---- Posterior predictive distributions (PPDs) ----
    # p(mc, q | data) = ∫ p(mc,q | Λ_pop) p(Λ_pop | data) dΛ_pop
    # Estimated by importance-weighting the COSMIC merger catalog under each
    # posterior sample of Λ_pop and averaging.
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style('ticks')

        if cosmic_raw is not None:
            m1_cos = cosmic_raw['m1_src'].astype(np.float64)
            m2_cos = cosmic_raw['m2_src'].astype(np.float64)
            mc_cos = (m1_cos * m2_cos)**(3/5) / (m1_cos + m2_cos)**(1/5)
            q_cos  = m2_cos / m1_cos
            theta_cos = cosmic_raw['theta'].astype(np.float64)
            params_cos = list(cosmic_raw['params'])
            lo_cos = cosmic_raw['lower_bound'].astype(np.float64)
            hi_cos = cosmic_raw['upper_bound'].astype(np.float64)
            kick_sigma_cos = float(
                cosmic_raw['kick_proposal_sigma'].ravel()[0]
                if 'kick_proposal_sigma' in cosmic_raw else 50.0
            )

            # Use the full COSMIC merger catalog when practical; otherwise use a
            # single random, unbiased subsample for the PPD postprocessing cost.
            rng_ppd = np.random.default_rng(opts.seed + 10_001)
            n_catalog_ppd = min(len(mc_cos), 50_000)
            if n_catalog_ppd < len(mc_cos):
                idx_catalog_ppd = rng_ppd.choice(len(mc_cos), n_catalog_ppd, replace=False)
                print(f"  PPD mass catalog: unbiased subsample "
                      f"{n_catalog_ppd:,}/{len(mc_cos):,} COSMIC mergers")
            else:
                idx_catalog_ppd = np.arange(len(mc_cos))
                print(f"  PPD mass catalog: full COSMIC merger catalog "
                      f"({len(mc_cos):,} mergers)")

            mc_cos_ppd = mc_cos[idx_catalog_ppd]
            q_cos_ppd = q_cos[idx_catalog_ppd]
            theta_cos_ppd = theta_cos[idx_catalog_ppd]
            log_q_cos_ppd = (cosmic['log_q_proposal'][idx_catalog_ppd]
                             if cosmic is not None and cosmic.get('log_q_proposal') is not None else None)
            log_pop_static_ppd = (cosmic['log_pop_static'][idx_catalog_ppd]
                                  if cosmic is not None and cosmic.get('log_pop_static') is not None else None)

            # Subsample posterior for PPD computation (up to 200 samples)
            n_ppd  = min(200, len(flat_samples))
            idx_ppd = rng_ppd.choice(len(flat_samples), n_ppd, replace=False)

            mc_ppd_all, q_ppd_all = [], []
            draws_per_hyper = 200
            for lp in flat_samples[idx_ppd]:
                log_wr_inj = compute_log_wr_injections_numpy(
                    lp,
                    theta_cos_ppd,
                    params_cos,
                    lo_cos,
                    hi_cos,
                    kick_sigma_cos,
                    log_q_cos_ppd,
                    log_pop_static_ppd,
                )
                log_wr_inj = log_wr_inj - np.max(log_wr_inj)
                wr = np.exp(log_wr_inj)
                wr_sum = wr.sum()
                if not np.isfinite(wr_sum) or wr_sum <= 0.0:
                    continue
                wr /= wr_sum
                idx_ = rng_ppd.choice(
                    len(mc_cos_ppd), size=draws_per_hyper, replace=True, p=wr
                )
                mc_ppd_all.extend(mc_cos_ppd[idx_])
                q_ppd_all.extend(q_cos_ppd[idx_])

            mc_ppd = np.array(mc_ppd_all)
            q_ppd  = np.array(q_ppd_all)

            # Per-event GW observations for comparison
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # mc distribution
            ax = axes[0]
            ax.hist(mc_ppd, bins=30, density=True, alpha=0.6,
                    color='steelblue', label='PPD (population model)',
                    histtype='stepfilled')
            # Overlay per-event mc medians
            for samp, pidx_ev, lo_ev, hi_ev, name in zip(
                    all_samples, all_pidx, all_lo, all_hi, event_names):
                if 'm1' in pidx_ev and 'q' in pidx_ev:
                    m1_ = np.array(samp[:, pidx_ev['m1']])
                    q_  = np.array(samp[:, pidx_ev['q']])
                    m2_ = q_ * m1_
                    mc_ = (m1_ * m2_)**(3/5) / (m1_ + m2_)**(1/5)
                    ax.axvline(np.median(mc_), color='gray', alpha=0.4,
                               linewidth=1, linestyle='--')
            ax.set_xlabel(r'$\mathcal{M}_c\ [M_\odot]$', fontsize=13)
            ax.set_ylabel('PDF', fontsize=12)
            ax.set_title('Chirp mass PPD', fontsize=12)
            ax.legend(fontsize=10)
            sns.despine(ax=ax)

            # q distribution
            ax = axes[1]
            ax.hist(q_ppd, bins=30, density=True, alpha=0.6,
                    color='darkorange', label='PPD (population model)',
                    histtype='stepfilled')
            for samp, pidx_ev, lo_ev, hi_ev, name in zip(
                    all_samples, all_pidx, all_lo, all_hi, event_names):
                if 'q' in pidx_ev:
                    q_ev = np.array(samp[:, pidx_ev['q']])
                    ax.axvline(np.median(q_ev), color='gray', alpha=0.4,
                               linewidth=1, linestyle='--')
            ax.set_xlabel(r'$q = m_2/m_1$', fontsize=13)
            ax.set_ylabel('PDF', fontsize=12)
            ax.set_title('Mass ratio PPD', fontsize=12)
            ax.legend(fontsize=10)
            sns.despine(ax=ax)

            fig.suptitle(
                f"Posterior Predictive Distributions — {n_events} events  |  "
                f"{opts.config_name}",
                fontsize=11,
            )
            fig.tight_layout()
            ppd_path = os.path.join(out_dir, "ppd_masses.pdf")
            fig.savefig(ppd_path, bbox_inches='tight', dpi=200)
            plt.close(fig)
            print(f"  PPD plot:    {ppd_path}")

            # ---- Kick velocity PPD ----
            fig_vk, axes_vk = plt.subplots(1, 2, figsize=(11, 4))
            vk_grid = np.linspace(0, 500, 500)

            for ax, key, title, color in zip(
                axes_vk,
                ['sigma_v1', 'sigma_v2'],
                [r'$v_{k,1}$ PPD (first SN)', r'$v_{k,2}$ PPD (second SN)'],
                ['steelblue', 'darkorange'],
            ):
                sig_samples = flat_samples[:, POP_PARAM_NAMES.index(key)]
                # PPD of vk: average Maxwell(sigma) over posterior sigma samples
                ppd_vk = np.zeros_like(vk_grid)
                for sig in sig_samples[:200]:
                    ppd_vk += np.exp(_maxwell_logpdf_numpy(vk_grid, sig, 0.0, 500.0))
                ppd_vk /= min(200, len(sig_samples))

                ax.plot(vk_grid, ppd_vk, color=color, linewidth=2,
                        label='PPD')
                ax.fill_between(vk_grid, 0, ppd_vk, alpha=0.2, color=color)

                # Shade 90% CI of sigma posterior
                sig_lo, sig_hi = np.percentile(sig_samples, [5, 95])
                ax.axvspan(sig_lo, sig_hi, alpha=0.15, color='grey',
                           label=fr'$\sigma_v$ 90% CI: [{sig_lo:.0f}, {sig_hi:.0f}] km/s')

                ax.set_xlabel(r'$v_k$ [km/s]', fontsize=13)
                ax.set_ylabel('PDF', fontsize=12)
                ax.set_title(title, fontsize=12)
                ax.legend(fontsize=9)
                sns.despine(ax=ax)

            fig_vk.suptitle(
                f"Natal Kick Velocity PPDs — {n_events} events  |  {opts.config_name}",
                fontsize=11,
            )
            fig_vk.tight_layout()
            vk_path = os.path.join(out_dir, "ppd_kicks.pdf")
            fig_vk.savefig(vk_path, bbox_inches='tight', dpi=200)
            plt.close(fig_vk)
            print(f"  Kick PPD:    {vk_path}")

    except Exception as e:
        print(f"  PPD plots skipped: {e}")

    # ---- Population distribution credible bands ----
    # These curves are the actual implied population PDFs p(theta | data), not
    # histograms of the hyperparameters that define those PDFs.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('ticks')

    try:
        def _prior_support(name, default_lo, default_hi):
            """Return plotted support from available event/injection priors."""
            lows, highs = [], []
            for pidx_ev, lo_ev, hi_ev in zip(all_pidx, all_lo, all_hi):
                if name in pidx_ev:
                    idx_ev = pidx_ev[name]
                    lows.append(float(lo_ev[idx_ev]))
                    highs.append(float(hi_ev[idx_ev]))
            if cosmic_raw is not None:
                params_inj = list(cosmic_raw['params'])
                if name in params_inj:
                    idx_inj = params_inj.index(name)
                    lows.append(float(cosmic_raw['lower_bound'][idx_inj]))
                    highs.append(float(cosmic_raw['upper_bound'][idx_inj]))
            if lows and highs:
                lo_plot = max(lows)
                hi_plot = min(highs)
                if np.isfinite(lo_plot) and np.isfinite(hi_plot) and lo_plot < hi_plot:
                    return lo_plot, hi_plot
            return default_lo, default_hi

        def _normalize_pdf(pdf, grid):
            area = np.trapz(pdf, grid)
            if np.isfinite(area) and area > 0.0:
                return pdf / area
            return pdf

        def _distribution_band(draws, grid, pdf_fn):
            pdf_draws = np.array([
                _normalize_pdf(pdf_fn(draw), grid)
                for draw in draws
            ])
            return np.percentile(pdf_draws, [2.5, 50.0, 97.5], axis=0)

        rng_pop = np.random.default_rng(opts.seed + 20_001)
        n_curve = min(1000, len(flat_samples))
        idx_curve = rng_pop.choice(len(flat_samples), n_curve, replace=False)
        curve_samples = flat_samples[idx_curve]

        alpha1_lo, alpha1_hi = _prior_support('alpha_1', 1e-2, 30.0)
        alpha2_lo, alpha2_hi = _prior_support('alpha_2', 1e-2, 30.0)
        alpha1_grid = np.geomspace(max(alpha1_lo, 1e-6), alpha1_hi, 600)
        alpha2_grid = np.geomspace(max(alpha2_lo, 1e-6), alpha2_hi, 600)
        flim_grid = np.linspace(1e-4, 1.0 - 1e-4, 600)
        vk1_lo, vk1_hi = _prior_support('vk1', 0.0, 500.0)
        vk2_lo, vk2_hi = _prior_support('vk2', 0.0, 500.0)
        vk1_grid = np.linspace(max(vk1_lo, 0.0), vk1_hi, 600)
        vk2_grid = np.linspace(max(vk2_lo, 0.0), vk2_hi, 600)

        dist_specs = [
            dict(
                grid=alpha1_grid,
                pdf=lambda draw: np.exp(_lognormal_logpdf_numpy(
                    alpha1_grid,
                    draw[POP_PARAM_NAMES.index('mu_logalpha1')],
                    draw[POP_PARAM_NAMES.index('sig_logalpha1')],
                    alpha1_lo,
                    alpha1_hi,
                )),
                xlabel=r'$\alpha_1$',
                ylabel=r'$p(\alpha_1\mid\mathrm{data})$',
                title=r'CE efficiency $\alpha_1$',
                color='steelblue',
                xscale='log',
            ),
            dict(
                grid=alpha2_grid,
                pdf=lambda draw: np.exp(_lognormal_logpdf_numpy(
                    alpha2_grid,
                    draw[POP_PARAM_NAMES.index('mu_logalpha2')],
                    draw[POP_PARAM_NAMES.index('sig_logalpha2')],
                    alpha2_lo,
                    alpha2_hi,
                )),
                xlabel=r'$\alpha_2$',
                ylabel=r'$p(\alpha_2\mid\mathrm{data})$',
                title=r'CE efficiency $\alpha_2$',
                color='darkorange',
                xscale='log',
            ),
            dict(
                grid=flim_grid,
                pdf=lambda draw: sp_stats.beta.pdf(
                    flim_grid,
                    draw[POP_PARAM_NAMES.index('a_f1')],
                    draw[POP_PARAM_NAMES.index('b_f1')],
                ),
                xlabel=r'$f_{\mathrm{lim},1}$',
                ylabel=r'$p(f_{\mathrm{lim},1}\mid\mathrm{data})$',
                title=r'Stable MT accretion limit $f_{\mathrm{lim},1}$',
                color='seagreen',
                xscale='linear',
            ),
            dict(
                grid=flim_grid,
                pdf=lambda draw: sp_stats.beta.pdf(
                    flim_grid,
                    draw[POP_PARAM_NAMES.index('a_f2')],
                    draw[POP_PARAM_NAMES.index('b_f2')],
                ),
                xlabel=r'$f_{\mathrm{lim},2}$',
                ylabel=r'$p(f_{\mathrm{lim},2}\mid\mathrm{data})$',
                title=r'Stable MT accretion limit $f_{\mathrm{lim},2}$',
                color='purple',
                xscale='linear',
            ),
            dict(
                grid=vk1_grid,
                pdf=lambda draw: np.exp(_maxwell_logpdf_numpy(
                    vk1_grid,
                    draw[POP_PARAM_NAMES.index('sigma_v1')],
                    vk1_lo,
                    vk1_hi,
                )),
                xlabel=r'$v_{k,1}$ [km/s]',
                ylabel=r'$p(v_{k,1}\mid\mathrm{data})$',
                title=r'Natal kick $v_{k,1}$',
                color='firebrick',
                xscale='linear',
            ),
            dict(
                grid=vk2_grid,
                pdf=lambda draw: np.exp(_maxwell_logpdf_numpy(
                    vk2_grid,
                    draw[POP_PARAM_NAMES.index('sigma_v2')],
                    vk2_lo,
                    vk2_hi,
                )),
                xlabel=r'$v_{k,2}$ [km/s]',
                ylabel=r'$p(v_{k,2}\mid\mathrm{data})$',
                title=r'Natal kick $v_{k,2}$',
                color='teal',
                xscale='linear',
            ),
        ]

        fig_pop, axes_pop = plt.subplots(3, 2, figsize=(12, 12))
        for ax, spec in zip(axes_pop.flat, dist_specs):
            pdf_lo, pdf_med, pdf_hi = _distribution_band(
                curve_samples, spec['grid'], spec['pdf']
            )
            ax.plot(spec['grid'], pdf_med, color=spec['color'], linewidth=2,
                    label='posterior median PDF')
            ax.fill_between(spec['grid'], pdf_lo, pdf_hi,
                            color=spec['color'], alpha=0.25,
                            label='95% credible band')
            ax.set_xscale(spec['xscale'])
            ax.set_xlabel(spec['xlabel'], fontsize=12)
            ax.set_ylabel(spec['ylabel'], fontsize=11)
            ax.set_title(spec['title'], fontsize=12)
            ax.legend(fontsize=8)
            sns.despine(ax=ax)

        fig_pop.suptitle(
            f"Population distribution PDFs — {n_events} events  |  {opts.config_name}",
            fontsize=12,
        )
        fig_pop.tight_layout()
        pop_dist_path = os.path.join(out_dir, "population_distributions_95ci.pdf")
        fig_pop.savefig(pop_dist_path, bbox_inches='tight', dpi=200)
        plt.close(fig_pop)
        print(f"  Pop PDFs:    {pop_dist_path} (n_curve={n_curve})")

    except Exception as e:
        print(f"  Population distribution plot skipped: {e}")

    # ---- CE and flim hyperparameter summary plots ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style('ticks')

        fig_ce, axes_ce = plt.subplots(2, 4, figsize=(16, 8))

        for ax, key, label in zip(
            axes_ce.flat,
            ['mu_logalpha1', 'sig_logalpha1', 'mu_logalpha2', 'sig_logalpha2',
             'a_f1', 'b_f1', 'a_f2', 'b_f2'],
            [r'$\mu_{\log\alpha_1}$', r'$\sigma_{\log\alpha_1}$',
             r'$\mu_{\log\alpha_2}$', r'$\sigma_{\log\alpha_2}$',
             r'$a_{f_1}$', r'$b_{f_1}$', r'$a_{f_2}$', r'$b_{f_2}$'],
        ):
            idx_k = POP_PARAM_NAMES.index(key)
            s     = flat_samples[:, idx_k]
            ax.hist(s, bins=40, density=True, color='steelblue',
                    alpha=0.8, histtype='stepfilled', linewidth=1.2)
            ax.axvline(np.median(s), color='k', linestyle='--',
                       linewidth=1.5, label=f'median={np.median(s):.2f}')
            lo5, hi95 = np.percentile(s, [5, 95])
            ax.axvspan(lo5, hi95, alpha=0.15, color='steelblue',
                       label=f'90% CI')
            ax.set_xlabel(label, fontsize=12)
            ax.set_ylabel('PDF', fontsize=10)
            ax.legend(fontsize=8)
            sns.despine(ax=ax)

        fig_ce.suptitle(
            f"CE efficiency and accretion hyperparameter posteriors — {n_events} events  |  {opts.config_name}",
            fontsize=12,
        )
        fig_ce.tight_layout()
        ce_path = os.path.join(out_dir, "posteriors_CE_flim.pdf")
        fig_ce.savefig(ce_path, bbox_inches='tight', dpi=200)
        plt.close(fig_ce)
        print(f"  CE/flim:     {ce_path}")

    except Exception as e:
        print(f"  CE/flim plots skipped: {e}")

    # ---- Metadata ----
    elapsed_total = time.time() - start_time
    jax_devices = [str(device) for device in jax.devices()]
    metadata = dict(
        **base_runtime_metadata("."),
        package_versions=get_package_versions(["numpy", "scipy", "jax", "jaxlib", "numpyro"]),
        event_names      = event_names,
        config_name      = opts.config_name,
        pop_param_names  = POP_PARAM_NAMES,
        default_hyperparams = np.array([default_hyperparams(as_vector=False)[k] for k in POP_PARAM_NAMES]),
        hyperprior_descriptions = np.array(HYPERPRIOR_DESCRIPTIONS, dtype=object),
        event_injection_support_bounds = np.array(EVENT_INJECTION_SUPPORT_BOUNDS, dtype=object),
        selection_mode   = selection_mode,
        num_warmup       = opts.num_warmup,
        num_samples      = opts.num_samples,
        num_chains       = opts.num_chains,
        r_hats           = [r_hats.get(k, np.nan) for k in POP_PARAM_NAMES],
        n_effs           = [n_effs.get(k, np.nan)  for k in POP_PARAM_NAMES],
        n_events         = n_events,
        wall_time_s      = elapsed_total,
        jax_backend      = jax.default_backend(),
        strict_lvk_sampling_pdf = opts.strict_lvk_sampling_pdf,
        lvk_sampling_pdf_verified = bool(lvk_sampling_pdf_metadata.get('verified', False)),
        lvk_sampling_pdf_status = str(lvk_sampling_pdf_metadata.get('status', 'unknown')),
        lvk_sampling_pdf_assumed_measure = str(lvk_sampling_pdf_metadata.get('assumed_measure', 'unknown')),
        lvk_sampling_pdf_validation_message = str(lvk_sampling_pdf_metadata.get('message', '')),
        lvk_sampling_pdf_root_attrs = np.array(lvk_sampling_pdf_metadata.get('root_attrs', {}), dtype=object),
        lvk_sampling_pdf_injection_attrs = np.array(lvk_sampling_pdf_metadata.get('injection_attrs', {}), dtype=object),
        lvk_sampling_pdf_dataset_attrs = np.array(lvk_sampling_pdf_metadata.get('dataset_attrs', {}), dtype=object),
        lvk_N_found_total = N_found_total,
        lvk_N_found_used = N_found_used,
        lvk_found_subsample_log_scaling = lvk_found_subsample_log_scale,
        allow_inconsistent_selection_model = opts.allow_inconsistent_selection_model,
        selection_model_consistent = selection_model_consistent if selection_mode == "lvk_farr" else True,
        selection_model_consistency_message = selection_model_consistency_message if selection_mode == "lvk_farr" else "Selection disabled; no injection metadata compared.",
        event_model_metadata = np.array(event_model_metadata, dtype=object),
        injection_model_metadata = injection_model_metadata if selection_mode == "lvk_farr" else {},
        injection_proposal_version = injection_model_metadata.get("proposal_version", "unknown") if selection_mode == "lvk_farr" else "not_used",
        selection_consistency_checks = dict(
            allow_inconsistent_selection_model=bool(opts.allow_inconsistent_selection_model),
            selection_model_consistent=bool(selection_model_consistent if selection_mode == "lvk_farr" else True),
            message=selection_model_consistency_message if selection_mode == "lvk_farr" else "Selection disabled; no injection metadata compared.",
        ),
        lvk_sampling_pdf_validation_status = dict(
            verified=bool(lvk_sampling_pdf_metadata.get('verified', False)),
            status=str(lvk_sampling_pdf_metadata.get('status', 'unknown')),
            assumed_measure=str(lvk_sampling_pdf_metadata.get('assumed_measure', 'unknown')),
            message=str(lvk_sampling_pdf_metadata.get('message', '')),
        ),
        found_injection_subsampling_counts = dict(
            N_found_total=int(N_found_total),
            N_found_used=int(N_found_used),
            log_scaling=float(lvk_found_subsample_log_scale),
        ),
        jax_x64_enabled=bool(jax.config.jax_enable_x64),
        jax_devices=jax_devices,
        jax_default_device=str(jax.devices()[0]) if jax.devices() else "none",
    )
    save_metadata(out_dir, metadata)
    with open(os.path.join(out_dir, "hyperpriors.json"), "w", encoding="utf-8") as f:
        json.dump({
            "hyperparameter_names": POP_PARAM_NAMES,
            "default_hyperparams": default_hyperparams(as_vector=False),
            "hyperprior_descriptions": HYPERPRIOR_DESCRIPTIONS,
            "note": "NumPyro priors are the actual hyperpriors; event_injection_support_bounds describe physical event/injection variables, not hyperparameter sampling bounds.",
        }, f, indent=2)

    print(f"\n Total wall time: {elapsed_total/60:.1f} min")
    print(" Done.")


if __name__ == "__main__":
    main()
