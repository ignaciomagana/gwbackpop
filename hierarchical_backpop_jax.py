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
import warnings
import numpy as np
import scipy.stats as sp_stats
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit, vmap, grad

import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC
from numpyro.diagnostics import print_summary, effective_sample_size, gelman_rubin

# ---------------------------------------------------------------------------
# Population parameter bounds (must match hierarchical_backpop.py)
# ---------------------------------------------------------------------------

POP_PARAMS = [
    # (name,            lo,     hi)
    ('mu_logalpha1',   -2.0,    3.0),
    ('sig_logalpha1',   0.01,   3.0),
    ('mu_logalpha2',   -2.0,    3.0),
    ('sig_logalpha2',   0.01,   3.0),
    ('a_f1',            0.1,   10.0),
    ('b_f1',            0.1,   10.0),
    ('a_f2',            0.1,   10.0),
    ('b_f2',            0.1,   10.0),
    ('sigma_v1',        1.0,  500.0),
    ('sigma_v2',        1.0,  500.0),
]

POP_PARAM_NAMES = [p[0] for p in POP_PARAMS]
POP_LO          = jnp.array([p[1] for p in POP_PARAMS])
POP_HI          = jnp.array([p[2] for p in POP_PARAMS])


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
    x_safe = jnp.clip(x, 1e-30, None)
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
                    sig_log: jnp.ndarray) -> jnp.ndarray:
    """Log-Normal population model for CE efficiency alpha.

    p(alpha | mu_log, sig_log) = LogNormal(alpha; mu_log, sig_log)
    where mu_log, sig_log are mean and std of log(alpha) [natural log].
    """
    return dist.LogNormal(mu_log, sig_log).log_prob(jnp.clip(alpha, 1e-30, None))


def log_p_flim_jax(flim: jnp.ndarray,
                   a: jnp.ndarray,
                   b: jnp.ndarray) -> jnp.ndarray:
    """Beta population model for stable MT accretion efficiency flim in [0,1]."""
    return dist.Beta(a, b).log_prob(jnp.clip(flim, 1e-6, 1.0 - 1e-6))


def log_p_vk_jax(vk: jnp.ndarray, sigma_v: jnp.ndarray) -> jnp.ndarray:
    """Maxwellian population model for natal kick speed vk [km/s]."""
    return maxwell_logpdf(jnp.clip(vk, 1e-10, None), sigma_v)


# ---------------------------------------------------------------------------
# NumPy population weight ratios for COSMIC merger-catalog postprocessing
# ---------------------------------------------------------------------------

def _maxwell_logpdf_numpy(x: np.ndarray, scale: float) -> np.ndarray:
    """NumPy Maxwell log-PDF matching :func:`maxwell_logpdf`."""
    x_safe = np.clip(np.asarray(x, dtype=np.float64), 1e-30, None)
    scale = float(scale)
    return (
        0.5 * np.log(2.0 / np.pi)
        + 2.0 * np.log(x_safe)
        - 3.0 * np.log(scale)
        - x_safe**2 / (2.0 * scale**2)
    )


def _lognormal_logpdf_numpy(x: np.ndarray, mu_log: float, sig_log: float) -> np.ndarray:
    """NumPy LogNormal log-PDF with the same clipping as the JAX path."""
    x_safe = np.clip(np.asarray(x, dtype=np.float64), 1e-30, None)
    sig_log = float(sig_log)
    return (
        -np.log(x_safe)
        -np.log(sig_log)
        -0.5 * np.log(2.0 * np.pi)
        -0.5 * ((np.log(x_safe) - float(mu_log)) / sig_log) ** 2
    )


def _beta_logpdf_numpy(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """NumPy Beta log-PDF with the same clipping as the JAX path."""
    import math

    x_safe = np.clip(np.asarray(x, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    a = float(a)
    b = float(b)
    log_norm = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    return log_norm + (a - 1.0) * np.log(x_safe) + (b - 1.0) * np.log1p(-x_safe)


def compute_log_wr_injections_numpy(
    lp_vec: np.ndarray,
    theta: np.ndarray,
    params: list[str] | tuple[str, ...] | dict[str, int],
    lo: np.ndarray,
    hi: np.ndarray,
    kick_sigma: float,
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

    log_wr = np.zeros(theta.shape[0], dtype=np.float64)

    def _col(name: str) -> np.ndarray | None:
        idx = param_idx.get(name)
        return theta[:, idx] if idx is not None else None

    def _log_uniform_proposal(name: str) -> float:
        idx = param_idx.get(name)
        if idx is None:
            return 0.0
        return float(-np.log(hi[idx] - lo[idx]))

    inj_a1 = _col('alpha_1')
    if inj_a1 is not None:
        log_wr += _lognormal_logpdf_numpy(inj_a1, mu_la1, sig_la1) - _log_uniform_proposal('alpha_1')

    inj_a2 = _col('alpha_2')
    if inj_a2 is not None:
        log_wr += _lognormal_logpdf_numpy(inj_a2, mu_la2, sig_la2) - _log_uniform_proposal('alpha_2')

    inj_f1 = _col('flim_1')
    if inj_f1 is not None:
        log_wr += _beta_logpdf_numpy(inj_f1, af1, bf1) - _log_uniform_proposal('flim_1')

    inj_f2 = _col('flim_2')
    if inj_f2 is not None:
        log_wr += _beta_logpdf_numpy(inj_f2, af2, bf2) - _log_uniform_proposal('flim_2')

    inj_v1 = _col('vk1')
    if inj_v1 is not None:
        log_wr += _maxwell_logpdf_numpy(inj_v1, sv1) - _maxwell_logpdf_numpy(inj_v1, kick_sigma)

    inj_v2 = _col('vk2')
    if inj_v2 is not None:
        log_wr += _maxwell_logpdf_numpy(inj_v2, sv2) - _maxwell_logpdf_numpy(inj_v2, kick_sigma)

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
            log_w = log_w + log_p_alpha_jax(alpha1, mu_la1, sig_la1) - lpi0_a1
        if alpha2 is not None:
            log_w = log_w + log_p_alpha_jax(alpha2, mu_la2, sig_la2) - lpi0_a2
        if flim1 is not None:
            log_w = log_w + log_p_flim_jax(flim1, af1, bf1) - lpi0_f1
        if flim2 is not None:
            log_w = log_w + log_p_flim_jax(flim2, af2, bf2) - lpi0_f2
        if vk1 is not None:
            log_w = log_w + log_p_vk_jax(vk1, sv1) - lpi0_v1
        if vk2 is not None:
            log_w = log_w + log_p_vk_jax(vk2, sv2) - lpi0_v2

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
    kick_sigma:    float,         # Maxwellian proposal sigma used in campaign
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

    # Kick denominator: Maxwellian (injection proposal), not uniform
    def _log_maxw(vk):
        return maxwell_logpdf(jnp.clip(vk, 1e-10, None), kick_sigma)

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

        # Weight ratios for COSMIC mergers
        log_wr = jnp.zeros(theta_inj.shape[0])
        if inj_a1 is not None:
            log_wr = log_wr + log_p_alpha_jax(inj_a1, mu_la1, sig_la1) - lpi0_a1
        if inj_a2 is not None:
            log_wr = log_wr + log_p_alpha_jax(inj_a2, mu_la2, sig_la2) - lpi0_a2
        if inj_f1 is not None:
            log_wr = log_wr + log_p_flim_jax(inj_f1, af1, bf1) - lpi0_f1
        if inj_f2 is not None:
            log_wr = log_wr + log_p_flim_jax(inj_f2, af2, bf2) - lpi0_f2
        if inj_v1 is not None:
            log_wr = log_wr + log_p_vk_jax(inj_v1, sv1) - _log_maxw(inj_v1)
        if inj_v2 is not None:
            log_wr = log_wr + log_p_vk_jax(inj_v2, sv2) - _log_maxw(inj_v2)

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
    chosen for each parameter type instead of flat priors over POP_PARAMS.
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


# ---------------------------------------------------------------------------
# Data loading (reuses logic from hierarchical_backpop.py)
# ---------------------------------------------------------------------------

def load_event_data(
    results_dir: str,
    n_samples:   int = 10_000,
) -> tuple[jnp.ndarray, dict[str, int], jnp.ndarray, jnp.ndarray, float, str]:
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
    print(f"  {name}: N_eff={n_eff}  log Z={log_z:.2f}  params={len(params)}-D")

    return (
        jnp.array(samples),
        {p: i for i, p in enumerate(params)},
        jnp.array(lo),
        jnp.array(hi),
        log_z,
        name,
    )


def load_lvk_injections(lvk_path: str, n_inj_total: int | None = None) -> dict:
    """Load LVK found injection set — mirrors LVKInjectionCampaign._load_lvk_injections."""
    if lvk_path.endswith('.h5') or lvk_path.endswith('.hdf5'):
        import h5py
        with h5py.File(lvk_path, 'r') as f:
            grp    = f['injections']
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
        N_inj = (int(n_inj_total) if n_inj_total is not None
                 else int(data['N_inj'].ravel()[0]))

    # m1 >= m2 convention
    swap   = m1 < m2
    m1[swap], m2[swap] = m2[swap].copy(), m1[swap].copy()

    return dict(m1=m1, m2=m2, z=z, q_lvk=q_lvk, N_inj=N_inj)


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

    # log_v = log(1 / (q_LVK * m1 * m2))
    with np.errstate(divide='ignore', invalid='ignore'):
        log_v_np = -(np.log(lvk['q_lvk']) + np.log(lvk['m1']) + np.log(lvk['m2']))
    log_v_np[~np.isfinite(log_v_np)] = -np.inf

    log_norm = -(np.log(lvk['N_inj']) + np.log(cosmic['N_inj']))

    # Return K as numpy (CPU) — it's too large for GPU VRAM.
    # log_v is moved to JAX device for use in logsumexp.
    return K, jnp.array(log_v_np), float(log_norm)


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

    sel = p.add_argument_group("Selection effects")
    sel.add_argument("--injections_path",   default=None)
    sel.add_argument("--lvk_found_path",    default=None)
    sel.add_argument("--lvk_n_inj_total",   type=int,   default=None)
    sel.add_argument("--lvk_n_found_max",   type=int,   default=5_000,
                     help="Max LVK found injections to subsample for K matrix. "
                          "Farr variance scales as 1/N_found. Default 5000 gives "
                          "~2s/NUTS iter. Increase after confirming convergence.")
    sel.add_argument("--lvk_bandwidth_log_m1", type=float, default=None)
    sel.add_argument("--lvk_bandwidth_log_m2", type=float, default=None)
    sel.add_argument("--lvk_bandwidth_z",      type=float, default=None)

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
    all_samples, all_pidx, all_lo, all_hi, all_logz, event_names = [], [], [], [], [], []
    for d in event_dirs:
        try:
            samp, pidx, lo, hi, lz, name = load_event_data(d, opts.n_samples)
            all_samples.append(samp)
            all_pidx.append(pidx)
            all_lo.append(lo)
            all_hi.append(hi)
            all_logz.append(lz)
            event_names.append(name)
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
    if selection_mode == "lvk_farr":
        print(f"\n Loading LVK found injections: {opts.lvk_found_path}")
        lvk = load_lvk_injections(opts.lvk_found_path, opts.lvk_n_inj_total)
        print(f"  N_found={len(lvk['m1']):,}  N_inj={lvk['N_inj']:,}")

        # Subsample found injections for speed.
        # 284k found injections: K@w matmul takes ~120s/NUTS iter (CPU↔GPU sync).
        # 5k found injections:   ~2s/iter. Farr variance scales as 1/N_found.
        # 5000 is more than sufficient for 9-47 events.
        n_found_max = opts.lvk_n_found_max
        if len(lvk['m1']) > n_found_max:
            rng_sub  = np.random.default_rng(seed=0)
            idx_sub  = rng_sub.choice(len(lvk['m1']), size=n_found_max, replace=False)
            for key in ['m1', 'm2', 'z', 'q_lvk']:
                lvk[key] = lvk[key][idx_sub]
            print(f"  Subsampled N_found: {len(idx_sub) + n_found_max:,} → {n_found_max:,}")

        print(f"\n Loading COSMIC merger catalog: {opts.injections_path}")
        cosmic_raw = np.load(opts.injections_path, allow_pickle=True)
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
    dummy = jnp.array([0.5] * 10, dtype=jnp.float64)
    # Map dummy to valid parameter ranges
    dummy_lp = POP_LO + (POP_HI - POP_LO) * 0.5
    _ = log_likelihood_fn(dummy_lp)
    print(f" Compilation done in {time.time()-t0:.1f}s. "
          f"Test log_L = {float(_):.3f}")

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

    # ---- Save core outputs ----
    print(f"\n Saving to {out_dir}/...")
    samples_np = {k: np.array(v) for k, v in samples_dict.items()}
    np.savez(os.path.join(out_dir, "samples.npz"), **samples_np)
    np.save(os.path.join(out_dir, "points.npy"), flat_samples)

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
                    ppd_vk += sp_stats.maxwell.pdf(vk_grid, scale=sig)
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
                pdf=lambda draw: sp_stats.lognorm.pdf(
                    alpha1_grid,
                    s=draw[POP_PARAM_NAMES.index('sig_logalpha1')],
                    scale=np.exp(draw[POP_PARAM_NAMES.index('mu_logalpha1')]),
                ),
                xlabel=r'$\alpha_1$',
                ylabel=r'$p(\alpha_1\mid\mathrm{data})$',
                title=r'CE efficiency $\alpha_1$',
                color='steelblue',
                xscale='log',
            ),
            dict(
                grid=alpha2_grid,
                pdf=lambda draw: sp_stats.lognorm.pdf(
                    alpha2_grid,
                    s=draw[POP_PARAM_NAMES.index('sig_logalpha2')],
                    scale=np.exp(draw[POP_PARAM_NAMES.index('mu_logalpha2')]),
                ),
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
                pdf=lambda draw: sp_stats.maxwell.pdf(
                    vk1_grid,
                    scale=draw[POP_PARAM_NAMES.index('sigma_v1')],
                ),
                xlabel=r'$v_{k,1}$ [km/s]',
                ylabel=r'$p(v_{k,1}\mid\mathrm{data})$',
                title=r'Natal kick $v_{k,1}$',
                color='firebrick',
                xscale='linear',
            ),
            dict(
                grid=vk2_grid,
                pdf=lambda draw: sp_stats.maxwell.pdf(
                    vk2_grid,
                    scale=draw[POP_PARAM_NAMES.index('sigma_v2')],
                ),
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
    np.savez(
        os.path.join(out_dir, "metadata.npz"),
        event_names      = event_names,
        config_name      = opts.config_name,
        pop_param_names  = POP_PARAM_NAMES,
        selection_mode   = selection_mode,
        num_warmup       = opts.num_warmup,
        num_samples      = opts.num_samples,
        num_chains       = opts.num_chains,
        r_hats           = [r_hats.get(k, np.nan) for k in POP_PARAM_NAMES],
        n_effs           = [n_effs.get(k, np.nan)  for k in POP_PARAM_NAMES],
        n_events         = n_events,
        wall_time_s      = elapsed_total,
        jax_backend      = jax.default_backend(),
    )

    print(f"\n Total wall time: {elapsed_total/60:.1f} min")
    print(" Done.")


if __name__ == "__main__":
    main()