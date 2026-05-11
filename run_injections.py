"""
run_injections.py
-----------------
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
  - vk1, vk2:  Maxwell    / Uniform[0, 500]
  - z_form:    1.0  (drew from SFR prior = population prior for z_form)
  - logZ:      1.0  (drew from P(logZ|z_form) = population prior for logZ)
  - m1, q, logtb, angles: 1.0  (flat in both population and proposal)

This means InjectionCampaign.log_weight_ratio only touches the 6 binary
physics dimensions, identical to EventPosterior.log_weight_ratio.

Usage
-----
  python run_injections.py \\
      --pdet_path     /path/to/pdet_interpolator.pkl \\
      --output_path   injections/gwtc3_injections.npz \\
      --n_inj         1000000 \\
      --n_workers     64

  # Estimate merger fraction first (fast, N=10000):
  python run_injections.py \\
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
import numpy as np
from argparse import ArgumentParser
from multiprocessing import Pool, cpu_count
from scipy.stats import maxwell

# ---------------------------------------------------------------------------
# Parameter space (same as backpop.get_backpop_config 'lucky_strikes')
# z_form and logZ are drawn separately from the SFR prior — not from π₀
# ---------------------------------------------------------------------------

PARAMS = [
    'm1', 'q', 'logtb',
    'alpha_1', 'alpha_2', 'flim_1', 'flim_2',
    'vk1', 'theta1', 'phi1', 'omega1',
    'vk2', 'theta2', 'phi2', 'omega2',
]

LOWER = np.array([50,  0.01, 0.0,   0.1, 0.1, 0.0, 0.0,   0,   0, -90,   0,   0,   0, -90,   0])
UPPER = np.array([150, 1.0,  3.699, 20,  20,  1.0, 1.0, 500, 360,  90, 360, 500, 360,  90, 360])

# Kick velocity injection proposal: Maxwellian(KICK_PROPOSAL_SIGMA km/s)
# The uniform prior [0, 500] has f_merge ~ 0 because large kicks disrupt all binaries.
# A Maxwellian concentrates draws where mergers actually occur while maintaining
# full support over [0, 500].  The weight ratio in InjectionCampaign / LVKInjectionCampaign
# divides by this proposal density rather than 1/500.
# Choice: σ=50 km/s gives median kick ~63 km/s.  BH kicks from fallback
# are typically O(10-100) km/s for massive progenitors (Fryer+2012, Mandel+2016).
KICK_PROPOSAL_SIGMA: float = 50.0   # km/s  — Maxwellian scale parameter

# logZ drawn from P(logZ|z_form); bounds used only for rejection safeguard
LOGZ_LO = np.log10(1e-4)
LOGZ_HI = np.log10(0.03)

# z_form drawn from SFR prior; upper limit where SFR is negligible
ZFORM_MAX = 15.0


# ---------------------------------------------------------------------------
# Worker function (runs in subprocess — must be importable at module level)
# ---------------------------------------------------------------------------

def _worker_init(pdet_path: str | None) -> None:
    """Load P_det interpolator once per worker process.
    If pdet_path is None, P_det evaluation is skipped and pdet=nan is stored.
    Use this mode when building the COSMIC merger catalog for LVKInjectionCampaign.
    """
    global _PDET
    if pdet_path is None:
        _PDET = None
        return
    with open(pdet_path, 'rb') as f:
        _PDET = pickle.load(f)


def _run_one(seed: int) -> dict | None:
    """Draw one injection, run COSMIC, return result dict or None.

    Returns None if binary doesn't form a merging BBH — these are discarded
    (contribute P_det = 0 to α regardless of Λ_pop).
    """
    rng = np.random.default_rng(seed)

    # ---- Draw binary physics + ZAMS ----
    # Non-kick parameters: flat prior π₀ (uniform over LOWER, UPPER)
    # Kick speeds: Maxwellian(KICK_PROPOSAL_SIGMA) clipped to [0, 500]
    # Kick angles: uniform — isotropic in population model anyway
    # Rationale: uniform [0,500] km/s gives f_merge ≈ 0 because most kicks
    # disrupt the binary.  Maxwellian concentrates where mergers occur
    # while maintaining support over the full [0,500] range.
    theta = rng.uniform(LOWER, UPPER)
    params_dict = dict(zip(PARAMS, theta))

    # Overwrite vk1 and vk2 with Maxwellian draws (clip to prior range)
    vk1_raw = maxwell.rvs(scale=KICK_PROPOSAL_SIGMA, random_state=rng)
    vk2_raw = maxwell.rvs(scale=KICK_PROPOSAL_SIGMA, random_state=rng)
    params_dict['vk1'] = float(np.clip(vk1_raw, 0.0, UPPER[PARAMS.index('vk1')]))
    params_dict['vk2'] = float(np.clip(vk2_raw, 0.0, UPPER[PARAMS.index('vk2')]))
    # Reflect truncation into theta for storage
    theta[PARAMS.index('vk1')] = params_dict['vk1']
    theta[PARAMS.index('vk2')] = params_dict['vk2']

    # ---- Draw z_form from SFR prior via inverse CDF (rejection sampling) ----
    # Import here so the worker subprocess gets its own copy
    from cosmo_prior import log_prior_z_form, log_prior_logZ_given_z

    z_form = _draw_z_form(rng)
    if z_form is None:
        return None

    # ---- Draw logZ from P(logZ | z_form) via rejection sampling ----
    log10_Z = _draw_logZ_given_z(rng, z_form)
    if log10_Z is None:
        return None

    params_dict['logZ'] = log10_Z

    # ---- Run COSMIC ----
    try:
        from backpop import evolv2
        final_state, bpp_raw, _ = evolv2(
            params_dict, ['mass_1', 'mass_2'], fixed_params={}
        )
    except Exception:
        return None

    if final_state is None:
        return None   # no BBH merger within Hubble time

    m1_src = float(final_state['mass_1'])
    m2_src = float(final_state['mass_2'])
    if m1_src < m2_src:
        m1_src, m2_src = m2_src, m1_src

    # ---- Delay time → z_merger ----
    from cosmo_prior import z_merger_from_t_delay
    import pandas as pd
    from backpop import COLS_KEEP

    # Extract t_delay from bpp
    bpp = pd.DataFrame(bpp_raw, columns=COLS_KEEP)
    merger_rows = bpp.loc[
        (bpp.kstar_1 == 14) & (bpp.kstar_2 == 14) & (bpp.evol_type == 3)
    ]
    if len(merger_rows) == 0:
        return None
    t_delay = float(merger_rows['tphys'].iloc[0])   # Myr

    z_merger = z_merger_from_t_delay(z_form, t_delay)
    if z_merger is None:
        return None   # merges in future or before formation

    # ---- P_det(m1_src, m2_src, z_merger) ----
    # If _PDET is None (LVK raw-injection mode), store nan — pdet is evaluated
    # later by LVKInjectionCampaign using the found injection set directly.
    if _PDET is None:
        pdet = np.nan
    else:
        try:
            pdet = float(_PDET(m1_src, m2_src, z_merger))
            pdet = float(np.clip(pdet, 0.0, 1.0))
        except Exception:
            return None

    return dict(
        theta      = theta,
        logZ       = log10_Z,
        z_form     = z_form,
        m1_src     = m1_src,
        m2_src     = m2_src,
        z_merger   = z_merger,
        t_delay    = t_delay,
        pdet       = pdet,
    )


def _draw_z_form(rng: np.random.Generator, max_tries: int = 200) -> float | None:
    """Draw z_form from P(z_form) ∝ (dV_c/dz) ψ(z) / (1+z) via rejection sampling.

    Uses a Uniform[0, ZFORM_MAX] proposal with the prior as the accept/reject weight.
    The normalised prior peaks near z~2 with max value computable from the table.
    """
    from cosmo_prior import _prior_weight_grid, _zgrid, _prior_norm

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
    """Draw logZ from P(logZ | z_form) — truncated Normal via rejection sampling."""
    from cosmo_prior import _log_Z_mean_interp, _SIGMA_LOG_Z, _LOG_Z_MIN, _LOG_Z_MAX

    mu  = float(_log_Z_mean_interp(z_form))
    sig = _SIGMA_LOG_Z

    for _ in range(max_tries):
        logZ = rng.normal(mu, sig)
        if _LOG_Z_MIN <= logZ <= _LOG_Z_MAX:
            return float(logZ)
    return None


# ---------------------------------------------------------------------------
# Campaign runner
# ---------------------------------------------------------------------------

def run_campaign(
    pdet_path: str | None,
    output_path: str,
    n_inj: int,
    n_workers: int,
    chunk_size: int = 500,
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
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    start = time.time()

    # Seeds: deterministic, reproducible, one per injection
    seeds = np.arange(n_inj, dtype=np.int64)

    results = []
    n_done  = 0

    print(f"[injections] n_inj={n_inj:,}  n_workers={n_workers}")
    print(f"[injections] pdet: {pdet_path}")
    print(f"[injections] output: {output_path}")
    print()

    with Pool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(pdet_path,),
    ) as pool:
        for i in range(0, n_inj, chunk_size):
            batch = seeds[i : i + chunk_size]
            batch_results = pool.map(_run_one, batch)
            results.extend([r for r in batch_results if r is not None])
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
    else:
        print("  P_det: not evaluated (LVK raw-injection mode — pdet=nan stored)")
    print(f"  m1_src range   = [{m1_src.min():.1f}, {m1_src.max():.1f}] M_sun")
    print(f"  z_merger range = [{z_merger.min():.3f}, {z_merger.max():.3f}]")

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
        params       = PARAMS,
        lower_bound  = LOWER,
        upper_bound  = UPPER,
        N_inj                = np.array([n_inj]),
        N_merge              = np.array([n_merge]),
        N_workers            = np.array([n_workers]),
        kick_proposal_sigma  = np.array([KICK_PROPOSAL_SIGMA]),
        wall_time_s          = np.array([elapsed]),
    )
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = ArgumentParser(description="BackPop injection campaign for selection effects.")
    p.add_argument("--pdet_path",   default=None,
                   help="Path to pickled P_det(m1_src, m2_src, z) interpolator. "
                        "If omitted, pdet values are not computed and stored as nan. "
                        "Use this when building the COSMIC merger catalog for "
                        "LVKInjectionCampaign (raw found-injection Farr estimator).")
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
    return p.parse_args()


def main():
    opts = parse_args()
    dry  = opts.dry_run.lower() in ('true', 't', '1', 'yes')
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
    )


if __name__ == "__main__":
    main()