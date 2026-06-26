"""
run_backpop.py
--------------
Single-event BackPop inference with Nautilus nested sampling.

Two likelihood modes, selectable via --use_redshift_likelihood:

  2D mode (default, --use_redshift_likelihood False):
    KDE over (mc, q).  Matches Lucky Strikes exactly.  logZ has a flat prior.

  3D mode (--use_redshift_likelihood True):
    KDE over (mc, q, z_merger_predicted).
    Adds z_form as a sampled parameter and applies:
      - P(z_form): SFR-weighted comoving volume prior (Madau & Dickinson 2014)
      - P(logZ | z_form): truncated Normal centred on the mean metallicity
        at z_form with sigma = 0.5 dex  (Andrews+2021 Eqs. 6-8)
    z_merger_predicted is computed from COSMIC's t_delay + z_form using
    the Planck15 lookback-time relation.
    Requires a '_zform' config variant.

Usage
-----
    # 2D (default):
    python run_backpop.py \\
        --samples_path /path/to/GW150914.h5 \\
        --event_name GW150914 \\
        --config_name lucky_strikes_fixed_vk1

    # 3D with cosmological prior:
    python run_backpop.py \\
        --samples_path /path/to/GW190814.h5 \\
        --event_name GW190814 \\
        --config_name lucky_strikes_zform \\
        --use_redshift_likelihood True \\
        --nlive 3000 --neff 30000

Output (./results/<event_name>/<config_name>/):
    points.npy     — posterior samples (N_eff, N_dim)
    log_w.npy      — log importance weights
    log_l.npy      — log likelihoods
    log_z.npy      — log evidence (scalar); used in hierarchical step
    blobs.npy      — COSMIC bpp + kick tracks per sample
    metadata.npz   — self-describing run metadata for hierarchical step

Support gates
-------------
By default ``--support_gate none`` leaves KDE tails to the KDE rather than
applying asymmetric ad hoc penalties.  ``hard`` and ``soft`` are available for
explicit HDI-based support truncation/penalization with ``--support_hdi``.
"""

from __future__ import annotations

import os
import time
import numpy as np
from argparse import ArgumentParser
from functools import partial

from nautilus import Prior, Sampler

from backpop import (
    get_backpop_config,
    load_gw_data,
    evolv2,
    str_to_bool,
    BPP_SHAPE,
    KICK_SHAPE,
    COLS_KEEP,
)
from cosmo_prior import (
    log_prior_z_form,
    log_prior_logZ_given_z_on_support,
    z_merger_from_t_delay,
)
from metadata_utils import base_runtime_metadata, get_package_versions, save_metadata


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _nan_blobs() -> tuple[np.ndarray, np.ndarray]:
    """Sentinel NaN blobs for failed COSMIC calls."""
    return (
        np.full(np.prod(BPP_SHAPE),  np.nan, dtype=float),
        np.full(np.prod(KICK_SHAPE), np.nan, dtype=float),
    )


def _flatten_blobs(bpp_raw, kick_raw) -> tuple[np.ndarray, np.ndarray] | None:
    """Flatten and validate COSMIC output arrays for Nautilus blob storage."""
    try:
        bpp_flat  = np.asarray(bpp_raw,  dtype=float).ravel()
        kick_flat = np.asarray(kick_raw, dtype=float).ravel()
    except (TypeError, ValueError):
        return None
    if bpp_flat.size != np.prod(BPP_SHAPE) or kick_flat.size != np.prod(KICK_SHAPE):
        return None
    return bpp_flat, kick_flat


def _merger_observables(final_state) -> tuple[float, float, float]:
    """Compute (m1, m2, mc, q) from COSMIC final state with m1 >= m2."""
    m1 = float(final_state['mass_1'])
    m2 = float(final_state['mass_2'])
    if m1 < m2:
        m1, m2 = m2, m1
    q  = m2 / m1
    mc = (m1 * m2)**(3.0/5.0) / (m1 + m2)**(1.0/5.0)
    return m1, m2, mc, q


_VALID_SUPPORT_GATES = ("none", "hard", "soft")


def _support_gate_penalty(
    value: float,
    bounds: tuple[float, float] | None,
    policy: str,
    *,
    soft_scale_fraction: float = 0.01,
    soft_offset: float = 100.0,
) -> float | None:
    """Return a support-gate log penalty, or None when KDE should be used.

    Policies are symmetric around the supplied HDI bounds:
      - ``none``: never gate; leave all tail behavior to the KDE.
      - ``hard``: return ``-np.inf`` outside the bounds.
      - ``soft``: return a quadratic penalty below/above the bounds.
    """
    if policy not in _VALID_SUPPORT_GATES:
        raise ValueError(
            f"Unsupported support gate '{policy}'. "
            f"Expected one of {_VALID_SUPPORT_GATES}."
        )
    if policy == "none" or bounds is None:
        return None

    lo, hi = map(float, bounds)
    if lo <= value <= hi:
        return None
    if policy == "hard":
        return -np.inf

    width = max(hi - lo, np.finfo(float).eps)
    scale = soft_scale_fraction * width
    distance = lo - value if value < lo else value - hi
    return -0.5 * (distance / scale)**2 - soft_offset


def _combined_support_gate_penalty(
    values: dict[str, float],
    bounds_by_name: dict[str, tuple[float, float] | None],
    policy: str,
) -> float | None:
    """Combine support penalties for all provided observables."""
    penalties = [
        _support_gate_penalty(values[name], bounds_by_name.get(name), policy)
        for name in values
    ]
    penalties = [penalty for penalty in penalties if penalty is not None]
    if not penalties:
        return None
    if any(np.isneginf(penalty) for penalty in penalties):
        return -np.inf
    return float(np.sum(penalties))


def _extract_t_delay_myr(bpp_raw: np.ndarray) -> float | None:
    """Extract COSMIC delay time (tphys at BBH merger row) in Myr."""
    import pandas as pd
    bpp = pd.DataFrame(bpp_raw, columns=COLS_KEEP)
    merger = bpp.loc[
        (bpp.kstar_1 == 14) & (bpp.kstar_2 == 14) & (bpp.evol_type == 3)
    ]
    if len(merger) == 0:
        return None
    return float(merger['tphys'].iloc[0])


# ---------------------------------------------------------------------------
# 2D likelihood: KDE over (mc, q) — Lucky Strikes mode
# ---------------------------------------------------------------------------

def likelihood_2d(
    params_in: dict,
    *,
    kde,
    q_bounds: tuple[float, float],
    mc_bounds: tuple[float, float],
    support_gate: str,
    fixed_params: dict,
    logZ_support: tuple[float, float],
) -> tuple[float, np.ndarray, np.ndarray]:
    """Log-likelihood for the 2D (mc, q) KDE mode.

    Calling convention: this function is pre-bound via functools.partial
    and passed to Nautilus as a single-argument callable lhood(params_in).
    The extra args (kde, q_bounds, fixed_params) are bound at construction
    time as keyword arguments — Nautilus never sees them.  Do NOT pass this
    function via Nautilus's likelihood_args= mechanism (which uses positional
    partial and would reverse the argument order).

    logZ has a flat Nautilus prior.  No cosmological constraints.
    This is the mode used in Lucky Strikes (Magana Hernandez & Breivik 2025).

    Parameters
    ----------
    params_in : dict
        Nautilus sample — the sole positional argument Nautilus supplies.
    kde : scipy.stats.gaussian_kde
        2D KDE over (mc_src, q_src).  Bound via partial.
    q_bounds : tuple[float, float]
        (q_lo, q_hi) HDI used by the configured support gate.  Bound via partial.
    mc_bounds : tuple[float, float]
        (mc_lo, mc_hi) HDI used by the configured support gate.  Bound via partial.
    support_gate : {"none", "hard", "soft"}
        Support policy applied consistently to KDE observables.
    fixed_params : dict
        Fixed COSMIC parameters.  Bound via partial.
    logZ_support : tuple[float, float]
        Active BackPop configuration support for log10 metallicity.

    Returns
    -------
    log_prob, bpp_flat, kick_flat
    """
    nan_bpp, nan_kick = _nan_blobs()

    final_state, bpp_raw, kick_raw = evolv2(params_in, ['mass_1', 'mass_2'], fixed_params)
    if final_state is None:
        return -np.inf, nan_bpp, nan_kick

    blobs = _flatten_blobs(bpp_raw, kick_raw)
    if blobs is None:
        return -np.inf, nan_bpp, nan_kick
    bpp_flat, kick_flat = blobs

    _, _, mc, q = _merger_observables(final_state)

    penalty = _combined_support_gate_penalty(
        {"mc": mc, "q": q}, {"mc": mc_bounds, "q": q_bounds}, support_gate
    )
    if penalty is not None:
        return penalty, bpp_flat, kick_flat

    log_prob = float(kde.logpdf(np.array([[mc], [q]]))[0])
    return log_prob, bpp_flat, kick_flat


# ---------------------------------------------------------------------------
# 3D likelihood: KDE over (mc, q, z_merger) + cosmological priors
# ---------------------------------------------------------------------------

def likelihood_3d(
    params_in: dict,
    *,
    kde,
    q_bounds: tuple[float, float],
    mc_bounds: tuple[float, float],
    z_bounds: tuple[float, float],
    support_gate: str,
    fixed_params: dict,
    logZ_support: tuple[float, float],
) -> tuple[float, np.ndarray, np.ndarray]:
    """Log-likelihood for the 3D (mc, q, z_merger) KDE mode.

    Calling convention: pre-bound via functools.partial, passed to Nautilus
    as a single-argument callable lhood(params_in).  Do NOT use Nautilus's
    likelihood_args= mechanism — it applies positional partial which reverses
    the argument order.

    Total log-probability:
        log p = log L_KDE(mc, q, z_merger_pred)
              + log P(z_form)          [Andrews+2021 Eq. 5 - SFR prior]
              + log P(logZ | z_form)   [Andrews+2021 Eq. 8 - metallicity prior]

    Notes on the factorisation:
      - z_form is sampled with a flat Nautilus prior over [1e-4, 20]; the
        physical P(z_form) and P(logZ|z_form) terms are evaluated here.
      - z_form is NOT passed to COSMIC; it enters only through the
        delay-time → z_merger mapping and the metallicity prior.
      - Cosmological priors evaluated before COSMIC for fast rejection.

    Parameters
    ----------
    params_in : dict
        Nautilus sample — must include 'z_form' and 'logZ'.
    kde : scipy.stats.gaussian_kde
        3D KDE over (mc_src, q_src, z_src).  Bound via partial.
    q_bounds : tuple[float, float]
        (q_lo, q_hi) HDI used by the configured support gate.  Bound via partial.
    mc_bounds : tuple[float, float]
        (mc_lo, mc_hi) HDI used by the configured support gate.  Bound via partial.
    z_bounds : tuple[float, float]
        (z_lo, z_hi) HDI used by the configured support gate.  Bound via partial.
    support_gate : {"none", "hard", "soft"}
        Support policy applied consistently to KDE observables.
    fixed_params : dict
        Fixed COSMIC parameters.  Bound via partial.
    logZ_support : tuple[float, float]
        Active BackPop configuration support for log10 metallicity.

    Returns
    -------
    log_prob, bpp_flat, kick_flat
    """
    nan_bpp, nan_kick = _nan_blobs()

    logZ_lo, logZ_hi = map(float, logZ_support)
    if not logZ_lo < logZ_hi:
        raise ValueError("logZ_support must satisfy lo < hi")

    z_form  = params_in.get('z_form')
    log10_Z = params_in.get('logZ')

    if z_form is None or log10_Z is None:
        raise KeyError(
            "likelihood_3d requires 'z_form' and 'logZ' in params_in. "
            "Use a '_zform' config variant."
        )

    # ---- Cosmological priors (fast rejection before COSMIC) ----
    lp_z = log_prior_z_form(z_form)
    if not np.isfinite(lp_z):
        return -np.inf, nan_bpp, nan_kick

    lp_logZ = log_prior_logZ_given_z_on_support(log10_Z, z_form, logZ_lo, logZ_hi)
    if not np.isfinite(lp_logZ):
        return -np.inf, nan_bpp, nan_kick

    # ---- COSMIC call (z_form stripped — not a COSMIC parameter) ----
    cosmic_params = {k: v for k, v in params_in.items() if k != 'z_form'}
    final_state, bpp_raw, kick_raw = evolv2(
        cosmic_params, ['mass_1', 'mass_2'], fixed_params
    )
    if final_state is None:
        return -np.inf, nan_bpp, nan_kick

    blobs = _flatten_blobs(bpp_raw, kick_raw)
    if blobs is None:
        return -np.inf, nan_bpp, nan_kick
    bpp_flat, kick_flat = blobs

    # ---- Delay time ----
    t_delay = _extract_t_delay_myr(bpp_raw)
    if t_delay is None:
        return -np.inf, nan_bpp, nan_kick

    # ---- Predicted merger redshift ----
    z_merger_pred = z_merger_from_t_delay(z_form, t_delay)
    if z_merger_pred is None:
        # Merges in the future or before formation
        return -np.inf, nan_bpp, nan_kick

    # ---- Merger observables ----
    _, _, mc, q = _merger_observables(final_state)

    # ---- Configured support policy (mc, q, z) ----
    penalty = _combined_support_gate_penalty(
        {"mc": mc, "q": q, "z": z_merger_pred},
        {"mc": mc_bounds, "q": q_bounds, "z": z_bounds},
        support_gate,
    )
    if penalty is not None:
        return penalty + lp_z + lp_logZ, bpp_flat, kick_flat

    # ---- 3D KDE likelihood ----
    coord  = np.array([[mc], [q], [z_merger_pred]])
    log_ll = float(kde.logpdf(coord)[0])

    log_prob = log_ll + lp_z + lp_logZ
    return log_prob, bpp_flat, kick_flat


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_VALID_CONFIGS_2D = [
    "lucky_strikes",
    "lucky_strikes_fixed_vk1",
    "bbh_no_kicks",
]
_VALID_CONFIGS_3D = [
    "lucky_strikes_zform",
    "lucky_strikes_fixed_vk1_zform",
    "bbh_no_kicks_zform",
]


def parse_args():
    p = ArgumentParser(description="Single-event BackPop inference (Nautilus).")

    p.add_argument("--samples_path", required=True,
                   help="Path to pesummary HDF5 posterior file.")
    p.add_argument("--event_name",   required=True,
                   help="Event identifier (e.g. GW150914, GW190814).")
    p.add_argument("--config_name",  required=True,
                   choices=_VALID_CONFIGS_2D + _VALID_CONFIGS_3D,
                   help="Parameter space config.  "
                        "Use '*_zform' variants with --use_redshift_likelihood True.")
    p.add_argument("--use_redshift_likelihood", type=str_to_bool, default=False,
                   help="3D mode: include z_merger in KDE and apply Andrews+2021 "
                        "cosmological priors P(z_form) and P(logZ|z_form).  "
                        "Requires a '_zform' config.")
    p.add_argument("--approximant",  default="C01:Mixed",
                   help="Posterior approximant label in the pesummary file.")
    p.add_argument("--use_pe_weights", type=str_to_bool, default=True,
                   help="Apply Callister (2021) Jacobian reweighting to PE samples.")
    p.add_argument("--mass_frame", choices=["auto", "source", "detector"], default="auto",
                   help="Interpret posterior mass columns as source-frame, detector-frame, or auto-detect aliases.")
    p.add_argument("--nlive",  type=int, default=3000,
                   help="Nautilus n_live.  3000 for hard events; 1000 for typical BBH.")
    p.add_argument("--neff",   type=int, default=30000,
                   help="Nautilus target effective sample size.")
    p.add_argument("--support_gate", choices=_VALID_SUPPORT_GATES, default="none",
                   help="Support treatment for KDE observables outside HDI bounds. "
                        "Default 'none' is safer: it leaves tails to the KDE "
                        "and avoids one-off asymmetric likelihood penalties.")
    p.add_argument("--support_hdi", type=float, default=0.999,
                   help="HDI probability used to define support bounds for "
                        "--support_gate hard/soft (default: 0.999).")
    p.add_argument("--resume", type=str_to_bool, default=False,
                   help="Resume from existing Nautilus checkpoint.")
    p.add_argument("--max_threads", type=int, default=None,
                   help="Worker thread cap (default: min(2*ncores-2, 64)).")

    return p.parse_args()


def main():
    start = time.time()
    opts  = parse_args()

    if not (0.0 < opts.support_hdi <= 1.0):
        raise ValueError(
            f"--support_hdi must be in the interval (0, 1]. Got {opts.support_hdi}."
        )

    # ---- Validate config / mode consistency ----
    use_z = opts.use_redshift_likelihood
    if use_z and opts.config_name not in _VALID_CONFIGS_3D:
        raise ValueError(
            f"--use_redshift_likelihood True requires a '_zform' config. "
            f"Got '{opts.config_name}'. Valid 3D configs: {_VALID_CONFIGS_3D}"
        )
    if not use_z and opts.config_name in _VALID_CONFIGS_3D:
        raise ValueError(
            f"Config '{opts.config_name}' requires --use_redshift_likelihood True. "
            f"Valid 2D configs: {_VALID_CONFIGS_2D}"
        )

    # ---- Output directory ----
    output_path = os.path.join("results", opts.event_name, opts.config_name)
    os.makedirs(output_path, exist_ok=True)
    checkpoint  = os.path.join(output_path, "checkpoint.hdf5")

    mode_tag = "3D+cosmo" if use_z else "2D"
    print(f"[run_backpop] Event:  {opts.event_name}")
    print(f"[run_backpop] Config: {opts.config_name}  [{mode_tag}]")
    print(f"[run_backpop] Output: {output_path}")

    # ---- Load GW data ----
    kde, q_bounds, mc_bounds, z_bounds, raw_samples, gw_metadata = load_gw_data(
        opts.samples_path,
        approximant=opts.approximant,
        use_pe_weights=opts.use_pe_weights,
        include_redshift=use_z,
        mass_frame=opts.mass_frame,
        return_metadata=True,
        hdi_prob=opts.support_hdi,
    )

    # ---- Build prior ----
    lower_bound, upper_bound, params_in_names, fixed_params = get_backpop_config(
        opts.config_name
    )
    prior = Prior()
    for name, lo, hi in zip(params_in_names, lower_bound, upper_bound):
        prior.add_parameter(name, dist=(lo, hi))

    print(f"[run_backpop] Parameter space ({len(params_in_names)}-D):")
    for name, lo, hi in zip(params_in_names, lower_bound, upper_bound):
        print(f"  {name:<16s}  [{lo:.4g}, {hi:.4g}]")
    if fixed_params:
        print(f"[run_backpop] Fixed: {fixed_params}")

    # ---- Thread count ----
    n_cores   = len(os.sched_getaffinity(0))
    n_threads = min(2 * n_cores - 2, 64)
    if opts.max_threads is not None:
        n_threads = min(n_threads, opts.max_threads)
    n_threads = max(n_threads, 1)
    print(f"[run_backpop] Threads: {n_threads} / {n_cores} cores")

    # ---- Blob dtype ----
    blob_dtype = [
        ('bpp',       float, (np.prod(BPP_SHAPE),)),
        ('kick_info', float, (np.prod(KICK_SHAPE),)),
    ]

    # ---- Bind likelihood with partial (Nautilus-version-safe) ----
    # Using functools.partial rather than likelihood_args= avoids a version-
    # dependent calling convention in Nautilus's multiprocessing pool where
    # the params_dict and extra args can be unpacked in the wrong order.
    # partial binds all extra args at construction time; Nautilus sees a
    # single-argument callable: lhood(params_dict).  scipy KDE and plain
    # dicts are both picklable so this works across worker processes.
    if use_z:
        lhood = partial(
            likelihood_3d,
            kde=kde,
            q_bounds=q_bounds,
            mc_bounds=mc_bounds,
            z_bounds=z_bounds,
            support_gate=opts.support_gate,
            fixed_params=fixed_params,
            logZ_support=(
                float(lower_bound[params_in_names.index("logZ")]),
                float(upper_bound[params_in_names.index("logZ")]),
            ),
        )
    else:
        lhood = partial(
            likelihood_2d,
            kde=kde,
            q_bounds=q_bounds,
            mc_bounds=mc_bounds,
            support_gate=opts.support_gate,
            fixed_params=fixed_params,
            logZ_support=(
                float(lower_bound[params_in_names.index("logZ")]),
                float(upper_bound[params_in_names.index("logZ")]),
            ),
        )

    # ---- Nautilus sampler ----
    sampler = Sampler(
        prior=prior,
        likelihood=lhood,
        n_live=opts.nlive,
        pool=n_threads,
        blobs_dtype=blob_dtype,
        filepath=checkpoint,
        resume=opts.resume,
    )

    sampler.run(n_eff=opts.neff, verbose=True, discard_exploration=True)

    # ---- Extract posterior ----
    points, log_w, log_l, blobs = sampler.posterior(return_blobs=True)
    log_z   = sampler.log_z
    weights = np.exp(log_w - log_z)
    weights /= weights.sum()
    n_eff_actual = int(1.0 / np.sum(weights**2))

    print(f"[run_backpop] log Z        = {log_z:.3f}")
    print(f"[run_backpop] N_eff actual = {n_eff_actual}")

    # ---- Save ----
    np.save(os.path.join(output_path, "points.npy"), points)
    np.save(os.path.join(output_path, "log_w.npy"),  log_w)
    np.save(os.path.join(output_path, "log_l.npy"),  log_l)
    np.save(os.path.join(output_path, "log_z.npy"),  np.array([log_z]))
    np.save(os.path.join(output_path, "blobs.npy"),  blobs)

    # ---- Delete checkpoint (recovers ~1-2 GB per event) ----
    # The checkpoint HDF5 is only needed for --resume. Once all outputs are
    # saved successfully it is redundant. Delete immediately to avoid disk
    # quota exhaustion when many events run in parallel.
    if os.path.exists(checkpoint):
        os.remove(checkpoint)
        print(f"[run_backpop] Checkpoint deleted: {checkpoint}")

    metadata = dict(
        **base_runtime_metadata("."),
        package_versions=get_package_versions(["numpy", "scipy", "astropy", "nautilus", "pesummary", "cosmic"]),
        event_name              = opts.event_name,
        config_name             = opts.config_name,
        likelihood_mode         = mode_tag,
        uses_z_form            = bool(use_z),
        uses_sfr_prior         = bool(use_z),
        uses_logZ_given_z_prior = bool(use_z),
        use_redshift_likelihood = use_z,
        params_in               = params_in_names,
        lower_bound             = lower_bound,
        upper_bound             = upper_bound,
        fixed_params_keys       = list(fixed_params.keys()),
        fixed_params_values     = list(fixed_params.values()),
        nlive                   = opts.nlive,
        neff                    = opts.neff,
        n_eff_actual            = n_eff_actual,
        log_z                   = log_z,
        support_gate            = opts.support_gate,
        support_hdi             = opts.support_hdi,
        q_bounds                = q_bounds,
        mc_bounds               = mc_bounds,
        z_bounds                = (
            z_bounds if z_bounds is not None else np.array([np.nan, np.nan])
        ),
        mass_frame_used         = gw_metadata["mass_frame_used"],
        pe_prior_weighting_used = gw_metadata["pe_prior_weighting_used"],
        mass_column_names       = gw_metadata["mass_column_names"],
        wall_time_s             = time.time() - start,
        samples_path            = opts.samples_path,
        pe_approximant          = opts.approximant,
        mass_frame_requested    = opts.mass_frame,
        mass_frame_interpretation = f"requested={opts.mass_frame}; used={gw_metadata['mass_frame_used']}",
        pe_prior_weighting_setting = bool(opts.use_pe_weights),
        support_gate_settings   = dict(policy=opts.support_gate, hdi=opts.support_hdi, q_bounds=q_bounds, mc_bounds=mc_bounds, z_bounds=z_bounds if z_bounds is not None else [None, None]),
    )
    save_metadata(output_path, metadata)

    elapsed = time.time() - start
    print(f"[run_backpop] Done.  Wall time: {elapsed/3600:.2f} hr ({elapsed:.0f} s)")


if __name__ == "__main__":
    main()
