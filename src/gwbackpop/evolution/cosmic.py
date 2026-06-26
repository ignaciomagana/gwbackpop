"""
backpop.py
----------
Core library for the BackPop framework: mapping individual gravitational-wave
merger observations back to their binary progenitor initial conditions and
binary evolution physics.

Based on:
  - Andrews et al. 2021 (ApJL 914, L32)   — single-event, fixed hyperparameters
  - Wong et al. 2023 (ApJ 950, 181)        — joint ZAMS + hyperparameter inference
  - Magana Hernandez & Breivik 2025        — Lucky Strikes (GW190814, full kicks)

Changes from the Lucky Strikes version
---------------------------------------
BUGFIXES
  1. Alpha naming mismatch: params_in used 'alpha_1'/'alpha_2' but set_flags
     checked for 'alpha1_1'/'alpha1_2' — CE efficiency was silently stuck at
     the default [1.0, 1.0] regardless of sampled values. Fixed by unifying
     on 'alpha_1'/'alpha_2' throughout.

  2. Kick handler read from wrong dict: inside the params_in loop the kick
     sub-handler was reading fixed_params[param] instead of params_in[param],
     which would KeyError for any sampled-kick config with empty fixed_params.
     Rewrote kick handling to use explicit dispatch instead of string matching.

  3. Kick string-matching fragility: "1" in param matched 'alpha_1', 'acc_lim_1'
     etc. before the list-guard fired — worked only by accident. Replaced with
     a dedicated _parse_kick_params() that uses structured dispatch.

REFACTORING / CLARITY
  4. Removed all commented-out legacy emcee wrapper functions (evolv2_fixed_kicks,
     evolv2_lowmass_secondary, etc.) — these are dead code since the switch to
     Nautilus and the dict-based API.

  5. Prior bounds are now defined inside get_backpop_config() rather than as
     module-level globals — easier to reason about per-config.

  6. Added a canonical 'lucky_strikes' config (17-parameter, full kicks,
     independent alpha/flim per RLOF event) and a 'lucky_strikes_fixed_vk1'
     variant (13-parameter, first-kick fixed to zero). These are the configs
     intended for the catalog-level hierarchical run.

  7. Cosmological interpolation tables are computed once at module import and
     cached as module-level constants — unchanged from original.

  8. Added type annotations and extended docstrings throughout.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from astropy.cosmology import Planck15
from astropy import constants
from cosmic import _evolvebin


_SOURCE_MASS_ALIASES = (
    ("mass_1_source", "mass_2_source"),
    ("mass1_source", "mass2_source"),
    ("m1_source", "m2_source"),
    ("mass_1_source_frame", "mass_2_source_frame"),
    ("mass_1_sourceframe", "mass_2_sourceframe"),
)
_DETECTOR_MASS_ALIASES = (
    ("mass_1_detector", "mass_2_detector"),
    ("mass1_detector", "mass2_detector"),
    ("m1_detector", "m2_detector"),
    ("mass_1_det", "mass_2_det"),
    ("mass1_det", "mass2_det"),
    ("m1_det", "m2_det"),
    ("mass_1_detector_frame", "mass_2_detector_frame"),
    ("mass_1_detectorframe", "mass_2_detectorframe"),
)
_AMBIGUOUS_MASS_COLUMNS = ("mass_1", "mass_2")


def _find_column_pair(samples, aliases):
    keys = set(samples.keys())
    for m1_name, m2_name in aliases:
        if m1_name in keys and m2_name in keys:
            return m1_name, m2_name
    return None


def resolve_mass_columns(samples, mass_frame: str = "auto") -> dict:
    """Resolve posterior mass columns and whether they are source/detector frame."""
    if mass_frame not in {"auto", "source", "detector"}:
        raise ValueError("mass_frame must be one of 'auto', 'source', or 'detector'.")

    source_pair = _find_column_pair(samples, _SOURCE_MASS_ALIASES)
    detector_pair = _find_column_pair(samples, _DETECTOR_MASS_ALIASES)
    ambiguous_pair = (
        _AMBIGUOUS_MASS_COLUMNS
        if all(name in samples for name in _AMBIGUOUS_MASS_COLUMNS)
        else None
    )

    warning = None
    if mass_frame == "source":
        pair = source_pair or (ambiguous_pair if detector_pair is None else None)
        if pair is None:
            raise KeyError("--mass_frame source requested, but no source-frame mass columns were found.")
        frame = "source"
        if pair == ambiguous_pair:
            warning = "Ambiguous mass_1/mass_2 columns interpreted as source-frame because --mass_frame source was requested."
    elif mass_frame == "detector":
        pair = detector_pair or (ambiguous_pair if source_pair is None else None)
        if pair is None:
            raise KeyError("--mass_frame detector requested, but no detector-frame mass columns were found.")
        frame = "detector"
        if pair == ambiguous_pair:
            warning = "Ambiguous mass_1/mass_2 columns interpreted as detector-frame because --mass_frame detector was requested."
    elif source_pair is not None:
        pair = source_pair
        frame = "source"
    elif detector_pair is not None:
        pair = detector_pair
        frame = "detector"
    elif ambiguous_pair is not None:
        pair = ambiguous_pair
        frame = "detector"
        warning = (
            "Posterior samples only contain ambiguous mass_1/mass_2 columns; "
            "assuming detector-frame masses by repository convention. Pass "
            "--mass_frame source or --mass_frame detector to make this explicit."
        )
    else:
        raise KeyError("Could not find a recognized pair of mass columns in posterior samples.")

    if warning is not None:
        warnings.warn(f"[load_gw_data] {warning}", UserWarning, stacklevel=2)

    return {"frame": frame, "columns": pair, "warning": warning}


# ---------------------------------------------------------------------------
# COSMIC column definitions
# ---------------------------------------------------------------------------

ALL_COLUMNS = [
    'tphys', 'mass_1', 'mass_2', 'kstar_1', 'kstar_2', 'sep', 'porb',
    'ecc', 'RRLO_1', 'RRLO_2', 'evol_type', 'aj_1', 'aj_2', 'tms_1',
    'tms_2', 'massc_1', 'massc_2', 'rad_1', 'rad_2', 'mass0_1',
    'mass0_2', 'lum_1', 'lum_2', 'teff_1', 'teff_2', 'radc_1',
    'radc_2', 'menv_1', 'menv_2', 'renv_1', 'renv_2', 'omega_spin_1',
    'omega_spin_2', 'B_1', 'B_2', 'bacc_1', 'bacc_2', 'tacc_1',
    'tacc_2', 'epoch_1', 'epoch_2', 'bhspin_1', 'bhspin_2',
    'deltam_1', 'deltam_2', 'SN_1', 'SN_2', 'bin_state', 'merger_type',
]

INTEGER_COLUMNS = [
    "bin_state", "bin_num", "kstar_1", "kstar_2", "SN_1", "SN_2", "evol_type",
]

# Columns written into the bpp output array (full binary evolution track)
BPP_COLUMNS = [
    'tphys', 'mass_1', 'mass_2', 'kstar_1', 'kstar_2',
    'sep', 'porb', 'ecc', 'RRLO_1', 'RRLO_2', 'evol_type',
    'aj_1', 'aj_2', 'tms_1', 'tms_2',
    'massc_1', 'massc_2', 'rad_1', 'rad_2',
    'mass0_1', 'mass0_2', 'lum_1', 'lum_2', 'teff_1', 'teff_2',
    'radc_1', 'radc_2', 'menv_1', 'menv_2', 'renv_1', 'renv_2',
    'omega_spin_1', 'omega_spin_2', 'B_1', 'B_2', 'bacc_1', 'bacc_2',
    'tacc_1', 'tacc_2', 'epoch_1', 'epoch_2',
    'bhspin_1', 'bhspin_2',
]

# Subset actually retained for blobs storage (keeps memory manageable)
COLS_KEEP = [
    'tphys', 'mass_1', 'mass_2', 'massc_1', 'massc_2',
    'menv_1', 'menv_2', 'kstar_1', 'kstar_2',
    'porb', 'ecc', 'evol_type', 'rad_1', 'rad_2', 'lum_1', 'lum_2',
]

BCM_COLUMNS = [
    'tphys', 'kstar_1', 'mass0_1', 'mass_1', 'lum_1', 'rad_1',
    'teff_1', 'massc_1', 'radc_1', 'menv_1', 'renv_1', 'epoch_1',
    'omega_spin_1', 'deltam_1', 'RRLO_1', 'kstar_2', 'mass0_2', 'mass_2',
    'lum_2', 'rad_2', 'teff_2', 'massc_2', 'radc_2', 'menv_2',
    'renv_2', 'epoch_2', 'omega_spin_2', 'deltam_2', 'RRLO_2',
    'porb', 'sep', 'ecc', 'B_1', 'B_2',
    'SN_1', 'SN_2', 'bin_state', 'merger_type',
]

KICK_COLUMNS = [
    'star', 'disrupted', 'natal_kick', 'phi', 'theta', 'mean_anomaly',
    'delta_vsysx_1', 'delta_vsysy_1', 'delta_vsysz_1', 'vsys_1_total',
    'delta_vsysx_2', 'delta_vsysy_2', 'delta_vsysz_2', 'vsys_2_total',
    'theta_euler', 'phi_euler', 'psi_euler', 'randomseed',
]

# Fixed array shapes for Nautilus blob storage
BPP_SHAPE = (25, len(COLS_KEEP))
KICK_SHAPE = (2, len(KICK_COLUMNS))


# ---------------------------------------------------------------------------
# Cosmological lookup tables (computed once at import)
# ---------------------------------------------------------------------------

_Z_MAX = 100
_zgrid = np.expm1(np.linspace(np.log(1), np.log(_Z_MAX + 1), 10_000))
_dL_grid = Planck15.luminosity_distance(_zgrid).to('Mpc').value
_t_grid = Planck15.lookback_time(_zgrid).to('Myr').value

# t_universe - lookback_time  →  cosmic time at formation
tofdL  = interp1d(_dL_grid, 13700 - _t_grid, bounds_error=False, fill_value=1e100)
dLoft  = interp1d(13700 - _t_grid, _dL_grid,  bounds_error=False, fill_value=1e100)
ddLdt  = interp1d(_t_grid, np.gradient(_dL_grid, _t_grid), bounds_error=False, fill_value=1e100)
zofdL  = interp1d(_dL_grid, _zgrid)
dLofz  = interp1d(_zgrid, _dL_grid,  bounds_error=False, fill_value=1e100)
ddLdz  = interp1d(_zgrid, np.gradient(_dL_grid, _zgrid), bounds_error=False, fill_value=1e100)
zoft   = interp1d(13700 - _t_grid, _zgrid, bounds_error=False, fill_value=1000)
tofz   = interp1d(_zgrid, 13700 - _t_grid, bounds_error=False, fill_value=1e100)
dtdz   = interp1d(_zgrid, np.gradient(13700 - _t_grid, _zgrid), bounds_error=False, fill_value=1e100)


# ---------------------------------------------------------------------------
# Prior / config definitions
# ---------------------------------------------------------------------------

from gwbackpop.config import get_backpop_config


# Human-readable labels for corner plots (LaTeX)
PARAM_LABELS = {
    'm1':      r'$m_1\ [M_\odot]$',
    'q':       r'$q_\mathrm{ZAMS}$',
    'logtb':   r'$\log_{10}(t_b/\mathrm{day})$',
    'logZ':    r'$\log_{10}(Z/Z_\odot)$',
    'alpha_1': r'$\alpha_1$',
    'alpha_2': r'$\alpha_2$',
    'flim_1':  r'$f_\mathrm{lim,1}$',
    'flim_2':  r'$f_\mathrm{lim,2}$',
    'vk1':     r'$v_{k,1}\ [\mathrm{km\,s^{-1}}]$',
    'theta1':  r'$\theta_1\ [\mathrm{deg}]$',
    'phi1':    r'$\phi_1\ [\mathrm{deg}]$',
    'omega1':  r'$\omega_1\ [\mathrm{deg}]$',
    'vk2':     r'$v_{k,2}\ [\mathrm{km\,s^{-1}}]$',
    'theta2':  r'$\theta_2\ [\mathrm{deg}]$',
    'phi2':    r'$\phi_2\ [\mathrm{deg}]$',
    'omega2':  r'$\omega_2\ [\mathrm{deg}]$',
    'z_form':  r'$z_\mathrm{form}$',
}


# ---------------------------------------------------------------------------
# COSMIC flags
# ---------------------------------------------------------------------------

def _parse_kick_params(
    params_in: dict,
    fixed_params: dict,
) -> np.ndarray:
    """Build the (2, 5) natal_kick_array expected by COSMIC's _evolvebin.

    COSMIC natal_kick_array layout (per star):
        [0] vk        — kick speed [km/s]
        [1] phi       — co-lateral polar angle [deg], valid [-90, 90]
        [2] theta     — azimuthal kick angle [deg], valid [0, 360]
        [3] omega     — mean anomaly at kick [deg]
        [4] reserved  — set to 0.0

    Parameters come from either the sampled dict (params_in) or the
    fixed dict (fixed_params).  params_in takes precedence.

    Note: COSMIC uses -100 as a sentinel meaning "draw from the population
    kick model".  We always supply explicit values, so -100 never appears.
    """
    # Merge: sampled params override fixed params
    all_params = {**fixed_params, **params_in}

    natal_kick = np.zeros((2, 5))

    # Star 1
    natal_kick[0, 0] = all_params.get('vk1',    0.0)
    natal_kick[0, 1] = all_params.get('phi1',   0.0)
    natal_kick[0, 2] = all_params.get('theta1', 0.0)
    natal_kick[0, 3] = all_params.get('omega1', 0.0)
    natal_kick[0, 4] = 0.0

    # Star 2
    natal_kick[1, 0] = all_params.get('vk2',    0.0)
    natal_kick[1, 1] = all_params.get('phi2',   0.0)
    natal_kick[1, 2] = all_params.get('theta2', 0.0)
    natal_kick[1, 3] = all_params.get('omega2', 0.0)
    natal_kick[1, 4] = 0.0

    return natal_kick


def set_flags(params_in: dict, fixed_params: dict) -> dict:
    """Build the full COSMIC flags dict from sampled and fixed parameters.

    Defaults follow Lucky Strikes (Magana Hernandez & Breivik 2025) §2:
      - Fryer et al. (2012) delayed remnant mass prescription (remnantflag=4)
      - Claeys et al. (2014) lambda_CE fits (lambdaf=0.0 → internal calculation)
      - Wagg et al. (2025) kick model (kickflag=1 → Pfahl+2002 based)
      - Vink et al. (2001, 2005) stellar wind mass loss (windflag=3)
      - PISN = -2 (pair-instability prescription active)
      - qHG = 3.0 fixed (not a free parameter in Lucky Strikes)
      - mxns = 1.0 M_sun (artificial NS mass ceiling to simplify sampling)

    Parameters
    ----------
    params_in : dict
        Sampled parameters from Nautilus (vary each call).
    fixed_params : dict
        Parameters held constant for this run configuration.

    Returns
    -------
    flags : dict
        Complete flags dict ready for set_evolvebin_flags().
    """
    flags = {}

    # ---- Wind / mass transfer defaults ----
    flags["neta"]         = 0.5
    flags["bwind"]        = 0.0
    flags["hewind"]       = 0.5
    flags["beta"]         = -1       # -1 → Hurley+2002 prescription
    flags["xi"]           = 0.5
    flags["acc2"]         = 1.5
    flags["epsnov"]       = 0.001
    flags["eddfac"]       = 1.0
    flags["gamma"]        = -2
    flags["windflag"]     = 3        # Vink+2001/2005 winds
    flags["don_lim"]      = -1
    flags["eddlimflag"]   = 0

    # ---- CE defaults ----
    # alpha1 is a 2-element array: [alpha_for_star1_CE, alpha_for_star2_CE]
    # Populated below from params_in / fixed_params.
    flags["alpha1"]       = np.array([1.0, 1.0])
    flags["lambdaf"]      = 0.0      # 0.0 → use Claeys+2014 fits internally
    flags["ceflag"]       = 0
    flags["cekickflag"]   = 2
    flags["cemergeflag"]  = 1
    flags["cehestarflag"] = 0

    # ---- Mass transfer stability ----
    # qcrit_array: 16-element array, one entry per stellar type (kstar).
    # Index 2 = Hertzsprung Gap (HG); fixed at 3.0 per Lucky Strikes §2.1.
    qcrit_array          = np.zeros(16)
    qcrit_array[2]       = 3.0      # qHG
    flags["qcrit_array"] = qcrit_array
    flags["qcflag"]      = 5

    # acc_lim: 2-element array — Eddington-factor limit on stable MT accretion.
    # Populated below from flim_1 / flim_2.
    flags["acc_lim"]     = np.array([-1.0, -1.0])  # -1 → no external limit

    # ---- Remnant / SN defaults ----
    flags["remnantflag"]   = 4       # Fryer+2012 delayed prescription
    flags["mxns"]          = 1.0     # artificial NS ceiling [M_sun]
    flags["pisn"]          = -2      # PISN active
    flags["ecsn"]          = 2.5
    flags["ecsn_mlow"]     = 1.6
    flags["aic"]           = 1
    flags["ussn"]          = 1
    flags["sigma"]         = 0       # sigma=0 → kicks set by natal_kick_array
    flags["sigmadiv"]      = -20.0
    flags["bhsigmafrac"]   = 1.0
    flags["polar_kick_angle"] = 90.0
    flags["kickflag"]      = 1       # Pfahl+2002 / Wagg+2025 kick model
    flags["rembar_massloss"] = 0.5
    flags["bhflag"]        = 1
    flags["bhms_coll_flag"] = 1

    # ---- Spins ----
    flags["bhspinflag"]    = 0
    flags["bhspinmag"]     = 0.0

    # ---- Misc ----
    flags["tflag"]         = 1
    flags["ifflag"]        = 0
    flags["wdflag"]        = 1
    flags["rtmsflag"]      = 0
    flags["grflag"]        = 1
    flags["bdecayfac"]     = 1
    flags["bconst"]        = 3000
    flags["ck"]            = 1000
    flags["pts1"]          = 0.05
    flags["pts2"]          = 0.01
    flags["pts3"]          = 0.02
    flags["rejuv_fac"]     = 1.0
    flags["rejuvflag"]     = 0
    flags["htpmb"]         = 1
    flags["ST_cr"]         = 1
    flags["ST_tide"]       = 1
    flags["zsun"]          = 0.014
    flags["fprimc_array"]  = np.zeros(16)
    flags["randomseed"]    = 42

    # ---- Merge sampled + fixed; sampled takes precedence ----
    all_params = {**fixed_params, **params_in}

    # CE efficiency: alpha_1 → first RLOF event, alpha_2 → second RLOF event.
    # BUGFIX: previously checked for 'alpha1_1'/'alpha1_2' which never matched
    # 'alpha_1'/'alpha_2' from params_in, leaving CE efficiency at default 1.0.
    alpha1 = flags["alpha1"].copy()
    if 'alpha_1' in all_params:
        alpha1[0] = all_params['alpha_1']
    if 'alpha_2' in all_params:
        alpha1[1] = all_params['alpha_2']
    flags["alpha1"] = alpha1

    # Stable MT accretion efficiency limit:
    # flim_i in [0,1] is the fraction of donor mass the accretor can accept,
    # expressed as a multiple of the Eddington rate (COSMIC acc_lim convention).
    acc_lim = flags["acc_lim"].copy()
    if 'flim_1' in all_params:
        acc_lim[0] = all_params['flim_1']
    if 'flim_2' in all_params:
        acc_lim[1] = all_params['flim_2']
    flags["acc_lim"] = acc_lim

    # Natal kicks — uses the dedicated parser to avoid string-matching fragility.
    flags["natal_kick_array"] = _parse_kick_params(params_in, fixed_params)

    return flags


def _fortran_assign(fortran_attr, value: np.ndarray | float) -> None:
    """Assign value to a Fortran module attribute, handling scalar/array mismatch.

    COSMIC's Fortran interface (f2py) is strict about array rank. Depending on
    the COSMIC version and build, some variables (notably cevars.alpha1 and
    mtvars.acc_lim) may be declared as scalars in one build and rank-1 arrays
    in another.

    This helper attempts the direct assignment and falls back to a scalar (first
    element) if the Fortran variable is rank-0 (scalar) but value is an array.

    Parameters
    ----------
    fortran_attr : f2py module attribute (writable)
        Target Fortran variable, e.g. _evolvebin.cevars.alpha1.
    value : np.ndarray or float
        Value to assign.
    """
    try:
        fortran_attr = value          # direct assignment (works if ranks match)
        return fortran_attr           # returned but caller must re-assign; see usage note
    except (ValueError, TypeError):
        pass
    # Rank mismatch — fall back to first scalar element
    return float(np.asarray(value).flat[0])


def set_evolvebin_flags(flags: dict) -> None:
    """Write flags dict into the _evolvebin Fortran module global state.

    Must be called immediately before each _evolvebin.evolv2() call since
    the module globals are shared across threads.  Nautilus handles thread
    safety via its worker pool (one process per worker).

    Handles two known COSMIC version differences via runtime rank detection:
      - cevars.alpha1: scalar in some builds, rank-1 array of size 2 in others
      - mtvars.acc_lim: same issue

    When the Fortran variable is scalar but the flag holds a 2-element array
    (separate alpha/flim per CE event), the first element is used and a
    warning is emitted once per process.
    """
    import warnings

    _evolvebin.windvars.neta           = flags["neta"]
    _evolvebin.windvars.bwind          = flags["bwind"]
    _evolvebin.windvars.hewind         = flags["hewind"]

    # ---- alpha1: handle scalar vs 2-element array across COSMIC builds ----
    alpha1_val = flags["alpha1"]
    try:
        _evolvebin.cevars.alpha1 = alpha1_val
    except ValueError:
        # This COSMIC build has scalar alpha1 — only alpha_1 (first CE event)
        # is passed to Fortran. alpha_2 is sampled but cannot be independently
        # controlled at the Fortran level in this COSMIC version.
        if not getattr(set_evolvebin_flags, '_alpha_warned', False):
            warnings.warn(
                "COSMIC cevars.alpha1 is a scalar in this build — "
                "assigning alpha_1 value only. alpha_2 samples will not "
                "independently affect CE evolution. "
                "Use a COSMIC build where alpha1 is a rank-1 array of size 2 "
                "for full independent-alpha inference.",
                RuntimeWarning, stacklevel=2,
            )
            set_evolvebin_flags._alpha_warned = True
        _evolvebin.cevars.alpha1 = float(np.asarray(alpha1_val).flat[0])

    _evolvebin.cevars.lambdaf          = flags["lambdaf"]
    _evolvebin.ceflags.ceflag          = flags["ceflag"]
    _evolvebin.flags.tflag             = flags["tflag"]
    _evolvebin.flags.ifflag            = flags["ifflag"]
    _evolvebin.flags.wdflag            = flags["wdflag"]
    _evolvebin.flags.rtmsflag          = flags["rtmsflag"]
    _evolvebin.snvars.pisn             = flags["pisn"]
    _evolvebin.flags.bhflag            = flags["bhflag"]
    _evolvebin.flags.remnantflag       = flags["remnantflag"]
    _evolvebin.ceflags.cekickflag      = flags["cekickflag"]
    _evolvebin.ceflags.cemergeflag     = flags["cemergeflag"]
    _evolvebin.ceflags.cehestarflag    = flags["cehestarflag"]
    _evolvebin.flags.grflag            = flags["grflag"]
    _evolvebin.flags.bhms_coll_flag    = flags["bhms_coll_flag"]
    _evolvebin.snvars.mxns             = flags["mxns"]
    _evolvebin.points.pts1             = flags["pts1"]
    _evolvebin.points.pts2             = flags["pts2"]
    _evolvebin.points.pts3             = flags["pts3"]
    _evolvebin.snvars.ecsn             = flags["ecsn"]
    _evolvebin.snvars.ecsn_mlow        = flags["ecsn_mlow"]
    _evolvebin.flags.aic               = flags["aic"]
    _evolvebin.ceflags.ussn            = flags["ussn"]
    _evolvebin.snvars.sigma            = flags["sigma"]
    _evolvebin.snvars.sigmadiv         = flags["sigmadiv"]
    _evolvebin.snvars.bhsigmafrac      = flags["bhsigmafrac"]
    _evolvebin.snvars.polar_kick_angle = flags["polar_kick_angle"]
    _evolvebin.snvars.natal_kick_array = flags["natal_kick_array"]
    _evolvebin.cevars.qcrit_array      = flags["qcrit_array"]
    _evolvebin.mtvars.don_lim          = flags["don_lim"]

    # ---- acc_lim: same scalar/array issue as alpha1 ----
    acc_lim_val = flags["acc_lim"]
    try:
        _evolvebin.mtvars.acc_lim = acc_lim_val
    except ValueError:
        if not getattr(set_evolvebin_flags, '_acclim_warned', False):
            warnings.warn(
                "COSMIC mtvars.acc_lim is a scalar in this build — "
                "assigning flim_1 value only. flim_2 will not independently "
                "affect stable MT in the second episode.",
                RuntimeWarning, stacklevel=2,
            )
            set_evolvebin_flags._acclim_warned = True
        _evolvebin.mtvars.acc_lim = float(np.asarray(acc_lim_val).flat[0])

    _evolvebin.windvars.beta           = flags["beta"]
    _evolvebin.windvars.xi             = flags["xi"]
    _evolvebin.windvars.acc2           = flags["acc2"]
    _evolvebin.windvars.epsnov         = flags["epsnov"]
    _evolvebin.windvars.eddfac         = flags["eddfac"]
    _evolvebin.windvars.gamma          = flags["gamma"]
    _evolvebin.flags.bdecayfac         = flags["bdecayfac"]
    _evolvebin.magvars.bconst          = flags["bconst"]
    _evolvebin.magvars.ck              = flags["ck"]
    _evolvebin.flags.windflag          = flags["windflag"]
    _evolvebin.flags.qcflag            = flags["qcflag"]
    _evolvebin.flags.eddlimflag        = flags["eddlimflag"]
    _evolvebin.tidalvars.fprimc_array  = flags["fprimc_array"]
    _evolvebin.rand1.idum1             = flags["randomseed"]
    _evolvebin.flags.bhspinflag        = flags["bhspinflag"]
    _evolvebin.snvars.bhspinmag        = flags["bhspinmag"]
    _evolvebin.mixvars.rejuv_fac       = flags["rejuv_fac"]
    _evolvebin.flags.rejuvflag         = flags["rejuvflag"]
    _evolvebin.flags.htpmb             = flags["htpmb"]
    _evolvebin.flags.st_cr             = flags["ST_cr"]
    _evolvebin.flags.st_tide           = flags["ST_tide"]
    _evolvebin.snvars.rembar_massloss  = flags["rembar_massloss"]
    _evolvebin.metvars.zsun            = flags["zsun"]
    _evolvebin.snvars.kickflag         = flags["kickflag"]
    _evolvebin.se_flags.using_metisse  = 0
    _evolvebin.se_flags.using_sse      = 1


# ---------------------------------------------------------------------------
# COSMIC binary evolution call
# ---------------------------------------------------------------------------

def evolv2(
    params_in: dict,
    params_out: list[str],
    fixed_params: dict,
) -> tuple:
    """Evolve a binary from ZAMS to merger (or dissolution) using COSMIC.

    Maps the 17-D parameter vector (θ, Λ, X) → merger observables θ_GW.
    This is the deterministic function f() in Eq. (1) of Lucky Strikes.

    Parameters
    ----------
    params_in : dict
        Sampled parameters from Nautilus.  Required keys depend on the config;
        at minimum {'m1', 'q', 'logtb', 'logZ'}.
    params_out : list[str]
        Column names to extract from the merger state (e.g. ['mass_1', 'mass_2']).
    fixed_params : dict
        Parameters held constant for this run (e.g. {'vk1': 0.0, ...}).

    Returns
    -------
    final_state : pd.Series or None
        Merger-time values of params_out columns.  None if no merger occurs.
    bpp_array : np.ndarray or None
        Full binary evolution track, shape (25, len(COLS_KEEP)), for blob storage.
    kick_array : np.ndarray or None
        Kick info array, shape (2, len(KICK_COLUMNS)), for blob storage.

    Notes
    -----
    kstar_1 == kstar_2 == 14 identifies a double BH binary.
    evol_type == 3 indicates inspiral / GW-driven merger within a Hubble time.
    The integration runs to tphysf = 13700 Myr (Hubble time).
    """
    # ---- Unpack binary initial conditions ----
    m1          = params_in["m1"]
    q           = params_in["q"]
    m2          = q * m1
    tb          = 10.0 ** params_in["logtb"]          # period in days
    e           = params_in.get("e", 0.0)             # eccentricity; fixed=0 in Lucky Strikes
    metallicity = 10.0 ** params_in["logZ"]

    # ---- Set COSMIC physics flags ----
    flags = set_flags(params_in, fixed_params)
    set_evolvebin_flags(flags)

    # ---- Column index arrays for bpp / bcm output ----
    col_inds_bpp = np.zeros(len(ALL_COLUMNS), dtype=int)
    col_inds_bpp[:len(BPP_COLUMNS)] = [ALL_COLUMNS.index(c) + 1 for c in BPP_COLUMNS]
    n_col_bpp = len(BPP_COLUMNS)

    col_inds_bcm = np.zeros(len(ALL_COLUMNS), dtype=int)
    col_inds_bcm[:len(BCM_COLUMNS)] = [ALL_COLUMNS.index(c) + 1 for c in BCM_COLUMNS]
    n_col_bcm = len(BCM_COLUMNS)

    _evolvebin.col.n_col_bpp    = n_col_bpp
    _evolvebin.col.col_inds_bpp = col_inds_bpp
    _evolvebin.col.n_col_bcm    = n_col_bcm
    _evolvebin.col.col_inds_bcm = col_inds_bcm

    # ---- Initial state arrays ----
    kstar    = np.array([1, 1], dtype=int)
    mass     = np.array([m1, m2])
    mass0    = np.array([m1, m2])
    epoch    = np.zeros(2)
    ospin    = np.zeros(2)
    tphys    = 0.0
    tphysf   = 13700.0          # Hubble time [Myr]
    dtp      = 0.0              # 0.0 → output at every event
    rad      = np.zeros(2)
    lumin    = np.zeros(2)
    massc    = np.zeros(2)
    radc     = np.zeros(2)
    menv     = np.zeros(2)
    renv     = np.zeros(2)
    B_0      = np.zeros(2)
    bacc     = np.zeros(2)
    tacc     = np.zeros(2)
    tms      = np.zeros(2)
    bhspin   = np.zeros(2)
    zpars    = np.zeros(20)
    bkick    = np.zeros(20)
    kick_info = np.zeros((2, 18))

    # ---- Call COSMIC Fortran kernel ----
    [_, bpp_index, bcm_index, kick_info_arrays] = _evolvebin.evolv2(
        kstar, mass, tb, e, metallicity, tphysf,
        dtp, mass0, rad, lumin, massc, radc,
        menv, renv, ospin, B_0, bacc, tacc, epoch, tms,
        bhspin, tphys, zpars, bkick, kick_info,
    )

    # ---- Extract and clear output arrays (avoids Fortran global state leakage) ----
    bpp_raw = _evolvebin.binary.bpp[:25, :n_col_bpp].copy()
    _evolvebin.binary.bpp[:25, :n_col_bpp] = 0.0

    bcm_raw = _evolvebin.binary.bcm[:bcm_index, :n_col_bcm].copy()
    _evolvebin.binary.bcm[:bcm_index, :n_col_bcm] = 0.0

    # ---- Build DataFrames ----
    bpp = pd.DataFrame(bpp_raw, columns=BPP_COLUMNS)[COLS_KEEP]
    kick_df = pd.DataFrame(
        kick_info_arrays,
        columns=KICK_COLUMNS,
        index=kick_info_arrays[:, -1].astype(int),
    )

    # ---- Find merger row: BBH (kstar=14/14) that inspirals (evol_type=3) ----
    merger_mask = (
        (bpp.kstar_1 == 14) & (bpp.kstar_2 == 14) & (bpp.evol_type == 3)
    )
    merger_rows = bpp.loc[merger_mask]

    if len(merger_rows) == 0:
        # Binary did not form a merging BBH — signal failure to Nautilus
        return None, None, None

    final_state = merger_rows[params_out].iloc[0]
    bpp_array   = bpp.to_numpy()
    kick_array  = kick_df.to_numpy()

    return final_state, bpp_array, kick_array


# ---------------------------------------------------------------------------
# GW likelihood utilities
# ---------------------------------------------------------------------------

def load_gw_data(
    samples_path: str,
    approximant: str = "C01:Mixed",
    use_pe_weights: bool = True,
    include_redshift: bool = False,
    mass_frame: str = "auto",
    verbose: bool = True,
    return_metadata: bool = False,
    hdi_prob: float = 0.999,
) -> tuple:
    """Load GW posterior samples and build a source-frame KDE likelihood.

    Implements the likelihood construction from Lucky Strikes Appendix A,
    with an optional extension to a 3D KDE that includes the merger redshift.

    2D mode (include_redshift=False):
        KDE over (mc_source, q).  Matches Lucky Strikes exactly.

    3D mode (include_redshift=True):
        KDE over (mc_source, q, z_src).  Enables constraint on the delay
        time and hence on the formation redshift / metallicity when z_form
        is a sampled parameter.  Requires the '_zform' config variants.

    Parameters
    ----------
    samples_path : str
        Path to a pesummary-compatible HDF5 posterior file.
    approximant : str
        Posterior approximant label.  Default 'C01:Mixed' covers most
        GWTC-2/3 BBH events.  Falls back gracefully if unavailable.
    use_pe_weights : bool
        Apply Callister (2021) Jacobian reweighting to recover the likelihood
        from the PE posterior samples.  Should be True for production runs only
        when the PE prior matches the documented Callister/Lucky-Strikes form.
    include_redshift : bool
        If True, build a 3D KDE over (mc, q, z_src) instead of 2D (mc, q).
    mass_frame : {"auto", "source", "detector"}
        How to interpret posterior mass columns.  ``auto`` prefers explicit
        source-frame aliases, then explicit detector-frame aliases, and only
        falls back to ambiguous ``mass_1``/``mass_2`` with a warning.
    verbose : bool
        Print available posterior sample keys and KDE diagnostics.
    return_metadata : bool
        If True, append a metadata dictionary describing mass-frame handling.
    hdi_prob : float
        Probability mass used for reported HDI support bounds.

    Returns
    -------
    kde : scipy.stats.gaussian_kde
        2D KDE over (mc, q) or 3D over (mc, q, z_src).
    q_bounds : tuple[float, float]
        HDI of q using ``hdi_prob`` — used by configured support gates.
    mc_bounds : tuple[float, float]
        HDI of mc using ``hdi_prob`` — for diagnostics and metadata.
    z_bounds : tuple[float, float] or None
        HDI of z_src using ``hdi_prob`` in 3D mode; None in 2D mode.
    raw_samples : np.ndarray
        Source-frame samples, shape (N, 2) or (N, 3).

    Notes
    -----
    Jacobian (Callister 2021 / Lucky Strikes App. A):
        π_PE(m1_src, m2_src, z) ∝ D_L^2 (1+z)^2 (dD_L/dz) m1_src^2 / mc_src
    Importance weights are the reciprocal, evaluated per sample.
    """
    from pesummary.io import read
    import arviz as az
    from scipy.stats import gaussian_kde

    data = read(samples_path, package="gw")

    available = list(data.samples_dict.keys())
    if approximant not in available:
        fallback = available[0]
        print(f"[load_gw_data] '{approximant}' not found; using '{fallback}'. "
              f"Available: {available}")
        approximant = fallback

    samples = data.samples_dict[approximant]
    sample_keys = list(samples.keys())
    if verbose:
        print(f"[load_gw_data] Posterior sample keys ({approximant}): {sample_keys}")

    mass_info = resolve_mass_columns(samples, mass_frame=mass_frame)
    m1_col, m2_col = mass_info["columns"]
    m1_raw = np.asarray(samples[m1_col])
    m2_raw = np.asarray(samples[m2_col])
    dL = np.asarray(samples['luminosity_distance'])

    # Source-frame conversion.  Explicit source-frame columns must not be
    # divided by (1+z); detector-frame columns are converted once.
    redshift = zofdL(dL)
    if mass_info["frame"] == "source":
        m1_src = m1_raw
        m2_src = m2_raw
    elif mass_info["frame"] == "detector":
        m1_src = m1_raw / (1.0 + redshift)
        m2_src = m2_raw / (1.0 + redshift)
    else:
        raise ValueError(f"Unsupported mass frame: {mass_info['frame']}")
    mc_src   = (m1_src * m2_src)**(3.0/5.0) / (m1_src + m2_src)**(1.0/5.0)
    q_src    = m2_src / m1_src   # ≤ 1

    # Importance weights: inverse PE prior Jacobian (Callister 2021).
    # This is valid for the standard PE prior documented in Lucky Strikes:
    # uniform in detector-frame component masses and comoving-volume-like
    # distance/redshift.  The calculation below always evaluates the Jacobian
    # in source-frame masses; the selected mass columns only control whether
    # a (1+z) conversion was required to obtain those source masses.
    pe_prior_weighting_used = bool(use_pe_weights)
    if use_pe_weights:
        jacobian = (
            dL**2
            * (1.0 + redshift)**2
            * ddLdz(redshift)
            * m1_src**2
            / mc_src
        )
        weights = 1.0 / jacobian
        weights = weights / weights.sum()
    else:
        weights = np.ones(len(m1_src)) / len(m1_src)

    q_lo,  q_hi  = az.hdi(q_src,  hdi_prob=hdi_prob)
    mc_lo, mc_hi = az.hdi(mc_src, hdi_prob=hdi_prob)

    if include_redshift:
        # Use the redshift values already implied by the PE D_L samples
        # (consistent with the PE cosmological model internally)
        z_src = redshift
        z_lo, z_hi = az.hdi(z_src, hdi_prob=hdi_prob)

        raw_samples = np.column_stack([mc_src, q_src, z_src])
        kde = gaussian_kde(raw_samples.T, weights=weights)

        if verbose:
            print(f"[load_gw_data] 3D KDE ({approximant}): "
                  f"mc=[{mc_lo:.3f},{mc_hi:.3f}] M_sun, "
                  f"q=[{q_lo:.4f},{q_hi:.4f}], "
                  f"z=[{z_lo:.4f},{z_hi:.4f}]")

        metadata = {
            "mass_frame_used": mass_info["frame"],
            "pe_prior_weighting_used": pe_prior_weighting_used,
            "mass_column_names": np.array([m1_col, m2_col]),
        }
        result = (kde, (q_lo, q_hi), (mc_lo, mc_hi), (z_lo, z_hi), raw_samples)
        return result + (metadata,) if return_metadata else result

    else:
        raw_samples = np.column_stack([mc_src, q_src])
        kde = gaussian_kde(raw_samples.T, weights=weights)

        if verbose:
            print(f"[load_gw_data] 2D KDE ({approximant}): "
                  f"mc=[{mc_lo:.3f},{mc_hi:.3f}] M_sun, "
                  f"q=[{q_lo:.4f},{q_hi:.4f}]")

        metadata = {
            "mass_frame_used": mass_info["frame"],
            "pe_prior_weighting_used": pe_prior_weighting_used,
            "mass_column_names": np.array([m1_col, m2_col]),
        }
        result = (kde, (q_lo, q_hi), (mc_lo, mc_hi), None, raw_samples)
        return result + (metadata,) if return_metadata else result


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def str_to_bool(value: str) -> bool:
    """Parse common string representations of boolean values."""
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    if value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f"'{value}' cannot be parsed as a boolean.")