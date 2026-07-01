"""
cosmo_prior.py
--------------
Cosmological priors for the BackPop framework, following Andrews et al. 2021
(ApJL 914, L32) §2.3.

Implements:
  1. Madau & Dickinson (2014) star formation rate ψ(z)
  2. Stellar mass density ρ_*(z) via numerical integration over SFR history
  3. Mean metallicity evolution Z_mean(z) = y · ρ_*(z) / ρ_b
  4. Formation redshift prior P(z_form) ∝ (dV_c/dz) · ψ(z) / (1+z)
  5. Metallicity prior P(log Z | z_form): truncated Normal centred on log Z_mean(z_form)
     with σ = 0.5 dex, bounded by [Z_min, Z_max]

All lookup tables are precomputed at import time on a fine z-grid and
interpolated at runtime — no per-sample numerical integration.

References
----------
Andrews et al. 2021, ApJL 914, L32 (§2.3 and Eqs. 4–8)
Madau & Dickinson 2014, ARA&A 52, 415
Planck Collaboration (Ade et al.) 2016, A&A 594, A13 (Planck 2015 params)
"""

from __future__ import annotations

import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d
from scipy.stats import truncnorm
from astropy.cosmology import Planck15

# ---------------------------------------------------------------------------
# Physical / cosmological constants (Andrews+2021 values)
# ---------------------------------------------------------------------------

# Yield: fraction of stellar mass returned as metals
# Computed by integrating yields over a Salpeter IMF from 10–60 M_sun
_Y_YIELD: float = 0.23

# Fraction of stellar mass returned to the ISM (Salpeter IMF, same limits)
_R_RETURN: float = 0.29

# Baryon density of the universe: ρ_b = 2.77e11 · Ω_b h^2  [M_sun Mpc^-3]
# Using Planck 2015: Ω_b h^2 = 0.0223  (Ade et al. 2016)
_OMEGA_B_H2: float = 0.0223
_RHO_B: float = 2.77e11 * _OMEGA_B_H2   # M_sun Mpc^-3

# Metallicity prior width (half-decade, following Andrews+2021 Eq. 8)
_SIGMA_LOG_Z: float = 0.5   # dex

# Metallicity bounds matching COSMIC's valid range (Andrews+2021 Eq. 8)
_Z_MIN: float = 5e-5
_Z_MAX: float = 3e-2
_LOG_Z_MIN: float = np.log10(_Z_MIN)
_LOG_Z_MAX: float = np.log10(_Z_MAX)

# Formation redshift integration range
_Z_FORM_MIN: float = 1e-4   # effectively z=0
_Z_FORM_MAX: float = 20.0   # well above peak of SFR; ψ negligible beyond this

# Resolution of precomputed tables
_N_ZGRID: int = 5_000


# ---------------------------------------------------------------------------
# Star formation rate: Madau & Dickinson (2014)
# ---------------------------------------------------------------------------

def sfr_madau_dickinson(z: np.ndarray | float) -> np.ndarray | float:
    """Volumetric star formation rate from Madau & Dickinson (2014).

    ψ(z) = 0.015 · (1+z)^2.7 / [1 + ((1+z)/2.9)^5.6]   [M_sun yr^-1 Mpc^-3]

    Andrews+2021 Eq. (4).
    """
    return 0.015 * (1.0 + z)**2.7 / (1.0 + ((1.0 + z) / 2.9)**5.6)


# ---------------------------------------------------------------------------
# Precomputed lookup tables (built once at import)
# ---------------------------------------------------------------------------

# Fine z-grid spanning the full prior range
_zgrid = np.linspace(_Z_FORM_MIN, _Z_FORM_MAX, _N_ZGRID)

# Hubble parameter H(z) in km/s/Mpc via astropy
_H_grid = Planck15.H(_zgrid).value          # km/s/Mpc

# Comoving distance D_C(z) in Mpc via astropy
_DC_grid = Planck15.comoving_distance(_zgrid).to('Mpc').value


def _build_stellar_density_table() -> np.ndarray:
    """Precompute ρ_*(z) on the z-grid via numerical integration.

    From Andrews+2021 Eq. (7):
        ρ_*(z) = (1-R) ∫_z^z_max  ψ(z') / [H(z') · (1+z')] dz'

    H must be in yr^-1 and ψ in M_sun/yr/Mpc^3 so that the result is
    in M_sun/Mpc^3.

    Conversion: H [yr^-1] = H [km/s/Mpc] × (1000 m/km) × (3.156e7 s/yr)
                           / (3.086e22 m/Mpc)
                           = H × 1.0221e-12 yr^-1
    """
    from scipy.integrate import cumulative_trapezoid

    # Convert H from km/s/Mpc to yr^-1
    _KM_TO_M      = 1.0e3
    _MPC_TO_M     = 3.0857e22
    _S_TO_YR      = 1.0 / 3.1558e7        # s → yr
    H_per_yr = _H_grid * _KM_TO_M / _MPC_TO_M / _S_TO_YR

    # Integrand: M_sun yr^-1 Mpc^-3 / yr^-1 = M_sun Mpc^-3
    integrand = (1.0 - _R_RETURN) * sfr_madau_dickinson(_zgrid) / (H_per_yr * (1.0 + _zgrid))

    # Cumulative integral from z[i] to z_max (reverse cumsum trick)
    cumulative = cumulative_trapezoid(integrand[::-1], _zgrid[::-1], initial=0.0)[::-1]
    rho_star = np.abs(cumulative)   # M_sun Mpc^-3

    return rho_star


_rho_star_grid = _build_stellar_density_table()

# Mean metallicity Z_mean(z) from Andrews+2021 Eq. (6)
_Z_mean_grid = _Y_YIELD * _rho_star_grid / _RHO_B

# Clip to physical range (tiny rounding at very high z)
_Z_mean_grid = np.clip(_Z_mean_grid, _Z_MIN, _Z_MAX)

# Formation prior weight: dV_c/dz · ψ(z) / (1+z)
# dV_c/dz = 4π D_C^2 · c/H(z)  [Mpc^3 per unit z]
# (4π and c factors cancel in the normalisation, so we drop them)
_c_km_s = 2.998e5   # speed of light [km/s]
_sfr_grid = sfr_madau_dickinson(_zgrid)
_dVc_dz_grid = _DC_grid**2 * _c_km_s / _H_grid          # Mpc^3 per unit z (up to 4π)
_prior_weight_grid = _dVc_dz_grid * _sfr_grid / (1.0 + _zgrid)

# Normalise to a proper PDF over [Z_FORM_MIN, Z_FORM_MAX]
_prior_norm = np.trapz(_prior_weight_grid, _zgrid)
_log_prior_norm = np.log(_prior_norm)

# Build interpolators (log-space for smoothness where values span orders of magnitude)
_log_Z_mean_interp = interp1d(
    _zgrid,
    np.log10(np.clip(_Z_mean_grid, 1e-10, None)),
    bounds_error=False,
    # left fill = low-z (high metallicity), right fill = high-z (low metallicity)
    fill_value=(np.log10(_Z_mean_grid[0]), np.log10(_Z_mean_grid[-1])),
)
_log_prior_weight_interp = interp1d(
    _zgrid,
    np.log(np.clip(_prior_weight_grid, 1e-300, None)),
    bounds_error=False,
    fill_value=-np.inf,
)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def log_prior_z_form(z_form: float) -> float:
    """Log prior probability on the formation redshift.

    P(z_form) ∝ (dV_c/dz) · ψ(z_form) / (1+z_form)

    This is the SFR-weighted comoving volume element — the probability that
    a binary forms at redshift z_form, accounting for both the available
    volume and the star formation activity (Andrews+2021 Eq. 5).

    Parameters
    ----------
    z_form : float
        Formation redshift. Valid range: [_Z_FORM_MIN, _Z_FORM_MAX].

    Returns
    -------
    float
        Log prior (normalised over [_Z_FORM_MIN, _Z_FORM_MAX]).
        Returns -inf outside the valid range.
    """
    if z_form < _Z_FORM_MIN or z_form > _Z_FORM_MAX:
        return -np.inf
    return float(_log_prior_weight_interp(z_form)) - _log_prior_norm


def z_mean_metallicity(z_form: float) -> float:
    """Mean metallicity Z_mean at formation redshift z_form.

    From Andrews+2021 Eq. (6): Z_mean(z) = y · ρ_*(z) / ρ_b.

    Returns metallicity in absolute units (X + Y + Z = 1).
    """
    return float(10.0 ** _log_Z_mean_interp(z_form))


def log_prior_logZ_given_z(log10_Z: float, z_form: float) -> float:
    """Log prior on log10(Z) conditioned on formation redshift z_form.

    From Andrews+2021 Eq. (8):
        P(Z | z_form) = TruncNorm(log10 Z; log10 Z_mean(z_form), 0.5, Z_min, Z_max)

    The truncated Normal is centred on the mean metallicity at z_form and
    has a width of 0.5 dex, reflecting the scatter in the galaxy
    mass–metallicity relation around the Madau & Dickinson (2014) mean.

    Parameters
    ----------
    log10_Z : float
        Sampled log10(metallicity) value.
    z_form : float
        Formation redshift (sampled alongside log10_Z).

    Returns
    -------
    float
        Log prior density. Returns -inf if outside [log10 Z_min, log10 Z_max].
    """
    return log_prior_logZ_given_z_on_support(log10_Z, z_form, _LOG_Z_MIN, _LOG_Z_MAX)


def log_prior_logZ_given_z_on_support(log10_Z: float, z_form: float, lo: float, hi: float) -> float:
    """Log P(log10 Z | z_form) normalized on the caller's finite support.

    BackPop configurations may choose a narrower metallicity interval than the
    default COSMIC/Andrews range.  Injection proposals and population
    numerators must use exactly the same ``[lo, hi]`` normalization.
    """
    lo = float(lo)
    hi = float(hi)
    if not lo < hi:
        raise ValueError("logZ support must satisfy lo < hi")
    if log10_Z < lo or log10_Z > hi:
        return -np.inf

    log10_Z_mean = float(_log_Z_mean_interp(z_form))

    # Truncated normal: standardise to unit normal, compute truncation limits
    a = (lo - log10_Z_mean) / _SIGMA_LOG_Z
    b = (hi - log10_Z_mean) / _SIGMA_LOG_Z
    x = (log10_Z - log10_Z_mean) / _SIGMA_LOG_Z

    return float(truncnorm.logpdf(x, a, b) - np.log(_SIGMA_LOG_Z))


def draw_logZ_given_z_on_support(rng: np.random.Generator, z_form: float, lo: float, hi: float) -> float:
    """Draw log10(Z) from P(log10 Z | z_form) normalized on ``[lo, hi]``."""
    lo = float(lo)
    hi = float(hi)
    if not lo < hi:
        raise ValueError("logZ support must satisfy lo < hi")
    mu = float(_log_Z_mean_interp(z_form))
    a = (lo - mu) / _SIGMA_LOG_Z
    b = (hi - mu) / _SIGMA_LOG_Z
    return float(truncnorm.rvs(a, b, loc=mu, scale=_SIGMA_LOG_Z, random_state=rng))


def z_merger_from_t_delay(z_form: float, t_delay_myr: float) -> float | None:
    """Compute the merger redshift given formation redshift and delay time.

    The delay time is the COSMIC output tphys at the merger row — the elapsed
    time from ZAMS formation to BBH inspiral in Myr.

    Parameters
    ----------
    z_form : float
        Formation redshift.
    t_delay_myr : float
        Delay time from COSMIC [Myr].

    Returns
    -------
    float or None
        Merger redshift z_merger > 0, or None if the binary would merge in
        the future (t_merger_lookback < 0) or before formation (unphysical).
    """
    # Lookback time at formation [Myr]
    t_lookback_form = Planck15.lookback_time(z_form).to('Myr').value

    # Lookback time at merger = lookback at formation minus delay time
    # (delay time moves forward in cosmic time → smaller lookback)
    t_lookback_merger = t_lookback_form - t_delay_myr

    if t_lookback_merger < 0.0:
        # Binary merges in the future relative to the observer — not observed
        return None

    if t_lookback_merger > t_lookback_form:
        # Merger before formation — unphysical
        return None

    # Convert lookback time to redshift via the backpop interpolator.
    # zoft is defined as interp1d(13700 - lookback_time, z) in backpop.py —
    # its x-axis is COSMIC TIME (age of universe in Myr), NOT lookback time.
    # BUG that was here: passing lookback time directly inverted the mapping,
    # returning z for the early universe (lookback=130 Myr → cosmic time
    # 130 Myr → z≈30) instead of the correct z (cosmic time 13570 Myr → z≈0.01).
    from gwbackpop.evolution.cosmic import zoft
    T_HUBBLE_MYR = 13700.0   # age of universe [Myr], matches backpop.py grid
    cosmic_time_merger = T_HUBBLE_MYR - t_lookback_merger
    z_merger = float(zoft(cosmic_time_merger))

    if z_merger < 0.0:
        return None

    return z_merger


# ---------------------------------------------------------------------------
# Diagnostic / utility
# ---------------------------------------------------------------------------

def print_prior_summary() -> None:
    """Print a summary of the cosmological prior at a few reference redshifts."""
    zs = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    print(f"{'z_form':>8}  {'log10 Z_mean':>14}  {'log P(z_form)':>14}")
    print("-" * 42)
    for z in zs:
        lZ  = float(_log_Z_mean_interp(z))
        lP  = log_prior_z_form(z)
        print(f"{z:8.1f}  {lZ:14.3f}  {lP:14.3f}")
