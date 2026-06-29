"""Diagnostic semi-analytic SNR-proxy detection probability.

This module is intentionally lightweight and approximate.  It is useful for
smoke tests, emulator seeding, and calibration diagnostics, but it is not a
production substitute for LVK/Farr selection or a validated pdet model.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.special import expit

try:
    from astropy.cosmology import Planck15 as _COSMO
except Exception:  # pragma: no cover
    _COSMO = None


_VALID_METHODS = {"hard_threshold", "orientation_monte_carlo", "logistic"}


def chirp_mass_detector(m1_src, m2_src, z):
    m1 = np.asarray(m1_src, dtype=np.float64) * (1.0 + np.asarray(z, dtype=np.float64))
    m2 = np.asarray(m2_src, dtype=np.float64) * (1.0 + np.asarray(z, dtype=np.float64))
    return (m1 * m2) ** (3.0 / 5.0) / (m1 + m2) ** (1.0 / 5.0)


def luminosity_distance_mpc(z):
    z = np.asarray(z, dtype=np.float64)
    if _COSMO is not None:
        return np.asarray(_COSMO.luminosity_distance(z).to_value("Mpc"), dtype=np.float64)
    # Low-z fallback; keeps tests usable if astropy is unavailable.
    c_over_h0 = 299792.458 / 67.74
    return c_over_h0 * z * (1.0 + 0.5 * z)


def snr_proxy_rho_opt(m1_src, m2_src, z, *, rho_ref=20.0, mc_ref=26.1, d_ref_mpc=1000.0, sensitivity_scale=1.0, high_mass_rolloff=True):
    mc = chirp_mass_detector(m1_src, m2_src, z)
    dl = np.maximum(luminosity_distance_mpc(z), 1e-9)
    rho = float(rho_ref) * (mc / float(mc_ref)) ** (5.0 / 6.0) * (float(d_ref_mpc) / dl) * float(sensitivity_scale)
    if high_mass_rolloff:
        mtot_det = (np.asarray(m1_src, dtype=np.float64) + np.asarray(m2_src, dtype=np.float64)) * (1.0 + np.asarray(z, dtype=np.float64))
        rho = rho / np.sqrt(1.0 + (mtot_det / 180.0) ** 4)
    return np.where(np.isfinite(rho), rho, 0.0)


@dataclass(frozen=True)
class SNRProxyPdet:
    method: str = "orientation_monte_carlo"
    rho_threshold: float = 10.0
    rho_ref: float = 20.0
    mc_ref: float = 26.1
    d_ref_mpc: float = 1000.0
    sensitivity_scale: float = 1.0
    logistic_width: float = 1.0
    seed: int = 1234
    n_orientation: int = 200000

    def __post_init__(self):
        if self.method not in _VALID_METHODS:
            raise ValueError(f"Unknown SNR pdet method {self.method!r}; choose {sorted(_VALID_METHODS)}")
        if self.rho_threshold <= 0 or self.rho_ref <= 0 or self.mc_ref <= 0 or self.d_ref_mpc <= 0 or self.sensitivity_scale < 0:
            raise ValueError("SNR proxy scales must be positive, with non-negative sensitivity_scale")
        if self.logistic_width <= 0:
            raise ValueError("logistic_width must be positive")
        if self.n_orientation <= 0:
            raise ValueError("n_orientation must be positive")
        rng = np.random.default_rng(self.seed)
        # Phenomenological bounded projection: antenna power and inclination.
        cosi = rng.uniform(-1.0, 1.0, self.n_orientation)
        inc = 0.5 * np.sqrt((1.0 + cosi**2) ** 2 + (2.0 * cosi) ** 2) / np.sqrt(2.0)
        antenna = rng.beta(2.0, 4.0, self.n_orientation) ** 0.5
        w = np.clip(inc * antenna, 0.0, 1.0)
        object.__setattr__(self, "_w_sorted", np.sort(w))

    def rho_opt(self, m1_src, m2_src, z):
        return snr_proxy_rho_opt(m1_src, m2_src, z, rho_ref=self.rho_ref, mc_ref=self.mc_ref, d_ref_mpc=self.d_ref_mpc, sensitivity_scale=self.sensitivity_scale)

    def pdet_from_rho(self, rho_opt):
        rho = np.asarray(rho_opt, dtype=np.float64)
        if self.method == "hard_threshold":
            out = (rho > self.rho_threshold).astype(np.float64)
        elif self.method == "logistic":
            out = expit((rho - self.rho_threshold) / self.logistic_width)
            out = np.where(rho > 0.0, out, 0.0)
        else:
            need = np.divide(self.rho_threshold, rho, out=np.full_like(rho, np.inf), where=rho > 0.0)
            idx = np.searchsorted(self._w_sorted, need, side="right")
            out = (self._w_sorted.size - idx) / self._w_sorted.size
        return np.clip(np.where(rho > 0.0, out, 0.0), 0.0, 1.0)

    def __call__(self, m1_src, m2_src, z):
        out = self.pdet_from_rho(self.rho_opt(m1_src, m2_src, z))
        return float(out) if np.ndim(out) == 0 else out


def make_snr_proxy_pdet_callable(method='orientation_monte_carlo', rho_threshold=10.0, rho_ref=20.0, mc_ref=26.1, d_ref_mpc=1000.0, sensitivity_scale=1.0, logistic_width=1.0, seed=1234, n_orientation=200000):
    return SNRProxyPdet(method=method, rho_threshold=rho_threshold, rho_ref=rho_ref, mc_ref=mc_ref, d_ref_mpc=d_ref_mpc, sensitivity_scale=sensitivity_scale, logistic_width=logistic_width, seed=seed, n_orientation=n_orientation)
