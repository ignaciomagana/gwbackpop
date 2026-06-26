"""BackPop prior/configuration definitions that do not import COSMIC."""
from __future__ import annotations

import numpy as np


def get_backpop_config(config_name: str) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """Return Nautilus prior bounds, sampled parameter names, and fixed values."""
    m1_range = (2.0, 150.0)
    q_range = (0.01, 1.0)
    logtb_range = (np.log10(0.1), np.log10(5000.0))
    logZ_range = (np.log10(1e-4), np.log10(0.03))
    alpha_range = (0.1, 20.0)
    flim_range = (0.0, 1.0)
    vk_range = (0.0, 500.0)
    theta_range = (0.0, 360.0)
    phi_range = (-90.0, 90.0)
    omega_range = (0.0, 360.0)

    def bounds(*ranges):
        lo = np.array([r[0] for r in ranges])
        hi = np.array([r[1] for r in ranges])
        return lo, hi

    if config_name == "lucky_strikes":
        params_in = [
            'm1', 'q', 'logtb', 'logZ',
            'alpha_1', 'alpha_2', 'flim_1', 'flim_2',
            'vk1', 'theta1', 'phi1', 'omega1',
            'vk2', 'theta2', 'phi2', 'omega2',
        ]
        lower_bound, upper_bound = bounds(
            m1_range, q_range, logtb_range, logZ_range,
            alpha_range, alpha_range, flim_range, flim_range,
            vk_range, theta_range, phi_range, omega_range,
            vk_range, theta_range, phi_range, omega_range,
        )
        fixed_params = {}
    elif config_name == "lucky_strikes_fixed_vk1":
        params_in = [
            'm1', 'q', 'logtb', 'logZ',
            'alpha_1', 'alpha_2', 'flim_1', 'flim_2',
            'vk2', 'theta2', 'phi2', 'omega2',
        ]
        lower_bound, upper_bound = bounds(
            m1_range, q_range, logtb_range, logZ_range,
            alpha_range, alpha_range, flim_range, flim_range,
            vk_range, theta_range, phi_range, omega_range,
        )
        fixed_params = {'vk1': 0.0, 'theta1': 0.0, 'phi1': 0.0, 'omega1': 0.0}
    elif config_name == "bbh_no_kicks":
        params_in = [
            'm1', 'q', 'logtb', 'logZ',
            'alpha_1', 'alpha_2', 'flim_1', 'flim_2',
        ]
        lower_bound, upper_bound = bounds(
            m1_range, q_range, logtb_range, logZ_range,
            alpha_range, alpha_range, flim_range, flim_range,
        )
        fixed_params = {
            'vk1': 0.0, 'theta1': 0.0, 'phi1': 0.0, 'omega1': 0.0,
            'vk2': 0.0, 'theta2': 0.0, 'phi2': 0.0, 'omega2': 0.0,
        }
    elif config_name == "lucky_strikes_zform":
        params_in = [
            'm1', 'q', 'logtb', 'logZ', 'z_form',
            'alpha_1', 'alpha_2', 'flim_1', 'flim_2',
            'vk1', 'theta1', 'phi1', 'omega1',
            'vk2', 'theta2', 'phi2', 'omega2',
        ]
        lower_bound, upper_bound = bounds(
            m1_range, q_range, logtb_range, logZ_range, (1e-4, 20.0),
            alpha_range, alpha_range, flim_range, flim_range,
            vk_range, theta_range, phi_range, omega_range,
            vk_range, theta_range, phi_range, omega_range,
        )
        fixed_params = {}
    elif config_name == "lucky_strikes_fixed_vk1_zform":
        params_in = [
            'm1', 'q', 'logtb', 'logZ', 'z_form',
            'alpha_1', 'alpha_2', 'flim_1', 'flim_2',
            'vk2', 'theta2', 'phi2', 'omega2',
        ]
        lower_bound, upper_bound = bounds(
            m1_range, q_range, logtb_range, logZ_range, (1e-4, 20.0),
            alpha_range, alpha_range, flim_range, flim_range,
            vk_range, theta_range, phi_range, omega_range,
        )
        fixed_params = {'vk1': 0.0, 'theta1': 0.0, 'phi1': 0.0, 'omega1': 0.0}
    elif config_name == "bbh_no_kicks_zform":
        params_in = [
            'm1', 'q', 'logtb', 'logZ', 'z_form',
            'alpha_1', 'alpha_2', 'flim_1', 'flim_2',
        ]
        lower_bound, upper_bound = bounds(
            m1_range, q_range, logtb_range, logZ_range, (1e-4, 20.0),
            alpha_range, alpha_range, flim_range, flim_range,
        )
        fixed_params = {
            'vk1': 0.0, 'theta1': 0.0, 'phi1': 0.0, 'omega1': 0.0,
            'vk2': 0.0, 'theta2': 0.0, 'phi2': 0.0, 'omega2': 0.0,
        }
    else:
        raise ValueError(f"Unknown config_name '{config_name}'")

    return lower_bound, upper_bound, params_in, fixed_params
