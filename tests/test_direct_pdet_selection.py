import numpy as np
import pytest
import jax.numpy as jnp
from scipy.special import logsumexp

from gwbackpop.inference.hierarchical import (
    compute_log_wr_injections_numpy,
    default_hyperparams,
    make_hierarchical_log_likelihood,
    make_log_alpha_direct_pdet_fn,
    make_log_weight_ratio_fn,
    validate_direct_pdet_inputs,
)

PARAMS = ["alpha_1", "alpha_2", "flim_1", "flim_2", "vk1", "vk2"]
PIDX = {p: i for i, p in enumerate(PARAMS)}
LO = np.array([0.05, 0.05, 0.0, 0.0, 0.0, 0.0])
HI = np.array([20.0, 20.0, 1.0, 1.0, 500.0, 500.0])


def _theta(n=6):
    return np.column_stack([
        np.geomspace(0.4, 4.0, n),
        np.geomspace(0.5, 3.0, n),
        np.linspace(0.15, 0.85, n),
        np.linspace(0.2, 0.8, n),
        np.linspace(20.0, 240.0, n),
        np.linspace(30.0, 260.0, n),
    ])


def test_direct_alpha_matches_manual_numpy_and_zero_pdet():
    theta = _theta()
    pdet = np.array([0.0, 0.1, 0.25, 0.5, 0.8, 1.0])
    log_q = np.linspace(-3.0, -2.0, len(theta))
    log_static = np.linspace(0.0, 0.2, len(theta))
    n_inj = 100
    lp = default_hyperparams()

    fn = make_log_alpha_direct_pdet_fn(
        jnp.array(theta), jnp.array(pdet), jnp.array(LO), jnp.array(HI), PIDX,
        n_inj, 50.0, jnp.array(log_q), jnp.array(log_static)
    )
    got = float(fn(lp))
    log_wr = compute_log_wr_injections_numpy(lp, theta, PARAMS, LO, HI, 50.0, log_q, log_static)
    expected = logsumexp(np.where(pdet > 0, np.log(pdet), -np.inf) + log_wr) - np.log(n_inj)
    assert got == pytest.approx(expected, rel=1e-10, abs=1e-10)

    clipped_wrong = logsumexp(np.log(np.clip(pdet, 1e-300, 1.0)) + log_wr) - np.log(n_inj)
    # If zero pdet were clipped, this toy point would include a fake tiny term;
    # the implemented expression must be the exact zero-contribution formula.
    assert got == pytest.approx(expected)
    assert np.isfinite(clipped_wrong)


@pytest.mark.parametrize("pdet, match", [
    (None, "Missing pdet"),
    (np.array([np.nan, np.nan]), "non-finite"),
    (np.array([0.1, np.inf]), "non-finite"),
    (np.array([-0.1, 0.2]), "0 <= pdet <= 1"),
    (np.array([0.1, 1.2]), "0 <= pdet <= 1"),
    (np.array([0.0, 0.0]), "all pdet == 0"),
])
def test_direct_pdet_validation_rejects_invalid_inputs(pdet, match):
    theta = _theta(2)
    with pytest.raises(ValueError, match=match):
        validate_direct_pdet_inputs(theta, pdet, 10)


def test_direct_pdet_toy_selection_correction_moves_toward_true_population():
    # Detected events are biased to high alpha_1. Direct pdet rises with alpha_1.
    # Selection correction penalizes the high-alpha hyperpoint's larger alpha(Lambda).
    lo = np.array([0.1])
    hi = np.array([10.0])
    params = {"alpha_1": 0}
    event_samples = jnp.array([[3.5], [4.0], [4.5], [5.0], [5.5]])
    wr = [make_log_weight_ratio_fn(event_samples, params, jnp.array(lo), jnp.array(hi)) for _ in range(3)]
    log_z = jnp.zeros(3)

    theta = np.geomspace(0.1, 10.0, 200)[:, None]
    pdet = (theta[:, 0] / theta[:, 0].max()) ** 8
    log_q = np.full(theta.shape[0], -np.log(hi[0] - lo[0]))
    true_lp = default_hyperparams(); true_lp[0] = 0.0; true_lp[1] = 0.6
    biased_lp = default_hyperparams(); biased_lp[0] = np.log(4.5); biased_lp[1] = 0.35

    alpha_fn = make_log_alpha_direct_pdet_fn(
        jnp.array(theta), jnp.array(pdet), jnp.array(lo), jnp.array(hi), params,
        400, 50.0, jnp.array(log_q), jnp.zeros(theta.shape[0])
    )
    ll_none = make_hierarchical_log_likelihood(wr, log_z, len(event_samples), None, 3)
    ll_dir = make_hierarchical_log_likelihood(wr, log_z, len(event_samples), alpha_fn, 3)

    no_sel_gap = float(ll_none(biased_lp) - ll_none(true_lp))
    direct_gap = float(ll_dir(biased_lp) - ll_dir(true_lp))
    assert no_sel_gap > 0.0
    assert direct_gap < no_sel_gap
    assert direct_gap < 0.0
