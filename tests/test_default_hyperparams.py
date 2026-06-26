import numpy as np
import pytest

hb = pytest.importorskip("hierarchical_backpop_jax")
jnp = pytest.importorskip("jax.numpy")


def test_default_hyperparams_gives_finite_event_and_selection_likelihoods():
    lp = hb.default_hyperparams()
    param_idx = {
        "alpha_1": 0,
        "alpha_2": 1,
        "flim_1": 2,
        "flim_2": 3,
        "vk1": 4,
        "vk2": 5,
    }
    lo = jnp.array([0.1, 0.1, 0.0, 0.0, 1.0, 1.0], dtype=jnp.float64)
    hi = jnp.array([10.0, 10.0, 1.0, 1.0, 500.0, 500.0], dtype=jnp.float64)
    samples = jnp.array(
        [
            [1.0, 1.2, 0.25, 0.75, 80.0, 120.0],
            [1.4, 0.8, 0.45, 0.55, 160.0, 200.0],
            [0.7, 1.6, 0.65, 0.35, 240.0, 60.0],
        ],
        dtype=jnp.float64,
    )

    log_wr_fn = hb.make_log_weight_ratio_fn(samples, param_idx, lo, hi)
    event_log_likelihood = hb.make_hierarchical_log_likelihood(
        [log_wr_fn], jnp.array([0.0], dtype=jnp.float64), samples.shape[0], None, 1
    )
    assert np.isfinite(np.asarray(event_log_likelihood(lp)))

    log_alpha_fn = hb.make_log_alpha_fn(
        K_np=np.ones((2, samples.shape[0]), dtype=np.float32),
        log_v=jnp.zeros(2, dtype=jnp.float64),
        log_norm=0.0,
        theta_inj=samples,
        lo_inj=lo,
        hi_inj=hi,
        param_idx_inj=param_idx,
        kick_sigma=150.0,
    )
    assert np.isfinite(np.asarray(log_alpha_fn(lp)))
