import numpy as np

from run_injections import LOWER, PARAMS, UPPER, compute_log_q_proposal


def _theta(vk1=30.0, vk2=70.0):
    theta = 0.5 * (LOWER + UPPER)
    theta[PARAMS.index("vk1")] = vk1
    theta[PARAMS.index("vk2")] = vk2
    return theta


def test_log_q_proposal_is_finite_for_representative_stored_injections():
    values = np.array([
        compute_log_q_proposal(_theta(20.0, 40.0), 1.0, -2.0, 50.0),
        compute_log_q_proposal(_theta(80.0, 120.0), 2.0, -2.5, 50.0),
        compute_log_q_proposal(_theta(5.0, 250.0), 0.5, -1.8, 50.0),
    ])
    assert np.all(np.isfinite(values))


def test_log_q_proposal_changes_with_kick_scale():
    theta = _theta(100.0, 150.0)
    log_q_50 = compute_log_q_proposal(theta, 1.5, -2.2, 50.0)
    log_q_100 = compute_log_q_proposal(theta, 1.5, -2.2, 100.0)
    assert np.isfinite(log_q_50)
    assert np.isfinite(log_q_100)
    assert not np.isclose(log_q_50, log_q_100)
