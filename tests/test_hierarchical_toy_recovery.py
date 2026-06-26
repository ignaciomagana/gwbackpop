"""Direct toy recovery tests for hierarchical reweighting and selection effects.

These tests intentionally avoid COSMIC, PESummary, LVK injection files, JAX, and
NumPy.  They exercise the same catalog likelihood structure used by the real
hierarchical analysis,

    sum_i log < p(x | Lambda) / pi_0(x) >_i - N log alpha(Lambda),

on a one-dimensional synthetic population where the selection integral is known.
"""

import math
import random

import pytest

TRUE_MU = 1.20
TRUE_SIGMA = 0.35
X_MIN = 0.05
X_MAX = 20.0
X0 = 4.0
DETECTION_SCALE = 0.45
N_EVENTS = 64
N_POSTERIOR_SAMPLES = 32
LOG_PRIOR_WIDTH = math.log(X_MAX - X_MIN)


def _sigmoid(z):
    if z >= 0.0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _lognormal_logpdf(x, mu, sigma):
    if x <= 0.0 or sigma <= 0.0:
        return -math.inf
    z = (math.log(x) - mu) / sigma
    return -math.log(x * sigma * math.sqrt(2.0 * math.pi)) - 0.5 * z * z


def _logsumexp(values):
    vmax = max(values)
    return vmax + math.log(sum(math.exp(v - vmax) for v in values))


def _draw_detected_events(seed=20240517):
    """Deterministic quantiles of the detected-event distribution.

    Using stratified quantiles keeps this CI test focused on the likelihood
    and selection correction rather than on occasional catalog-count noise.
    """
    del seed  # kept in the signature to document deterministic seeding policy
    weights = [
        _sigmoid((x - X0) / DETECTION_SCALE)
        * math.exp(_lognormal_logpdf(x, TRUE_MU, TRUE_SIGMA))
        for x in _QUAD_POINTS
    ]
    total = sum(weights)
    cdf = []
    running = 0.0
    for weight in weights:
        running += weight / total
        cdf.append(running)

    events = []
    k = 0
    for q_index in range(N_EVENTS):
        q = (q_index + 0.5) / N_EVENTS
        while cdf[k] < q:
            k += 1
        events.append(_QUAD_POINTS[k])
    return events


def _fake_posterior_samples(events, seed=20240518):
    """Posterior samples under a uniform baseline prior pi_0(x)=1/(X_MAX-X_MIN)."""
    rng = random.Random(seed)
    samples = []
    for x in events:
        row = []
        while len(row) < N_POSTERIOR_SAMPLES:
            sample = rng.gauss(x, 0.06 * x)
            if X_MIN < sample < X_MAX:
                row.append(sample)
        samples.append(row)
    return samples


@pytest.fixture(scope="module")
def toy_samples():
    return _fake_posterior_samples(_draw_detected_events())


_GRID = [(0.90 + i * 0.60 / 40.0, 0.22 + j * 0.28 / 40.0) for i in range(41) for j in range(41)]
_QUAD_N = 700
_QUAD_DX = (X_MAX - X_MIN) / _QUAD_N
_QUAD_POINTS = [X_MIN + (k + 0.5) * _QUAD_DX for k in range(_QUAD_N)]
_QUAD_PDET = [_sigmoid((x - X0) / DETECTION_SCALE) for x in _QUAD_POINTS]


def _analytic_alpha(mu, sigma):
    return _QUAD_DX * sum(
        pdet * math.exp(_lognormal_logpdf(x, mu, sigma))
        for x, pdet in zip(_QUAD_POINTS, _QUAD_PDET)
    )


def _mc_injections(n_injections, seed=42):
    rng = random.Random(seed)
    return [
        (x := rng.uniform(X_MIN, X_MAX), _sigmoid((x - X0) / DETECTION_SCALE))
        for _ in range(n_injections)
    ]


def _mc_alpha(mu, sigma, injections):
    # Injection proposal is the same uniform pi_0 used for event posterior samples.
    return (X_MAX - X_MIN) * sum(
        pdet * math.exp(_lognormal_logpdf(x, mu, sigma)) for x, pdet in injections
    ) / len(injections)


def _fit_grid(samples, selection="none", injections=None):
    best = (-math.inf, None)
    for mu, sigma in _GRID:
        log_likelihood = 0.0
        for event_samples in samples:
            log_weights = [
                _lognormal_logpdf(x, mu, sigma) + LOG_PRIOR_WIDTH
                for x in event_samples
            ]
            log_likelihood += _logsumexp(log_weights) - math.log(len(log_weights))

        if selection == "analytic":
            log_likelihood -= len(samples) * math.log(_analytic_alpha(mu, sigma))
        elif selection == "mc":
            log_likelihood -= len(samples) * math.log(_mc_alpha(mu, sigma, injections))
        elif selection != "none":
            raise ValueError(selection)

        if log_likelihood > best[0]:
            best = (log_likelihood, (mu, sigma))
    return best[1]


def _distance_from_truth(theta):
    mu, sigma = theta
    return math.hypot(mu - TRUE_MU, sigma - TRUE_SIGMA)


def test_omitting_selection_correction_biases_toy_recovery(toy_samples):
    recovered = _fit_grid(toy_samples, selection="none")

    assert recovered[0] > TRUE_MU + 0.12
    assert _distance_from_truth(recovered) > 0.14


def test_analytic_selection_correction_recovers_injected_population(toy_samples):
    recovered = _fit_grid(toy_samples, selection="analytic")

    assert recovered[0] == pytest.approx(TRUE_MU, abs=0.08)
    assert recovered[1] == pytest.approx(TRUE_SIGMA, abs=0.08)


def test_monte_carlo_selection_correction_improves_with_more_injections(toy_samples):
    small_injection_recovery = _fit_grid(
        toy_samples, selection="mc", injections=_mc_injections(120)
    )
    large_injection_recovery = _fit_grid(
        toy_samples, selection="mc", injections=_mc_injections(1200)
    )

    assert _distance_from_truth(large_injection_recovery) < _distance_from_truth(
        small_injection_recovery
    )
    assert large_injection_recovery[0] == pytest.approx(TRUE_MU, abs=0.10)
    assert large_injection_recovery[1] == pytest.approx(TRUE_SIGMA, abs=0.10)
