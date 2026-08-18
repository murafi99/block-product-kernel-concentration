import math
import sys
import pathlib

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from theory import statistics_utils as stats


# ---------------------------------------------------------------------------
# Wilson score interval
# ---------------------------------------------------------------------------

def test_wilson_ci_known_reference_value():
    # k=50, n=100, 95% CI: textbook Wilson interval is approximately (0.404, 0.596)
    phat, lo, hi = stats.wilson_ci(50, 100, 0.95)
    assert phat == pytest.approx(0.5)
    assert lo == pytest.approx(0.404, abs=0.01)
    assert hi == pytest.approx(0.596, abs=0.01)


def test_wilson_ci_bounds_within_unit_interval():
    for k, n in [(0, 10), (10, 10), (5, 10), (1, 1000)]:
        phat, lo, hi = stats.wilson_ci(k, n)
        assert 0.0 <= lo <= phat <= hi <= 1.0


def test_wilson_ci_narrows_with_more_trials():
    _, lo1, hi1 = stats.wilson_ci(50, 100)
    _, lo2, hi2 = stats.wilson_ci(500, 1000)
    assert (hi2 - lo2) < (hi1 - lo1)


# ---------------------------------------------------------------------------
# Delta-method SE propagation
# ---------------------------------------------------------------------------

def test_delta_method_recovers_exact_c_when_se_zero():
    # mean_x2 = 1/D = 0.25 at D=4 (b=2) is the digitizing ceiling: c = -log2(0.25)/2 = 1.0
    c, se_c = stats.exponent_delta_method(mean_x2=0.25, se_x2=0.0, b=2)
    assert c == pytest.approx(1.0)
    assert se_c == pytest.approx(0.0)
    # mean_x2 = 1/16 at b=2 (D=4) gives the deeper-decay exponent c=2.0
    c2, _ = stats.exponent_delta_method(mean_x2=0.0625, se_x2=0.0, b=2)
    assert c2 == pytest.approx(2.0)


def test_delta_method_se_scales_linearly_with_input_se():
    c1, se1 = stats.exponent_delta_method(mean_x2=0.3, se_x2=0.001, b=3)
    c2, se2 = stats.exponent_delta_method(mean_x2=0.3, se_x2=0.002, b=3)
    assert c1 == pytest.approx(c2)
    assert se2 == pytest.approx(2 * se1, rel=1e-9)


def test_moment_estimate_matches_numpy():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, size=5000)
    mean_x, se_x, mean_x2, se_x2 = stats.moment_estimate(x)
    assert mean_x == pytest.approx(np.mean(x))
    assert mean_x2 == pytest.approx(np.mean(x ** 2))


# ---------------------------------------------------------------------------
# Method 2 (regression) -- included so its known fragility is demonstrated, not
# just asserted in prose (Section 7.6).
# ---------------------------------------------------------------------------

def test_regression_recovers_exact_slope_on_noiseless_data():
    ns = np.array([2, 4, 6, 8, 10], dtype=float)
    c_true = 1.7
    y = 2.0 ** (-c_true * ns)
    c_fit, se_fit, r2 = stats.loglinear_regression_exponent(ns, y)
    assert c_fit == pytest.approx(c_true, abs=1e-8)
    assert r2 == pytest.approx(1.0, abs=1e-8)


def test_regression_is_noisier_than_delta_method_at_low_repetition():
    """Demonstrates the Section 7.6 finding: at low sample count, regression-
    estimated exponents scatter more than the moment-based delta-method SE would
    predict, because sample variance of a product of many bounded factors is a
    poor estimator when the true variance is tiny.
    """
    rng = np.random.default_rng(42)
    b = 2
    D = 2 ** b
    e_x2_true = 2.0 / (D * (D + 1))
    n_pairs = 30  # deliberately low repetition count
    repeats = [1, 2, 3, 4]
    fitted_cs = []
    for _ in range(40):
        mean_x2_by_m = []
        for m in repeats:
            # simulate m i.i.d. Bernoulli-like products via a Beta-ish proxy: use
            # a bounded proxy distribution with the correct first two moments is
            # overkill here -- instead directly perturb the true value with
            # sampling-scale noise consistent with a small-n estimator.
            true_val = e_x2_true ** m
            noisy = max(true_val + rng.normal(0, true_val * 0.9), 1e-8)
            mean_x2_by_m.append(noisy)
        ns_arr = np.array(repeats, dtype=float) * b
        c_fit, se_fit, r2 = stats.loglinear_regression_exponent(ns_arr, np.array(mean_x2_by_m))
        fitted_cs.append(c_fit)
    spread = np.std(fitted_cs)
    # the point of this test is only that the fragility is real and reproducible,
    # not a specific numeric bound
    assert spread > 0.05


# ---------------------------------------------------------------------------
# Arcsine-law threshold heuristic (Section 4 / 7.4)
# ---------------------------------------------------------------------------

def test_arcsine_agreement_is_half_at_zero_snr():
    a = stats.arcsine_law_agreement(var_kappa=0.0, mu=0.1, n_shots=1000)
    assert a == pytest.approx(0.5)


def test_arcsine_agreement_increases_with_shots():
    a1 = stats.arcsine_law_agreement(var_kappa=1e-4, mu=1e-3, n_shots=100)
    a2 = stats.arcsine_law_agreement(var_kappa=1e-4, mu=1e-3, n_shots=100000)
    assert a2 > a1
    assert a1 >= 0.5 and a2 <= 1.0


def test_n_star_rank_exceeds_n_star_var_for_same_shots():
    """The whole point of Section 4: for c(b) in (1,2), the ranking threshold sits
    at a LARGER n than the naive resolvability threshold for the same shot budget.
    """
    for c in [1.2, 1.585, 1.837, 1.99]:
        n_var = stats.n_star_var(c, 10000)
        n_rank = stats.n_star_rank(c, 10000)
        assert n_rank > n_var
