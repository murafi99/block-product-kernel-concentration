"""
Statistics helpers shared across experiments.

Method 1 (direct moment estimation, primary throughout this project) needs delta-
method SE propagation from E[X], E[X^2] to the exponent c. Method 2 (log-linear
regression) is implemented too, but flagged as unreliable at low repetition count
for products of many bounded factors (Section 7.6 / tests/test_statistics.py).
"""
from __future__ import annotations
import math
import numpy as np


def moment_estimate(samples: np.ndarray) -> tuple[float, float, float, float]:
    """Sample mean/SE of X and X^2 from an array of per-block fidelities.

    Returns (mean_x, se_x, mean_x2, se_x2).
    """
    n = samples.size
    x = samples
    x2 = samples ** 2
    mean_x, se_x = float(np.mean(x)), float(np.std(x, ddof=1) / math.sqrt(n))
    mean_x2, se_x2 = float(np.mean(x2)), float(np.std(x2, ddof=1) / math.sqrt(n))
    return mean_x, se_x, mean_x2, se_x2


def exponent_delta_method(mean_x2: float, se_x2: float, b: int) -> tuple[float, float]:
    """c = -log2(mean_x2)/b, with SE propagated via the delta method:
    Var[c] approx (d c / d mean_x2)^2 * Var[mean_x2], d c/d(mean_x2) = -1/(b * ln2 * mean_x2).
    """
    c = -math.log2(mean_x2) / b
    se_c = se_x2 / (b * math.log(2) * mean_x2)
    return c, se_c


def loglinear_regression_exponent(ns: np.ndarray, mean_x2_by_n: np.ndarray) -> tuple[float, float, float]:
    """Method 2 (secondary/cross-check): fit log2(E[kappa^2]) = -c*n + const across
    several n via ordinary least squares. Returns (slope_c, se_slope, r_squared).

    KNOWN FRAGILE at low repetition count -- see Section 7.6 / test_statistics.py.
    Do not treat this as more reliable than Method 1 without a large repetition count.
    """
    y = np.log2(mean_x2_by_n)
    x = np.asarray(ns, dtype=float)
    n = x.size
    xbar, ybar = x.mean(), y.mean()
    sxx = np.sum((x - xbar) ** 2)
    sxy = np.sum((x - xbar) * (y - ybar))
    slope = sxy / sxx
    intercept = ybar - slope * xbar
    resid = y - (slope * x + intercept)
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - ybar) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    dof = max(n - 2, 1)
    s2 = ss_res / dof
    se_slope = math.sqrt(s2 / sxx) if sxx > 0 else float("nan")
    return -slope, se_slope, r2


def wilson_ci(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion. Returns (phat, lo, hi).

    Used throughout Experiment B/C for ranking-agreement confidence intervals
    (no batch reuse, independent quadruplets per trial).
    """
    if trials == 0:
        return float("nan"), float("nan"), float("nan")
    z = {0.90: 1.6448536269514722, 0.95: 1.959963984540054, 0.99: 2.5758293035489004}[confidence]
    n = trials
    phat = successes / n
    denom = 1 + z * z / n
    centre = phat + z * z / (2 * n)
    adj = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    lo = (centre - adj) / denom
    hi = (centre + adj) / denom
    # The Wilson interval always contains phat by construction; clamp against
    # floating-point rounding at the phat in {0,1} boundary (e.g. hi computing to
    # 0.999999999999999 instead of 1.0) rather than let a spurious rounding error
    # violate lo <= phat <= hi.
    lo = min(max(0.0, lo), phat)
    hi = max(min(1.0, hi), phat)
    return phat, lo, hi


def arcsine_law_agreement(var_kappa: float, mu: float, n_shots: int) -> float:
    """The Gaussian/arcsine-law heuristic for ranking-agreement probability
    (Section 4 of the paper): treats Delta = kappa1-kappa2 and its shot noise as
    independent zero-mean Gaussians and returns P(sign(Delta_hat)=sign(Delta)).

    SNR = sqrt(Var[kappa]/mu) * sqrt(N_shots); agreement = 1/2 + arcsin(rho)/pi
    with rho = SNR / sqrt(1+SNR^2) (equivalently, the standard "probability two
    correlated Gaussians share sign" formula applied to Delta vs Delta+noise).

    Reported as a CALIBRATED HEURISTIC with known error (median ~0.017, worst-case
    ~0.235 against the full phase-diagram grid, Section 7.4) -- not a theorem.
    """
    if var_kappa <= 0 or mu <= 0 or n_shots <= 0:
        return 0.5
    snr = math.sqrt(var_kappa / mu) * math.sqrt(n_shots)
    rho = snr / math.sqrt(1 + snr * snr)
    return 0.5 + math.asin(rho) / math.pi


def n_star_var(c_b: float, n_shots: int) -> float:
    """Naive resolvability threshold n*_var = log2(N)/c(b) (A ~ 0.75, Section 4/7.4)."""
    return math.log2(n_shots) / c_b


def n_star_rank(c_b: float, n_shots: int) -> float:
    """Ranking-agreement threshold n*_rank = log2(N)/(c(b)-1) (A ~ 0.55, Section 4).

    NOTE: corresponds to a materially weaker agreement bar than n_star_var -- see
    Section 7.4 for why the two must not be treated as the same threshold.
    """
    if c_b <= 1:
        return float("inf")
    return math.log2(n_shots) / (c_b - 1)
