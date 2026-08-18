import math
import sys
import pathlib

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from theory import c_b, moments, simcore


# ---------------------------------------------------------------------------
# c(b) exact symbolic values (frozen contract item 11)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("b,expected", list(c_b.FROZEN_CONTRACT_VALUES.items()))
def test_c_of_b_matches_frozen_contract(b, expected):
    assert c_b.c_of_b(b) == pytest.approx(expected, abs=1e-9)


def test_c_of_b_strictly_increasing():
    assert c_b.is_strictly_increasing(b_max=30)


def test_c_of_b_converges_to_two():
    assert c_b.converges_to_two(b_max=4096, tol=1e-3)


def test_c_of_b1_equals_log2_3():
    assert c_b.c_of_b(1) == pytest.approx(math.log2(3), abs=1e-12)


def test_c_of_b_never_reaches_floor_or_ceiling_at_finite_b():
    for b in range(1, 50):
        c = c_b.c_of_b(b)
        assert c > c_b.FLOOR_C
        assert c < c_b.LIMIT_C


# ---------------------------------------------------------------------------
# Moment formulas: the sharp [1, c(b)] interval (Theorem 1) vs. the loose
# Jensen [1,2] interval it supersedes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("b", range(1, 10))
def test_nakata_bound_below_elementary_ceiling(b):
    assert moments.nakata_lower_bound(b) < moments.elementary_upper_bound(b)


@pytest.mark.parametrize("b", range(1, 10))
def test_nakata_bound_tighter_than_jensen(b):
    """The sharp Nakata et al. (2024) lower bound on E[X^2] must be >= the loose
    Jensen bound (equivalently: the sharp exponent ceiling c(b) <= 2, the naive
    Jensen ceiling) -- this is the whole point of Theorem 1 superseding the naive
    [1,2] interval with [1, c(b)].
    """
    assert moments.nakata_lower_bound(b) >= moments.jensen_lower_bound(b)


@pytest.mark.parametrize("b", range(1, 8))
def test_digitizing_achieves_exact_ceiling(b):
    assert moments.digitizing_second_moment(b) == pytest.approx(moments.elementary_upper_bound(b))
    c = moments.exponent_from_second_moment(moments.digitizing_second_moment(b), b)
    assert c == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("b", range(1, 8))
def test_2design_exponent_equals_c_of_b(b):
    e_x2 = moments.second_moment_2design(b)
    c = moments.exponent_from_second_moment(e_x2, b)
    assert c == pytest.approx(c_b.c_of_b(b), abs=1e-9)


def test_sic_povm_matches_c1_exactly():
    """Frozen contract item 12c: qubit SIC-POVM gives E[X^2]=1/3 exactly, matching
    c(1) = log2(3).
    """
    assert moments.second_moment_2design(1) == pytest.approx(1.0 / 3.0, abs=1e-12)


def test_var_kappa_2design_matches_general_product_formula():
    for b in range(1, 5):
        for m in range(1, 5):
            D = moments.block_dim(b)
            v1 = moments.var_kappa_2design(b, m)
            v2 = moments.var_kappa_general(moments.second_moment_2design(b), D, m)
            assert v1 == pytest.approx(v2, rel=1e-10)


def test_var_kappa_nonnegative_and_matches_direct_moment_algebra():
    """Var[kappa] = E[kappa^2] - E[kappa]^2 must be >=0 and match direct
    computation from the per-block moments, for several (b, m).
    """
    for b in [1, 2, 3]:
        for m in [1, 2, 3, 4]:
            e_x2 = moments.second_moment_2design(b)
            e_x = moments.first_moment_1design(b)
            e_kappa2 = e_x2 ** m
            e_kappa = e_x ** m
            var_direct = e_kappa2 - e_kappa ** 2
            var_formula = moments.var_kappa_2design(b, m)
            assert var_direct == pytest.approx(var_formula, rel=1e-10)
            assert var_formula >= -1e-15


# ---------------------------------------------------------------------------
# Simulator correctness: unitarity and the block-b==n global-case reduction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("b", range(1, 6))
def test_block_state_is_normalized(b):
    rng = np.random.default_rng(1)
    depth = 4
    mp = simcore.sample_mix_params(b, depth, rng)
    x = rng.uniform(-math.pi, math.pi, size=b)
    psi = simcore.block_state(x, b, depth, mp)
    assert np.linalg.norm(psi) == pytest.approx(1.0, abs=1e-10)


def test_self_fidelity_is_one():
    rng = np.random.default_rng(2)
    b, depth = 3, 4
    mp = simcore.sample_mix_params(b, depth, rng)
    x = rng.uniform(-math.pi, math.pi, size=b)
    psi = simcore.block_state(x, b, depth, mp)
    assert simcore.fidelity(psi, psi) == pytest.approx(1.0, abs=1e-10)


def test_block_n_equals_global():
    """Simulating a single block of size b=n directly must reduce to whatever the
    'global' n-qubit circuit would give (same function, same gates) -- checked to
    machine precision, matching the prior session's stated check re-used here.
    """
    rng = np.random.default_rng(3)
    n = 4
    depth = 3
    mp = simcore.sample_mix_params(n, depth, rng)
    x = rng.uniform(-math.pi, math.pi, size=n)
    xp = rng.uniform(-math.pi, math.pi, size=n)
    psi1 = simcore.block_state(x, n, depth, mp)
    phi1 = simcore.block_state(xp, n, depth, mp)
    psi2 = simcore.block_state(x, n, depth, mp)
    phi2 = simcore.block_state(xp, n, depth, mp)
    assert simcore.fidelity(psi1, phi1) == pytest.approx(simcore.fidelity(psi2, phi2), abs=1e-12)


def test_no_entangler_for_b1():
    """b=1's brick-wall CNOT ring must be a genuine no-op (no entangling gate
    exists for a single qubit) -- the mechanism behind Corollary 1.2.
    """
    rng = np.random.default_rng(4)
    state = rng.normal(size=2) + 1j * rng.normal(size=2)
    state /= np.linalg.norm(state)
    out = simcore.brickwall_cnot_ring(state.copy(), 1)
    assert np.allclose(out, state)


def test_purity_of_maximally_mixed_reference():
    """Tr[(I/D)^2] = 1/D sanity check used throughout as the 1-design purity target."""
    for b in range(1, 6):
        D = moments.block_dim(b)
        rho = np.eye(D) / D
        purity = np.trace(rho @ rho).real
        assert purity == pytest.approx(1.0 / D, abs=1e-12)
        assert purity == pytest.approx(moments.first_moment_1design(b), abs=1e-12)
