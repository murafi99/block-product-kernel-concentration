"""
Exact moment formulas for block-product fidelity quantum kernels.

Every function here is a closed-form expression, not an estimate. Simulation code
in experiments/ compares its Monte Carlo estimates against these; tests/test_theory.py
checks the formulas against the symbolic values recorded in the frozen contract.

Notation follows the paper: D = 2**b (per-block Hilbert space dimension), m = number
of blocks, n = m*b (total qubits). X_i = |<psi_i(x)|psi_i(x')>|^2 is the per-block
fidelity; kappa = prod_i X_i is the full kernel.
"""
from __future__ import annotations
import math


def block_dim(b: int) -> int:
    """D = 2**b."""
    if b < 1:
        raise ValueError("block size b must be >= 1")
    return 2 ** b


def first_moment_1design(b: int) -> float:
    """E[X_i] under a per-block 1-design (Lemma 3): exactly 1/D.

    Requires only A3 (average block state maximally mixed) -- not a 2-design.
    """
    D = block_dim(b)
    return 1.0 / D


def second_moment_2design(b: int) -> float:
    """E[X_i^2] for an exact per-block 2-design (Theorem 2): exactly 2/(D(D+1)).

    Derivation: for Haar |psi> in C^D and fixed |phi>, X = |<psi|phi>|^2 ~ Beta(1, D-1),
    whose second moment is (1*2)/(D*(D+1)). A unitary 2-design reproduces this exactly
    by definition. At m=1 (single global block) this is Thanasilp et al. (2024)
    Appendix B's beta_Haar = 1/(2**(n-1)*(2**n+1)); it is not re-derived as new here.
    """
    D = block_dim(b)
    return 2.0 / (D * (D + 1))


def nakata_lower_bound(b: int) -> float:
    """Sharp lower bound on E[X_i^2] over ALL 1-design ensembles of block size b.

    This is Nakata et al. (2024), Proposition 1, cited per the frozen contract and
    not re-derived: 2-design ensembles minimize the second moment among all
    distributions matching the 1-design (first-moment) condition. Equal to
    second_moment_2design(b) -- kept as a separate name because the two facts
    (2-design achieves the value; no 1-design can go lower) have different proofs
    and different citations.
    """
    return second_moment_2design(b)


def elementary_upper_bound(b: int) -> float:
    """E[X_i^2] <= 1/D for any 1-design (elementary: X_i<=1 => X_i^2<=X_i => E[X_i^2]<=E[X_i]=1/D).

    Achieved with equality exactly by digitizing/basis-encoding ensembles.
    """
    return first_moment_1design(b)


def jensen_lower_bound(b: int) -> float:
    """The classical, LOOSER lower bound E[X_i^2] >= (E[X_i])^2 = 1/D^2.

    Included only for comparison against nakata_lower_bound; the paper's sharp
    result supersedes this with the tighter [1, c(b)] interval instead of the
    naive [1, 2] a pure-Jensen argument would give.
    """
    D = block_dim(b)
    return 1.0 / (D * D)


def exponent_from_second_moment(e_x2: float, b: int) -> float:
    """c such that e_x2 = 2**(-c*b), i.e. the per-qubit exponent implied by a
    measured or exact per-block second moment at block size b.
    """
    if e_x2 <= 0:
        raise ValueError("second moment must be positive")
    return -math.log2(e_x2) / b


def var_kappa_2design(b: int, m: int) -> float:
    """Exact Var[kappa] for m independent, per-block-2-design blocks of size b (Theorem 2).

    Var[kappa] = [2/(D(D+1))]^m - D^(-2m), exact, not asymptotic.
    """
    D = block_dim(b)
    e_x2 = second_moment_2design(b)
    return e_x2 ** m - D ** (-2.0 * m)


def var_kappa_general(e_x2: float, D: float, m: int) -> float:
    """Var[kappa] given an arbitrary (measured or exact) per-block E[X^2] and the
    1-design first moment 1/D, for m i.i.d. independent blocks (Lemma 2).
    """
    return e_x2 ** m - (1.0 / D) ** (2.0 * m)


def digitizing_second_moment(b: int) -> float:
    """Exact E[X_i^2] for a Bernoulli/digitizing block ensemble: exactly 1/D
    (Corollary 1.1 -- equality case of the elementary upper bound, exact for
    every finite n, not just asymptotically).
    """
    return elementary_upper_bound(b)


def sanity_check_interval(b: int, atol: float = 1e-12) -> bool:
    """True iff nakata_lower_bound(b) <= elementary_upper_bound(b) with the
    expected strict inequality for D>1 (they coincide only in the trivial D=1 case).
    """
    lo = nakata_lower_bound(b)
    hi = elementary_upper_bound(b)
    return lo <= hi + atol
