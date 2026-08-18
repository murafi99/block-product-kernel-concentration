"""
The exact 2-design concentration exponent c(b) = [(b-1) + log2(2**b+1)] / b.

Defined via E[kappa^2] = 2**(-c*n) at n=b (single block = m=1), c(b) is the per-qubit
decay exponent of an exact per-block 2-design ensemble. c(b) is strictly increasing
in b, c(1) = log2(3), c(b) -> 2 as b -> infinity. The sharp ceiling for ANY 1-design
ensemble of block size b is also c(b) (Theorem 1, via Nakata et al. 2024's Prop. 1
for the lower-second-moment / upper-exponent bound) -- the floor is c=1 for every b
(digitizing, Corollary 1.1).
"""
from __future__ import annotations
import math
from fractions import Fraction

FLOOR_C = 1.0          # digitizing, exact for every finite n, every b
LIMIT_C = 2.0           # c(b) -> 2 as b -> infinity; never reached at finite b


def c_of_b(b: int) -> float:
    """Exact per-qubit 2-design exponent for block size b."""
    if b < 1:
        raise ValueError("block size b must be >= 1")
    D = 2 ** b
    return ((b - 1) + math.log2(D + 1)) / b


def c_of_b_symbolic(b: int) -> str:
    """Human-readable symbolic form, e.g. '[3 + log2(17)] / 4' for b=4."""
    D = 2 ** b
    if b == 1:
        return f"log2({D + 1})"
    return f"[{b - 1} + log2({D + 1})] / {b}"


# Exact symbolic values recorded in the frozen contract (item 11), before decimal
# approximation. Kept as a literal table (not just c_of_b(b) at runtime) so a test
# can catch any future accidental change to the formula.
FROZEN_CONTRACT_VALUES = {
    1: 1.5849625007,
    2: 1.6609640474,
    3: 1.7233083338,
    4: 1.7718657103,
    5: 1.8088788239,
    6: 1.8370613022,
}


def table(b_max: int = 6) -> dict[int, float]:
    """c(b) for b = 1..b_max."""
    return {b: c_of_b(b) for b in range(1, b_max + 1)}


def is_strictly_increasing(b_max: int = 20) -> bool:
    vals = [c_of_b(b) for b in range(1, b_max + 1)]
    return all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))


def converges_to_two(b_max: int = 4096, tol: float = 1e-3) -> bool:
    return abs(c_of_b(b_max) - LIMIT_C) < tol


if __name__ == "__main__":
    for b, exact in FROZEN_CONTRACT_VALUES.items():
        computed = c_of_b(b)
        print(f"b={b}: c(b)={computed:.10f}  symbolic={c_of_b_symbolic(b)}  "
              f"contract={exact:.10f}  diff={computed - exact:+.2e}")
