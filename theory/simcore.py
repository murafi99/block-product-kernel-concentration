"""
Minimal, dense, gate-level statevector simulator for the block circuit family used
throughout the experiments: data-reuploading RX(x) + fixed random RY/RZ mixing +
brick-wall CNOT ring, per block.

Deliberately not a wrapped external library -- every gate is applied explicitly so
the block-b == n global-case reduction can be checked to machine precision
(tests/test_theory.py::test_block_n_equals_global).

Because the theory (Lemma 1/2) only needs the distribution of a SINGLE block's
fidelity X_i, every experiment simulates one b-qubit block at a time (2**b <= 64 for
b<=6) and combines blocks analytically via the exact product formulas in
theory/moments.py -- never by brute-force simulating the full n-qubit register.
"""
from __future__ import annotations
import numpy as np

_I2 = np.eye(2, dtype=complex)


def rx(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(theta: float) -> np.ndarray:
    e = np.exp(-1j * theta / 2)
    return np.array([[e, 0], [0, np.conj(e)]], dtype=complex)


def apply_1q(state: np.ndarray, gate: np.ndarray, q: int, n: int) -> np.ndarray:
    """Apply a 2x2 gate to qubit q (0-indexed) of an n-qubit statevector."""
    psi = state.reshape([2] * n)
    psi = np.tensordot(gate, psi, axes=([1], [q]))
    psi = np.moveaxis(psi, 0, q)
    return psi.reshape(-1)


def apply_cnot(state: np.ndarray, control: int, target: int, n: int) -> np.ndarray:
    psi = state.reshape([2] * n)
    psi = np.moveaxis(psi, [control, target], [0, 1])
    out = psi.copy()
    # flip target where control == 1
    out[1, 0] = psi[1, 1]
    out[1, 1] = psi[1, 0]
    out = np.moveaxis(out, [0, 1], [control, target])
    return out.reshape(-1)


def brickwall_cnot_ring(state: np.ndarray, n: int) -> np.ndarray:
    """One brick-wall layer of CNOTs arranged in a ring: (0,1),(2,3),... then
    (1,2),(3,4),...,(n-1,0). For n==1 this is a no-op (no entangling gate exists
    for a single qubit -- this is the mechanism behind the b=1 structural finding,
    Corollary 1.2).
    """
    if n < 2:
        return state
    for start in (0, 1):
        for c in range(start, n, 2):
            t = (c + 1) % n
            if t != c:
                state = apply_cnot(state, c, t, n)
    return state


def block_state(x: np.ndarray, b: int, depth: int, mix_params: np.ndarray) -> np.ndarray:
    """Build the final statevector of a b-qubit block under `depth` layers of
    [RX(x_k) data-reuploading] -> [fixed random RY(theta)/RZ(phi) mixing] ->
    [brick-wall CNOT ring].

    x: length-b array of data features (reused identically every layer -- data
       re-uploading), one component per qubit.
    mix_params: shape (depth, b, 2) fixed random angles [theta_ry, phi_rz] per
       layer per qubit, drawn once per circuit instance (seeded) and held fixed
       across all data samples for that instance.
    """
    assert x.shape == (b,)
    assert mix_params.shape == (depth, b, 2)
    state = np.zeros(2 ** b, dtype=complex)
    state[0] = 1.0
    for layer in range(depth):
        for q in range(b):
            state = apply_1q(state, rx(x[q]), q, b)
        for q in range(b):
            theta, phi = mix_params[layer, q]
            state = apply_1q(state, ry(theta), q, b)
            state = apply_1q(state, rz(phi), q, b)
        state = brickwall_cnot_ring(state, b)
    return state


def fidelity(psi: np.ndarray, phi: np.ndarray) -> float:
    return float(np.abs(np.vdot(psi, phi)) ** 2)


def average_state_purity(b: int, depth: int, mix_params: np.ndarray, rng: np.random.Generator,
                          n_samples: int = 4000, domain=(-np.pi, np.pi)) -> float:
    """Tr[rho_bar^2] estimated from n_samples data draws -- the diagnostic used in
    Section 7.1 to check assumption A3 before trusting any exponent comparison.
    """
    D = 2 ** b
    rho_bar = np.zeros((D, D), dtype=complex)
    lo, hi = domain
    for _ in range(n_samples):
        x = rng.uniform(lo, hi, size=b)
        psi = block_state(x, b, depth, mix_params)
        rho_bar += np.outer(psi, psi.conj())
    rho_bar /= n_samples
    return float(np.real(np.trace(rho_bar @ rho_bar)))


def sample_mix_params(b: int, depth: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(0, 2 * np.pi, size=(depth, b, 2))
