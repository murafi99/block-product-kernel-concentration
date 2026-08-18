"""
Experiment D -- controlled exponent interpolation (Section 7.5).

Two candidate ways to build a 1-design ensemble that interpolates, with a fraction
p, between digitizing (c=1) and a 2-design (c=c(b)) at fixed block size b. Both are
valid 1-designs; they give DIFFERENT formulas for c(p) despite both naively
describing "a p-mixture of the same two ingredients" -- this is the trap the
project's construction-design protocol warns against, and it is documented here
rather than left implicit.

Construction 1 (static block-type partition): with probability p a block is
"digitizing-type" for its entire lifetime (drawn once per block, not per use);
with probability 1-p it is "2-design-type." Since the block's TYPE is fixed and
only the state changes per data draw, cross terms between a digitizing draw and a
2-design draw never arise within a single block's statistics -- the two types
simply average linearly in Var[kappa]'s controlling second moment:
    E[X^2](p) = p * (1/D) + (1-p) * (2/(D(D+1)))   =>   c(p) = p*1 + (1-p)*c(b)   (EXACTLY linear)

Construction 2 (per-use random coin per block): every time the block is used, an
independent coin decides digitizing-vs-2-design for THAT draw. Now cross terms
appear: a digitizing draw of x against a 2-design draw of x' behaves exactly like a
fixed vector against an independent Haar state (Beta(1,D-1) statistics) REGARDLESS
of what the fixed vector is, contributing MORE to E[X^2] than naive linear mixing
would suggest. This gives a materially different, quadratic-in-p formula. It is
derived analytically below; simulating it is flagged as future work (matching the
paper's own reporting -- see run_construction2(simulate=False) default).

Usage:
    python experiments/interpolation.py --config configs/... (none required; runs standalone)
"""
from __future__ import annotations
import argparse
import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from theory import moments, simcore, statistics_utils as stats


def construction1_c_of_p(b: int, p: float) -> float:
    """Exact linear formula, Construction 1 (static block-type partition)."""
    D = moments.block_dim(b)
    e_x2 = p * (1.0 / D) + (1 - p) * moments.second_moment_2design(b)
    return moments.exponent_from_second_moment(e_x2, b)


def construction2_c_of_p(b: int, p: float) -> float:
    """Exact quadratic formula, Construction 2 (per-use random coin per block).

    Derivation: let each draw independently be "digitizing" (prob p) or "2-design"
    (prob 1-p). For a pair (x,x'), four cases arise:
      (dig,dig)   prob p^2:       X ~ Bernoulli(1/D), E[X^2] = 1/D
      (2des,2des) prob (1-p)^2:   E[X^2] = 2/(D(D+1))
      (dig,2des) or (2des,dig)   prob 2p(1-p): a FIXED digitizing basis state
                  against an INDEPENDENT Haar/2-design state is itself
                  Beta(1,D-1)-distributed regardless of the fixed vector, so
                  E[X^2] in this cross case is ALSO 2/(D(D+1)), not some new
                  quantity and not simply "absent" as a naive linear-mixture
                  reading would assume.
    Net: E[X^2](p) = p^2 * (1/D) + (1 - p^2) * (2/(D(D+1)))  -- weight p^2 on the
    digitizing component, not p -- a materially different formula from
    Construction 1 despite an identical-looking p-mixture description.
    """
    D = moments.block_dim(b)
    e_x2 = (p ** 2) * (1.0 / D) + (1 - p ** 2) * moments.second_moment_2design(b)
    return moments.exponent_from_second_moment(e_x2, b)


def simulate_construction1(b: int, p: float, depth: int, n_pairs: int,
                            rng: np.random.Generator, domain=(-math.pi, math.pi)) -> dict:
    """Simulate Construction 1 via the reliable moment-based method (Method 1).
    A block is digitizing-type with probability p (decided ONCE, before sampling
    any data), else a 2-design-type block built from the standard circuit family.
    """
    is_digitizing = rng.uniform() < p
    if is_digitizing:
        labels_x = rng.integers(0, moments.block_dim(b), n_pairs)
        labels_xp = rng.integers(0, moments.block_dim(b), n_pairs)
        fidelities = (labels_x == labels_xp).astype(float)
    else:
        mix_params = simcore.sample_mix_params(b, depth, rng)
        lo, hi = domain
        xs = rng.uniform(lo, hi, size=(n_pairs, b))
        xps = rng.uniform(lo, hi, size=(n_pairs, b))
        fidelities = np.empty(n_pairs)
        for i in range(n_pairs):
            psi = simcore.block_state(xs[i], b, depth, mix_params)
            phi = simcore.block_state(xps[i], b, depth, mix_params)
            fidelities[i] = simcore.fidelity(psi, phi)
    mean_x, se_x, mean_x2, se_x2 = stats.moment_estimate(fidelities)
    c_meas, se_c = stats.exponent_delta_method(mean_x2, se_x2, b)
    return dict(p=p, is_digitizing_instance=bool(is_digitizing), mean_x2=mean_x2, se_x2=se_x2,
                c_measured=c_meas, se_c=se_c, c_exact=construction1_c_of_p(b, p))


def main(b: int = 2, depth: int = 32, n_pairs: int = 20000,
         p_values=(0.0, 0.25, 0.5, 0.75, 1.0), out_json: str = "data/processed/expD.json",
         out_table: str = "tables/table_experiment_D.md") -> dict:
    rng = np.random.default_rng(20260818)
    results = {"construction1": [], "construction2_analytic_only": []}

    for p in p_values:
        r = simulate_construction1(b, p, depth, n_pairs, rng)
        deviation = r["c_measured"] - r["c_exact"]
        r["deviation"] = deviation
        results["construction1"].append(r)
        c2 = construction2_c_of_p(b, p)
        results["construction2_analytic_only"].append(dict(p=p, c_exact=c2, note="derived, not simulated"))
        print(f"p={p:.2f}  C1: measured={r['c_measured']:.5f} exact={r['c_exact']:.5f} "
              f"dev={deviation:+.4f}   |   C2 (analytic only): {c2:.5f}")

    out_json_p = pathlib.Path(out_json)
    out_json_p.parent.mkdir(parents=True, exist_ok=True)
    out_json_p.write_text(json.dumps(results, indent=2))

    lines = ["| p | C1 measured | C1 exact | deviation | C2 exact (not simulated) |",
             "|---:|---:|---:|---:|---:|"]
    for r, r2 in zip(results["construction1"], results["construction2_analytic_only"]):
        lines.append(f"| {r['p']:.2f} | {r['c_measured']:.5f} +/- {r['se_c']:.5f} | "
                      f"{r['c_exact']:.5f} | {r['deviation']:+.4f} | {r2['c_exact']:.5f} |")
    out_table_p = pathlib.Path(out_table)
    out_table_p.parent.mkdir(parents=True, exist_ok=True)
    out_table_p.write_text("\n".join(lines) + "\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--b", type=int, default=2)
    parser.add_argument("--depth", type=int, default=32)
    parser.add_argument("--n_pairs", type=int, default=20000)
    args = parser.parse_args()
    main(b=args.b, depth=args.depth, n_pairs=args.n_pairs)
