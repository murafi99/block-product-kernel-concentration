"""
Experiment B -- independent-quadruplet ranking agreement (Section 7.3, Figure 3).

For a given (block size b or 'global', n qubits, N_shots), draws a genuinely
independent quadruplet (x1,x2,x3,x4) EVERY trial (no batch reuse -- this is the
methodological upgrade over the earlier shared-batch design, Section 11), computes
the exact kappa(x1,x2) and kappa(x3,x4) via the block-product formula, corrupts both
with independent Binomial(N_shots, kappa)/N_shots shot noise, and checks whether the
sign of the noisy difference matches the sign of the true difference. Reports the
Wilson 95% CI ranking-agreement curve A(n).

For block size b, n qubits means m=n/b independent blocks multiplied together
(kappa = prod of m iid block draws); for 'global', n IS the block size (single
b=n block, no product) -- capped at global_max_n for compute-budget reasons
(matches Section 7.3).

Usage:
    python experiments/ranking_agreement.py --config configs/ranking.yaml
"""
from __future__ import annotations
import argparse
import json
import pathlib
import sys

import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from theory import simcore, statistics_utils as stats


def sample_kappa(b: int, m: int, depth: int, domain, rng: np.random.Generator,
                  cache: dict) -> float:
    """kappa for one draw of (x, x') under an m-block product of block size b.
    `cache` holds per-(b,) fixed mixing parameters, reused across calls within a
    run so all draws come from the same circuit-instance ensemble.
    """
    lo, hi = domain
    kappa = 1.0
    for j in range(m):
        key = (b, j)
        if key not in cache:
            cache[key] = simcore.sample_mix_params(b, depth, rng)
        mp = cache[key]
        x = rng.uniform(lo, hi, size=b)
        xp = rng.uniform(lo, hi, size=b)
        psi = simcore.block_state(x, b, depth, mp)
        phi = simcore.block_state(xp, b, depth, mp)
        kappa *= simcore.fidelity(psi, phi)
    return kappa


def shot_noisy_estimate(kappa: float, n_shots: int, rng: np.random.Generator) -> float:
    """Binomial(N_shots, kappa)/N_shots -- the standard shot-noise model for a
    fidelity estimated from repeated Bernoulli(kappa) readouts.
    """
    kappa = min(max(kappa, 0.0), 1.0)
    return rng.binomial(n_shots, kappa) / n_shots


def agreement_at_n(b_or_global: str, b: int, n: int, n_shots: int, trials: int,
                    depth: int, domain, rng: np.random.Generator) -> dict:
    if b_or_global == "global":
        m, block = 1, n
    else:
        if n % b != 0:
            return None
        m, block = n // b, b

    cache: dict = {}
    successes = 0
    for _ in range(trials):
        k1 = sample_kappa(block, m, depth, domain, rng, cache)
        k2 = sample_kappa(block, m, depth, domain, rng, cache)
        true_sign = np.sign(k1 - k2)
        if true_sign == 0:
            continue
        k1_hat = shot_noisy_estimate(k1, n_shots, rng)
        k2_hat = shot_noisy_estimate(k2, n_shots, rng)
        est_sign = np.sign(k1_hat - k2_hat)
        if est_sign == true_sign:
            successes += 1
    phat, lo, hi = stats.wilson_ci(successes, trials)
    return dict(n=n, n_shots=n_shots, trials=trials, successes=successes,
                agreement=phat, ci_lo=lo, ci_hi=hi)


def main(config_path: str) -> dict:
    cfg = yaml.safe_load(open(config_path))
    rng = np.random.default_rng(20260818)
    domain = (-np.pi, np.pi)
    results = {}

    series = [("global", None)] + [("block", b) for b in cfg["block_sizes"]]
    for kind, b in series:
        label = "global" if kind == "global" else f"block_b{b}"
        curve = []
        for n in cfg["n_values"]:
            if kind == "global" and n > cfg["global_max_n"]:
                continue
            if kind == "block" and n % b != 0:
                continue
            r = agreement_at_n(kind, b, n, cfg["n_shots"], cfg["trials_per_n"],
                                cfg["depth"], domain, rng)
            if r:
                curve.append(r)
                print(f"{label:10s} n={n:3d}  A={r['agreement']:.3f} "
                      f"[{r['ci_lo']:.3f},{r['ci_hi']:.3f}]")
        results[label] = curve

    out_json = pathlib.Path(cfg["output_json"])
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))

    try:
        plot_figure3(results, cfg["output_figure"])
    except Exception as e:  # matplotlib absence should not fail the run
        print(f"(figure generation skipped: {e})")
    return results


def plot_figure3(results: dict, out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for label, curve in results.items():
        if not curve:
            continue
        ns = [c["n"] for c in curve]
        a = [c["agreement"] for c in curve]
        lo = [max(0.0, c["agreement"] - c["ci_lo"]) for c in curve]
        hi = [max(0.0, c["ci_hi"] - c["agreement"]) for c in curve]
        ax.errorbar(ns, a, yerr=[lo, hi], marker="o", capsize=3, label=label)
    ax.axhline(0.75, ls="--", color="gray", lw=1)
    ax.axhline(0.5, ls=":", color="gray", lw=1)
    ax.set_xlabel("n"); ax.set_ylabel("ranking agreement A")
    ax.set_title("Ranking agreement A(n) (independent quadruplets, Wilson 95% CI)")
    ax.legend()
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/ranking.yaml")
    args = p.parse_args()
    main(args.config)
