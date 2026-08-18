"""
Experiment A -- block-size exponent sweep (Section 7.2 of the paper).

For each block size b, draws n_pairs of independent (x, x') per circuit-instance
seed, computes the per-block fidelity X = |<psi(x)|psi(x')>|^2, estimates E[X^2] by
direct sample mean (Method 1), propagates the SE to the exponent c via the delta
method, and separately estimates the average-state purity Tr[rho_bar^2] to check
assumption A3 *before* trusting any comparison to theory (Section 7.1's protocol
requirement).

b=1 is NOT included in the main sweep: per Corollary 1.2, a single continuous data
parameter traces a measure-zero curve on the Bloch sphere and cannot reach an exact
2-design at any depth. It is instead checked against two analytically-known
constructions (bare RX(x), matching Thanasilp et al. Prop. 1; and digitizing).

Usage:
    python experiments/exponent_sweep.py --config configs/exponent_sweep.yaml
"""
from __future__ import annotations
import argparse
import json
import math
import pathlib
import sys

import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from theory import c_b, moments, simcore, statistics_utils as stats


def run_one_seed(b: int, depth: int, n_pairs: int, domain, purity_samples: int,
                  rng: np.random.Generator) -> dict:
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
    c_measured, se_c = stats.exponent_delta_method(mean_x2, se_x2, b)

    purity = simcore.average_state_purity(b, depth, mix_params, rng, n_samples=purity_samples,
                                           domain=domain)
    target_purity = moments.first_moment_1design(b)
    purity_dev = abs(purity - target_purity) / target_purity

    return dict(mean_x=mean_x, se_x=se_x, mean_x2=mean_x2, se_x2=se_x2,
                c_measured=c_measured, se_c=se_c, purity=purity, purity_dev=purity_dev)


def run_regression_crosscheck(b: int, depth: int, n_pairs: int, domain, repeats,
                               rng: np.random.Generator) -> dict:
    """Method 2 (secondary): simulate `repeats` independent copies of the same
    block ensemble multiplied together (an m-block product with all blocks of
    size b), estimate E[kappa^2] at each m via direct sampling, then fit a
    log-linear slope across m. Flagged fragile at low n_pairs -- Section 7.6.
    """
    mean_x2_by_m = []
    for m in repeats:
        mix_params_list = [simcore.sample_mix_params(b, depth, rng) for _ in range(m)]
        lo, hi = domain
        kappas = np.ones(n_pairs)
        for mp in mix_params_list:
            xs = rng.uniform(lo, hi, size=(n_pairs, b))
            xps = rng.uniform(lo, hi, size=(n_pairs, b))
            for i in range(n_pairs):
                psi = simcore.block_state(xs[i], b, depth, mp)
                phi = simcore.block_state(xps[i], b, depth, mp)
                kappas[i] *= simcore.fidelity(psi, phi)
        mean_x2_by_m.append(float(np.mean(kappas ** 2)))
    ns = np.array(repeats, dtype=float) * b
    c_reg, se_reg, r2 = stats.loglinear_regression_exponent(ns, np.array(mean_x2_by_m))
    return dict(repeats=list(repeats), mean_x2_by_m=mean_x2_by_m, c_regression=c_reg,
                se_regression=se_reg, r_squared=r2)


def run_b1_structural(cfg: dict, master_rng: np.random.Generator) -> dict:
    """b=1: bare RX(x) product embedding (Thanasilp et al. Prop. 1 analogue) and
    digitizing, both analytically known and reproduced numerically here.
    """
    n_pairs = cfg["n_pairs"]
    lo, hi = -math.pi, math.pi
    xs = master_rng.uniform(lo, hi, n_pairs)
    xps = master_rng.uniform(lo, hi, n_pairs)
    # bare RX(x): depth=1, no mixing, no entangler (b=1 has none anyway)
    zero_mix = np.zeros((1, 1, 2))
    fid = np.empty(n_pairs)
    for i in range(n_pairs):
        psi = simcore.block_state(np.array([xs[i]]), 1, 1, zero_mix)
        phi = simcore.block_state(np.array([xps[i]]), 1, 1, zero_mix)
        fid[i] = simcore.fidelity(psi, phi)
    mean_x, se_x, mean_x2, se_x2 = stats.moment_estimate(fid)
    c_ry, se_c_ry = stats.exponent_delta_method(mean_x2, se_x2, 1)
    exact_ry_c = -math.log2(3.0 / 8.0)

    # digitizing at D=2: two equally-likely orthogonal basis states
    bins = cfg["digitizing_bins"]
    labels_x = master_rng.integers(0, bins, n_pairs)
    labels_xp = master_rng.integers(0, bins, n_pairs)
    dig_fid = (labels_x == labels_xp).astype(float)
    mean_x2_dig = float(np.mean(dig_fid ** 2))
    c_dig, se_c_dig = stats.exponent_delta_method(mean_x2_dig, float(np.std(dig_fid ** 2, ddof=1) / math.sqrt(n_pairs)), 1)

    return dict(
        ry_only=dict(mean_x2=mean_x2, se_x2=se_x2, c_measured=c_ry, se_c=se_c_ry, c_exact=exact_ry_c),
        digitizing=dict(mean_x2=mean_x2_dig, c_measured=c_dig, se_c=se_c_dig, c_exact=1.0),
    )


def main(config_path: str) -> dict:
    cfg = yaml.safe_load(open(config_path))
    seeds_cfg = yaml.safe_load(open(pathlib.Path(config_path).parent / "seeds.yaml"))
    ss = np.random.SeedSequence(seeds_cfg["master_seed"])

    results = {"b1_structural": run_b1_structural(cfg["b1_structural"], np.random.default_rng(ss.spawn(1)[0]))}

    for b in cfg["block_sizes"]:
        n_seeds = cfg["seeds_per_block"][b]
        per_seed = []
        for s in range(n_seeds):
            rng = np.random.default_rng(ss.spawn(1)[0])
            per_seed.append(run_one_seed(b, cfg["depth"], cfg["n_pairs"], cfg["domain"],
                                          cfg["purity_check_samples"], rng))
        c_vals = np.array([r["c_measured"] for r in per_seed])
        c_theory = c_b.c_of_b(b)
        c_mean = float(np.mean(c_vals))
        c_sem = float(np.std(c_vals, ddof=1) / math.sqrt(len(c_vals))) if len(c_vals) > 1 else per_seed[0]["se_c"]
        rel_err = (c_mean - c_theory) / c_theory
        entry = dict(b=b, c_theory=c_theory, c_measured_mean=c_mean, c_measured_se=c_sem,
                     rel_error=rel_err, purity_dev_mean=float(np.mean([r["purity_dev"] for r in per_seed])),
                     n_seeds=n_seeds, per_seed=per_seed)
        if cfg.get("run_regression_crosscheck"):
            rng = np.random.default_rng(ss.spawn(1)[0])
            entry["regression_crosscheck"] = run_regression_crosscheck(
                b, cfg["depth"], min(cfg["n_pairs"], 4000), cfg["domain"],
                cfg["regression_block_repeats"], rng)
        results[f"b={b}"] = entry
        print(f"b={b}: theory={c_theory:.5f} measured={c_mean:.5f}+-{c_sem:.5f} "
              f"rel_err={rel_err:+.3%} purity_dev={entry['purity_dev_mean']:.3%}")

    out_json = pathlib.Path(cfg["output_json"])
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))

    write_markdown_table(cfg, results)
    return results


def write_markdown_table(cfg: dict, results: dict) -> None:
    lines = ["| b | c(b) theory | c measured (mean +/- SE) | rel. error | purity dev. | seeds |",
             "|---:|---:|---:|---:|---:|---:|"]
    ry = results["b1_structural"]["ry_only"]
    lines.append(f"| 1 | {c_b.c_of_b(1):.5f} | structural (see Section 7.1); "
                  f"bare-RX check: {ry['c_measured']:.5f} +/- {ry['se_c']:.5f} | -- | -- | -- |")
    for b in cfg["block_sizes"]:
        e = results[f"b={b}"]
        lines.append(f"| {b} | {e['c_theory']:.5f} | {e['c_measured_mean']:.5f} +/- "
                      f"{e['c_measured_se']:.5f} | {e['rel_error']:+.2%} | {e['purity_dev_mean']:.2%} | "
                      f"{e['n_seeds']} |")
    out = pathlib.Path(cfg["output_table"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/exponent_sweep.yaml")
    args = p.parse_args()
    main(args.config)
