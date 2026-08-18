"""
Experiment C -- finite-shot phase diagram (Section 7.4, Figure 4).

Grid over (n, N_shots) for each block size, reusing experiments/ranking_agreement's
sampling machinery. Extracts, per column (fixed n), the smallest N_shots at which
agreement first reaches the PREREGISTERED boundary (0.75, fixed in configs/
phase_diagram.yaml before this script is ever run -- see README/REPRODUCTION for
the preregistration statement), and compares that empirical boundary against the
two named heuristics n*_var and n*_rank (theory/statistics_utils.py).

Also computes the full arcsine-law goodness-of-fit across every grid cell (Section
7.4's "180 grid points" statistic, generalized to whatever grid size the config
specifies) -- reported honestly as a calibrated heuristic with known error, not
re-fit to the data.

Usage:
    python experiments/phase_diagram.py --config configs/phase_diagram.yaml
"""
from __future__ import annotations
import argparse
import json
import pathlib
import sys

import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from theory import c_b, moments, simcore, statistics_utils as stats
from experiments.ranking_agreement import agreement_at_n


def main(config_path: str) -> dict:
    cfg = yaml.safe_load(open(config_path))
    rng = np.random.default_rng(20260818)
    domain = (-np.pi, np.pi)
    boundary_level = cfg["preregistered_agreement_boundary"]

    results = {}
    errors = []
    for b in cfg["block_sizes"]:
        grid = []
        c_theory = c_b.c_of_b(b)
        mu_n = {}  # cache mu=E[kappa], var=Var[kappa] per n for the heuristic
        for n in cfg["n_grid"]:
            if n % b != 0:
                continue
            m = n // b
            mu = moments.first_moment_1design(b) ** m
            var = moments.var_kappa_2design(b, m)
            mu_n[n] = (mu, var)
            row = []
            for n_shots in cfg["n_shots_grid"]:
                r = agreement_at_n("block", b, n, n_shots, cfg["trials_per_cell"],
                                    cfg["depth"], domain, rng)
                heuristic = stats.arcsine_law_agreement(var, mu, n_shots)
                err = abs(r["agreement"] - heuristic)
                errors.append(err)
                row.append(dict(n=n, n_shots=n_shots, agreement=r["agreement"],
                                 ci_lo=r["ci_lo"], ci_hi=r["ci_hi"],
                                 heuristic=heuristic, abs_error=err))
            grid.append(row)
            print(f"b={b} n={n:3d} done (last cell A={row[-1]['agreement']:.3f}, "
                  f"heuristic={row[-1]['heuristic']:.3f})")

        boundary_n_shots = extract_boundary(grid, boundary_level)
        comparison = compare_boundary_to_heuristics(grid, c_theory)
        results[f"b={b}"] = dict(grid=grid, boundary_at_A=boundary_level,
                                  boundary_n_shots_by_n=boundary_n_shots,
                                  heuristic_comparison=comparison)

    errors = np.array(errors)
    results["_goodness_of_fit"] = dict(
        n_points=int(errors.size),
        median_abs_error=float(np.median(errors)),
        mean_abs_error=float(np.mean(errors)),
        p90_abs_error=float(np.percentile(errors, 90)),
        max_abs_error=float(np.max(errors)),
    )
    print("\nArcsine-law heuristic goodness-of-fit over full grid:")
    print(json.dumps(results["_goodness_of_fit"], indent=2))

    out_json = pathlib.Path(cfg["output_json"])
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))

    try:
        plot_figure4(results, cfg["block_sizes"], cfg["output_figure"])
    except Exception as e:
        print(f"(figure generation skipped: {e})")
    return results


def extract_boundary(grid, level):
    """For each n (row), find the smallest N_shots at which agreement >= level,
    via linear interpolation in log2(N_shots) between the bracketing grid points.
    """
    out = {}
    for row in grid:
        n = row[0]["n"]
        prev = None
        crossing = None
        for cell in row:
            if cell["agreement"] >= level:
                if prev is None:
                    crossing = cell["n_shots"]
                else:
                    x0, y0 = np.log2(prev["n_shots"]), prev["agreement"]
                    x1, y1 = np.log2(cell["n_shots"]), cell["agreement"]
                    if y1 > y0:
                        frac = (level - y0) / (y1 - y0)
                        crossing = 2 ** (x0 + frac * (x1 - x0))
                    else:
                        crossing = cell["n_shots"]
                break
            prev = cell
        out[n] = crossing
    return out


def compare_boundary_to_heuristics(grid, c_theory):
    out = []
    boundary = extract_boundary(grid, 0.75)
    for n, ns in boundary.items():
        if ns is None:
            continue
        n_var = stats.n_star_var(c_theory, ns)
        n_rank = stats.n_star_rank(c_theory, ns)
        out.append(dict(n=n, empirical_boundary_n_shots=ns, n_star_var_at_that_n_shots=n_var,
                         n_star_rank_at_that_n_shots=n_rank,
                         closer_to=("n_star_var" if abs(n - n_var) < abs(n - n_rank) else "n_star_rank")))
    return out


def plot_figure4(results: dict, block_sizes, out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(block_sizes), figsize=(6 * len(block_sizes), 5), squeeze=False)
    for ax, b in zip(axes[0], block_sizes):
        grid = results[f"b={b}"]["grid"]
        ns = sorted({c["n"] for row in grid for c in row})
        shots = sorted({c["n_shots"] for row in grid for c in row})
        Z = np.zeros((len(shots), len(ns)))
        for row in grid:
            for c in row:
                i, j = shots.index(c["n_shots"]), ns.index(c["n"])
                Z[i, j] = c["agreement"]
        im = ax.imshow(Z, aspect="auto", origin="lower", cmap="RdYlBu_r", vmin=0.5, vmax=1.0,
                        extent=[0, len(ns), 0, len(shots)])
        ax.set_xticks(np.arange(len(ns)) + 0.5); ax.set_xticklabels(ns)
        ax.set_yticks(np.arange(len(shots)) + 0.5); ax.set_yticklabels([f"{s:.0e}" for s in shots])
        ax.set_xlabel("n"); ax.set_ylabel("N_shots")
        ax.set_title(f"block b={b}: agreement A")
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/phase_diagram.yaml")
    args = p.parse_args()
    main(args.config)
