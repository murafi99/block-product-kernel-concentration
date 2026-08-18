#!/usr/bin/env python3
"""
Runs the full validation campaign end to end: theory self-checks, all four
experiments (A-D), and writes results_manifest.json recording exactly which
artifacts were produced, from which config, with what git-independent content
hash, so a later run can be checked against this one.

By default runs the SMOKE-scale configs (fast, for CI and for verifying the
pipeline works) rather than the full paper-scale configs in configs/*.yaml, which
can take from minutes (Experiment A/B at small b) to on the order of an hour
(Experiment A at b=6, or the full Experiment C grid) depending on hardware -- see
REPRODUCTION.md for expected runtimes and how to run the full-scale versions.

Usage:
    python reproduce_all.py --scale smoke     # fast, default -- verifies pipeline
    python reproduce_all.py --scale full      # paper-scale, see REPRODUCTION.md
"""
from __future__ import annotations
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import time

import yaml

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def sha256_of(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else None


def make_smoke_config(base_path: pathlib.Path, overrides: dict, out_path: pathlib.Path) -> pathlib.Path:
    cfg = yaml.safe_load(base_path.read_text())
    cfg.update(overrides)
    out_path.write_text(yaml.dump(cfg))
    return out_path


def run_theory_self_check() -> dict:
    from theory import c_b, moments
    ok = True
    checks = {}
    for b, expected in c_b.FROZEN_CONTRACT_VALUES.items():
        computed = c_b.c_of_b(b)
        match = abs(computed - expected) < 1e-9
        checks[f"c({b})"] = dict(computed=computed, expected=expected, match=match)
        ok = ok and match
    for b in range(1, 7):
        interval_ok = moments.sanity_check_interval(b)
        checks[f"interval_b{b}"] = dict(nakata_lower=moments.nakata_lower_bound(b),
                                         elementary_upper=moments.elementary_upper_bound(b),
                                         ok=interval_ok)
        ok = ok and interval_ok
    return dict(all_passed=ok, checks=checks)


def main(scale: str) -> None:
    manifest = dict(generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), scale=scale,
                     artifacts=[])

    print("=" * 70)
    print("1/5  Theory self-check (exact formulas vs. frozen contract)")
    print("=" * 70)
    theory_check = run_theory_self_check()
    print(f"All theory checks passed: {theory_check['all_passed']}")
    manifest["theory_self_check"] = theory_check
    if not theory_check["all_passed"]:
        print("FATAL: theory self-check failed -- refusing to run experiments against "
              "a possibly-broken formula. See theory/c_b.py, theory/moments.py.")
        sys.exit(1)

    cfg_dir = ROOT / "configs"
    tmp_dir = ROOT / "data" / "raw" / f"_run_{scale}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if scale == "smoke":
        overrides = dict(
            exponent_sweep=dict(block_sizes=[2, 3], depth=4, n_pairs=300,
                                 seeds_per_block={2: 1, 3: 1}, run_regression_crosscheck=True,
                                 regression_block_repeats=[1, 2],
                                 output_json=str(tmp_dir / "expA.json"),
                                 output_table=str(ROOT / "tables" / "table_experiment_A.md"),
                                 b1_structural=dict(ry_only_depth=1, n_pairs=300, digitizing_bins=2)),
            stress_test=dict(n_pairs=3000, output_json=str(tmp_dir / "stress.json"),
                              output_table=str(ROOT / "tables" / "table_stress_test.md")),
            ranking=dict(block_sizes=[1, 2], global_max_n=6, n_values=[2, 4, 6, 8, 10],
                         trials_per_n=80, depth=4, output_json=str(tmp_dir / "expB.json"),
                         output_figure=str(ROOT / "figures" / "figure3_ranking_agreement.png")),
            phase_diagram=dict(block_sizes=[1, 2], n_grid=[4, 8, 12], n_shots_grid=[100, 1000, 10000, 100000],
                                trials_per_cell=80, depth=4,
                                output_json=str(tmp_dir / "expC.json"),
                                output_figure=str(ROOT / "figures" / "figure4_phase_diagram.png")),
        )
    else:
        overrides = dict(exponent_sweep={}, stress_test={}, ranking={}, phase_diagram={})

    print("\n" + "=" * 70)
    print("2/5  Experiment A: block-size exponent sweep")
    print("=" * 70)
    cfgA = make_smoke_config(cfg_dir / "exponent_sweep.yaml", overrides["exponent_sweep"],
                              tmp_dir / "exponent_sweep.yaml")
    (cfg_dir / "seeds.yaml").read_text()  # sanity: file must exist
    import shutil
    shutil.copy(cfg_dir / "seeds.yaml", tmp_dir / "seeds.yaml")
    from experiments import exponent_sweep
    resA = exponent_sweep.main(str(cfgA))

    print("\n" + "=" * 70)
    print("3/5  Construction stress test (Nakata bound, five ensembles)")
    print("=" * 70)
    cfgS = make_smoke_config(cfg_dir / "stress_test.yaml", overrides["stress_test"], tmp_dir / "stress_test.yaml")
    from experiments import construction_stress_test
    resS = construction_stress_test.main(str(cfgS))

    print("\n" + "=" * 70)
    print("4/5  Experiment B: independent-quadruplet ranking")
    print("=" * 70)
    cfgB = make_smoke_config(cfg_dir / "ranking.yaml", overrides["ranking"], tmp_dir / "ranking.yaml")
    from experiments import ranking_agreement
    resB = ranking_agreement.main(str(cfgB))

    print("\n" + "=" * 70)
    print("5/5  Experiment C: finite-shot phase diagram")
    print("=" * 70)
    cfgC = make_smoke_config(cfg_dir / "phase_diagram.yaml", overrides["phase_diagram"], tmp_dir / "phase_diagram.yaml")
    from experiments import phase_diagram
    resC = phase_diagram.main(str(cfgC))

    print("\n" + "=" * 70)
    print("Bonus: Experiment D (controlled interpolation, b=2)")
    print("=" * 70)
    from experiments import interpolation
    depth_d, n_pairs_d = (4, 300) if scale == "smoke" else (32, 20000)
    resD = interpolation.main(b=2, depth=depth_d, n_pairs=n_pairs_d)

    for tag, path in [
        ("expA_json", overrides["exponent_sweep"].get("output_json", "data/processed/expA.json")),
        ("expA_table", "tables/table_experiment_A.md"),
        ("stress_json", overrides["stress_test"].get("output_json", "data/processed/stress_test.json")),
        ("stress_table", "tables/table_stress_test.md"),
        ("expB_json", overrides["ranking"].get("output_json", "data/processed/expB.json")),
        ("figure3", "figures/figure3_ranking_agreement.png"),
        ("expC_json", overrides["phase_diagram"].get("output_json", "data/processed/expC.json")),
        ("figure4", "figures/figure4_phase_diagram.png"),
        ("expD_json", "data/processed/expD.json"),
        ("expD_table", "tables/table_experiment_D.md"),
    ]:
        p = ROOT / path if not pathlib.Path(path).is_absolute() else pathlib.Path(path)
        try:
            rel = str(p.relative_to(ROOT))
        except ValueError:
            rel = str(p)  # outside repo root (e.g. a custom tmp_dir) -- keep absolute
        manifest["artifacts"].append(dict(tag=tag, path=rel, exists=p.exists(),
                                           sha256_16=sha256_of(p) if p.exists() else None))

    manifest["experiment_C_goodness_of_fit"] = resC.get("_goodness_of_fit")
    manifest["stress_test_all_within_bounds"] = resS["_summary"]["all_within_nakata_interval"]

    out_path = ROOT / "results_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"\nWrote {out_path}")
    print(f"Scale: {scale}. See REPRODUCTION.md for full-scale (paper-precision) runs.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scale", choices=["smoke", "full"], default="smoke")
    args = p.parse_args()
    main(args.scale)
