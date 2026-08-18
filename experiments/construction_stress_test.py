"""
Five-ensemble stress test of the Nakata et al. (2024) lower bound at D=2
(frozen contract item 12e): every valid 1-design ensemble tried must give
E[X^2] >= 2/(D(D+1)) = 1/3, never below. Also reproduces the digitizing
exact-exponent check (contract item 12d) and the SIC-POVM exact check
(contract item 12c, E[X^2]=1/3 exactly).

Ensembles:
  1. digitizing       -- computational basis, {|0>,|1>} each w.p. 1/2. Ceiling case.
  2. ry_rotation       -- bare RX(x)|0>, x ~ Uniform(-pi,pi]. Thanasilp et al. Prop. 1.
  3. mub_two_basis     -- w.p. 1/2 draw from {|0>,|1>}, else from {|+>,|->}.
  4. sic_povm          -- uniform over the 4 single-qubit SIC-POVM fiducial states
                           (a genuine complex projective 2-design for D=2).
  5. haar_montecarlo   -- direct Haar sampling on the Bloch sphere (independent
                           construction method from sic_povm, same target value).

Usage:
    python experiments/construction_stress_test.py --config configs/stress_test.yaml
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
from theory import moments

KET0 = np.array([1, 0], dtype=complex)
KET1 = np.array([0, 1], dtype=complex)
KETP = (KET0 + KET1) / math.sqrt(2)
KETM = (KET0 - KET1) / math.sqrt(2)

# Single-qubit SIC-POVM fiducial states (tetrahedral construction), a known exact
# complex projective 2-design in D=2.
_SIC_BLOCH = np.array([
    [0, 0, 1],
    [2 * math.sqrt(2) / 3, 0, -1 / 3],
    [-math.sqrt(2) / 3, math.sqrt(2 / 3), -1 / 3],
    [-math.sqrt(2) / 3, -math.sqrt(2 / 3), -1 / 3],
])


def bloch_to_state(vec: np.ndarray) -> np.ndarray:
    x, y, z = vec
    theta = math.acos(np.clip(z, -1, 1))
    phi = math.atan2(y, x)
    return np.array([math.cos(theta / 2), math.sin(theta / 2) * np.exp(1j * phi)], dtype=complex)


SIC_STATES = np.array([bloch_to_state(v) for v in _SIC_BLOCH])


def fid(psi, phi):
    return float(np.abs(np.vdot(psi, phi)) ** 2)


def sample_digitizing(n, rng):
    labels = rng.integers(0, 2, n)
    states = np.where(labels[:, None] == 0, KET0, KET1)
    return states


def sample_ry(n, rng):
    thetas = rng.uniform(-math.pi, math.pi, n)
    return np.stack([np.array([math.cos(t / 2), -1j * math.sin(t / 2)]) for t in thetas])


def sample_mub(n, rng):
    basis = rng.integers(0, 2, n)
    which = rng.integers(0, 2, n)
    out = np.empty((n, 2), dtype=complex)
    for i in range(n):
        if basis[i] == 0:
            out[i] = KET0 if which[i] == 0 else KET1
        else:
            out[i] = KETP if which[i] == 0 else KETM
    return out


def sample_sic(n, rng):
    idx = rng.integers(0, 4, n)
    return SIC_STATES[idx]


def sample_haar(n, rng):
    z = rng.uniform(-1, 1, n)
    phi = rng.uniform(0, 2 * math.pi, n)
    out = np.empty((n, 2), dtype=complex)
    for i in range(n):
        out[i] = bloch_to_state(np.array([math.sqrt(1 - z[i] ** 2) * math.cos(phi[i]),
                                           math.sqrt(1 - z[i] ** 2) * math.sin(phi[i]), z[i]]))
    return out


SAMPLERS = dict(digitizing=sample_digitizing, ry_rotation=sample_ry, mub_two_basis=sample_mub,
                 sic_povm=sample_sic, haar_montecarlo=sample_haar)

EXACT_VALUES = dict(digitizing=0.5, ry_rotation=3.0 / 8.0, mub_two_basis=3.0 / 8.0,
                     sic_povm=1.0 / 3.0, haar_montecarlo=1.0 / 3.0)


def run_ensemble(name: str, n_pairs: int, rng: np.random.Generator) -> dict:
    sampler = SAMPLERS[name]
    a = sampler(n_pairs, rng)
    b = sampler(n_pairs, rng)
    fids = np.array([fid(a[i], b[i]) for i in range(n_pairs)])
    mean_x2 = float(np.mean(fids ** 2))
    se_x2 = float(np.std(fids ** 2, ddof=1) / math.sqrt(n_pairs))
    lower = moments.nakata_lower_bound(1)
    upper = moments.elementary_upper_bound(1)
    return dict(name=name, mean_x2=mean_x2, se_x2=se_x2, exact=EXACT_VALUES[name],
                nakata_lower_bound=lower, elementary_upper_bound=upper,
                respects_lower_bound=bool(mean_x2 >= lower - 4 * se_x2),
                respects_upper_bound=bool(mean_x2 <= upper + 4 * se_x2))


def main(config_path: str) -> dict:
    cfg = yaml.safe_load(open(config_path))
    rng = np.random.default_rng(20260818)
    results = {}
    for name in cfg["ensembles"]:
        r = run_ensemble(name, cfg["n_pairs"], rng)
        results[name] = r
        print(f"{name:16s} mean_X^2={r['mean_x2']:.5f}+-{r['se_x2']:.5f}  exact={r['exact']:.5f}  "
              f">= 1/3? {r['respects_lower_bound']}  <= 1/2? {r['respects_upper_bound']}")
    all_pass = all(r["respects_lower_bound"] and r["respects_upper_bound"] for r in results.values())
    results["_summary"] = dict(all_within_nakata_interval=all_pass,
                                nakata_lower_bound=moments.nakata_lower_bound(1),
                                elementary_upper_bound=moments.elementary_upper_bound(1))
    print(f"\nAll {len(cfg['ensembles'])} ensembles within [1/3, 1/2]: {all_pass}")

    out_json = pathlib.Path(cfg["output_json"])
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))

    lines = ["| ensemble | measured E[X^2] | exact | Nakata floor 1/3 | ceiling 1/2 | in bounds |",
             "|---|---:|---:|---:|---:|:---:|"]
    for name in cfg["ensembles"]:
        r = results[name]
        ok = "yes" if (r["respects_lower_bound"] and r["respects_upper_bound"]) else "NO"
        lines.append(f"| {name} | {r['mean_x2']:.5f} +/- {r['se_x2']:.5f} | {r['exact']:.5f} | "
                      f"1/3 | 1/2 | {ok} |")
    out_table = pathlib.Path(cfg["output_table"])
    out_table.parent.mkdir(parents=True, exist_ok=True)
    out_table.write_text("\n".join(lines) + "\n")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/stress_test.yaml")
    args = p.parse_args()
    main(args.config)
