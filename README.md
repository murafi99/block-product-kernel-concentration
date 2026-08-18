# block-product-kernel-concentration

Reproducibility repository for *Concentration of Block-Product Quantum Fidelity
Kernels: Exact Theory and Experimental Validation*.

Proves a sharp, non-asymptotic interval **1 ≤ c ≤ c(b)** for the second-moment
concentration exponent of fidelity quantum kernels built from disjoint b-qubit
blocks (tighter than the naive Jensen interval [1,2], via Nakata et al. 2024's
Proposition 1), gives the exact closed form under a per-block 2-design
specialization, formally separates the kernel's variance-decay exponent from the
practical shot-noise threshold at which pairwise ranking of kernel values
collapses, and validates every claim against a **theoretical contract frozen
before any validation code was written** (`00_FROZEN_CONTRACT.md`).

## What's here

| Path | Contents |
|---|---|
| `theory/` | Exact formulas only — `moments.py` (per-block moment bounds), `c_b.py` (the exponent formula), `simcore.py` (gate-level statevector simulator), `statistics_utils.py` (Wilson CI, delta method, arcsine-law heuristic) |
| `experiments/` | Four experiment scripts (A–D), each independently runnable via a YAML config |
| `configs/` | Paper-scale configuration for every experiment, plus the seed registry |
| `tests/` | `pytest` suite: exact-formula checks, simulator correctness, statistics helpers, table generation |
| `reproduce_all.py` | Orchestrates theory self-check → all experiments → `results_manifest.json` |
| `results_manifest.json` | SHA-256-tagged manifest of every artifact from the last `reproduce_all.py` run (currently: smoke-scale, see below) |
| `dashboard/index.html` | Static viewer for the generated JSON results |

## Quick start

```bash
pip install -r requirements.txt
pytest tests/ -q                       # ~69 tests, exact-formula + simulator checks
python reproduce_all.py --scale smoke  # fast pipeline check, a few minutes
```

For paper-precision numbers (the ones actually reported in the paper — see
`REPRODUCTION.md` for expected runtimes, which range from minutes to roughly an
hour per experiment depending on block size and hardware):

```bash
python reproduce_all.py --scale full
```

## The central claim, in one table

Exact 2-design exponent c(b) = [(b−1) + log₂(2ᵇ+1)] / b, and its numerical
confirmation (Experiment A, direct moment estimation with propagated SE):

| b | c(b) theory | measured | rel. error |
|---:|---:|---:|---:|
| 2 | 1.66096 | 1.61271 ± 0.00642 | −2.91% |
| 3 | 1.72331 | 1.71467 ± 0.00230 | −0.50% |
| 4 | 1.77187 | 1.77135 ± 0.00221 | −0.03% |
| 5 | 1.80888 | 1.80660 ± 0.00300 | −0.13% |
| 6 | 1.83706 | 1.83965 ± 0.00267 | +0.14% |

b=1 is deliberately excluded from this table: a single continuous data parameter
traces a measure-zero curve on the Bloch sphere and provably cannot reach an exact
2-design at any depth (Corollary 1.2 / `REPRODUCTION.md` §4) — it is checked
instead against two constructions with known closed forms (bare RX(x), matching
Thanasilp et al. 2024 Proposition 1; and digitizing).

## Honesty notes

- `00_FROZEN_CONTRACT.md` was extracted from the paper before any code in this
  validation campaign was written and is not adjusted to fit results.
- `FINAL_REPORT.md` records every finding, including two genuine methodological
  problems (regression-based exponent estimation is fragile at low repetition
  count; the b=1 circuit family used elsewhere in the project does not reliably
  satisfy the 1-design assumption) rather than smoothing over them.
- The overall verdict is **YELLOW**, not GREEN: this campaign sharpens and
  stress-tests existing theoretical claims with real numbers and honest error
  bars; it does not establish a qualitatively new experimental consequence beyond
  what the theory already claimed. See `FINAL_REPORT.md` §15 for the full
  reasoning.

## Citing

See `CITATION.cff`.

## License

MIT — see `LICENSE`.
