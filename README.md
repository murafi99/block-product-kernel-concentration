# Data companion — Exact Bounds on Exponential Concentration for Block-Product Quantum Fidelity Kernels

An interactive, browser-based companion to the paper. Open `index.html` in any browser — no build step, no dependencies beyond a CDN load of Chart.js.

## What's in here

```
paper-data-repo/
├── README.md
├── index.html                          interactive dashboard (11 tabs)
└── data/
    ├── table1_exponent_sweep.csv       Table 1 — finite-depth circuit validation, b=1..6
    ├── table2_construction_stress_test.csv   Table 2 — five b=1 constructions
    ├── table3_ranking_agreement.csv    Table 3 — measured ranking agreement under shot noise
    └── cb_theory_curve.csv             c(b) closed form, b=1..20, generated from Eq. 3
```

## Tabs

1. **Explorer** — drag block size b, everything recomputes from Eq. 3
2. **Theorem Figures** — Fig. 1/2 recreations with hover tooltips into Table 1
3. **Shot-Budget Tool** — live Section 5.1 arcsine-heuristic calculator
4. **Raw Tables** — sortable Tables 1–3 plus the full c(b) curve
5. **Assumptions & Scope** — A1–A4 and the out-of-scope construction comparison
6. **Live Simulator** — genuine in-browser Monte Carlo: draws Haar-random states, checks the empirical distribution and moments against the Beta(1,D−1) law used in Theorem 2's proof
7. **Interpolation Lab** — simulates Open Question 5's per-use random-coin construction, which the paper itself describes as "derived but not simulated," and compares it live against the static per-block-type partition
8. **Proof Walkthrough** — accordion of Lemma 1–3, Theorem 1, Corollary 1.1, Proposition 1, Theorem 2, with proof sketches
9. **Glossary** — 11 terms used across the manuscript
10. **References** — all 8 citations with links; two were independently verified against primary sources (Kairon/Jäger/Krems and Nakata et al.) and are flagged as such
11. **Download Repo** — this file listing

## Provenance — what's original data vs. what's recomputed

Everything in this repo falls into exactly one of two buckets:

1. **Verbatim from the manuscript.** Tables 1, 2, and 3 are transcribed exactly as printed (values, standard errors, seed counts). No smoothing, no interpolation, no reading values off the figure images.
2. **Recomputed live from the manuscript's own closed-form equations.** The Explorer, Fig. 1/2 recreations, and the Shot-Budget Tool call the actual formulas:
   - `c(b) = [(b−1) + log2(2^b+1)] / b` — Eq. 3 (Theorem 2)
   - `Var[κ] = [2/(D(D+1))]^m − D^(−2m)` — Eq. 2 (Theorem 2)
   - `agreement(r) = 1/2 + arcsin(sqrt(r/(r+1)))/π`, `r = Var[κ]·N` — Section 5.1's Gaussian arcsine heuristic

The dashboard says so inline wherever this matters — e.g. the Shot-Budget Tool explicitly reminds you it's using the *heuristic* from §5.1 (median absolute error 0.0165 against 180 simulated grid points in the paper), not a re-run of the underlying finite-depth-circuit simulation. Figure 4's phase-diagram heatmaps and the exact 180-point calibration grid are **not** reproduced here because the paper doesn't publish the raw grid values — only the summary statistics, which are in Table 1 and Section 5.3's text.

## Known caveats carried over from the paper

- Table 1, b=1 row: not a 2-design test — it uses the Thanasilp Ry construction, see manuscript §4.2/§6.1.
- Table 1, b=5 and b=6: single circuit-parameter seed. The reported SE likely reflects shot noise only, not seed-to-seed circuit variability — treat those two error bars as a lower bound on true uncertainty.
- Figure 4 phase diagrams (exact heatmap grid) are approximated in the Shot-Budget Tool via the arcsine heuristic, not reproduced pixel-for-pixel.

## Regenerating `cb_theory_curve.csv`

```python
import math
for b in range(1, 21):
    D = 2**b
    c = ((b - 1) + math.log2(2**b + 1)) / b
    print(b, D, round(c, 6))
```

## License / status

Companion tool only. Not peer-reviewed on its own. Cite the manuscript, not this repo, for the results themselves.
