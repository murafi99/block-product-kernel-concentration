# FROZEN THEORETICAL CONTRACT

Extracted from `paper.tex` before any code in this validation campaign was written. These
equations are the target; nothing below is adjusted to fit experimental results.

1. **Fidelity kernel**: κ(x,x') = |⟨Ψ(x)|Ψ(x')⟩|².
2. **Block-product architecture**: n=mb qubits, m disjoint blocks of b qubits,
   |Ψ(x)⟩ = |ψ₁(x)⟩⊗···⊗|ψₘ(x)⟩, no entangling gate crosses a block boundary at any depth.
3. **A1**: block-product structure (circuit-topology statement, no randomness required).
4. **A2**: cross-block statistical independence of X₁,...,Xₘ over whatever randomness Ω defines
   variance (data-marginal independence, or ensemble/circuit-family independence — both used
   in this campaign, kept distinguished).
5. **A3**: per-block 1-design — 𝔼_Ω[|ψᵢ(x)⟩⟨ψᵢ(x)|] = 𝟙_D/D exactly.
6. **Xᵢ := |⟨ψᵢ(x)|ψᵢ(x')⟩|²** ∈ [0,1].
7. **κ := ∏ᵢ₌₁ᵐ Xᵢ** (exact tensor-product algebra, A1 only).
8. **Exact moments**: 𝔼[κ]=∏𝔼[Xᵢ], 𝔼[κ²]=∏𝔼[Xᵢ²] (A1+A2). Under A3: 𝔼[Xᵢ]=1/D exactly
   (Lemma: 𝔼[Xᵢ]=Tr[ρ̄ᵢ²], the purity of the average block state).
9. **General 1-design theorem**: 2/(D(D+1)) ≤ 𝔼[Xᵢ²] ≤ 1/D for *any* ensemble (1-design or not);
   lower bound is Nakata et al. 2024 Prop. 1 (cited, not re-derived); upper bound is the
   elementary Xᵢ≤1 ⟹ Xᵢ²≤Xᵢ argument (original). Gives, for the block-product kernel,
   **1 ≤ c ≤ c(b)** exactly, for every finite n — not the looser [1,2].
10. **2-design specialization**: under A1,A2 + per-block 2-design, 𝔼[Xᵢ²]=2/(D(D+1)) exactly,
    Var[κ] = [2/(D(D+1))]^m − D^(−2m) exactly (not asymptotic).
11. **Concentration exponent**: c(b) := [(b−1)+log₂(2^b+1)]/b, defined via 𝔼[κ²]=2^(−cn).
    Exact values (recorded symbolically before decimal approximation):
    c(1)=[0+log₂3]/1=1.5849625007, c(2)=[1+log₂5]/2=1.6609640474,
    c(3)=[2+log₂9]/3=1.7233083338, c(4)=[3+log₂17]/4=1.7718657103,
    c(5)=[4+log₂33]/5=1.8088788239, c(6)=[5+log₂65]/6=1.8370613022, c(b)→2 as b→∞.
12. **Experimental claims already established in the paper** (to be re-tested, not assumed):
    (a) idealized Haar-block product formula matches exact/MC to high precision;
    (b) real finite-depth circuits converge toward c(b) as depth L increases, shown for b=1..4
    at L=1,2,4,8, with b=1 flagged as non-convergent (no entangling gate exists);
    (c) qubit SIC-POVM gives E[X²]=1/3 exactly, matching c(1);
    (d) digitizing gives exponent exactly 1 for b=1,2,3,4,8,16 (exact algebra + spot numerics);
    (e) five-ensemble stress test of the Nakata bound, D=2, none below 1/3;
    (f) shared-batch (150-point, correlated pairs) ranking-agreement experiment showed the
    predicted ordering (global < b=2 < b=1 collapse point) but was explicitly flagged as a
    weaker design, to be replaced in this campaign (Part IV).
13. **Limitations already identified in the paper**: (i) real-circuit exponent fit was noisy for
    blocks in the earlier off-diagonal-variance diagnostic due to a small shared-batch sampling
    design; (ii) the SNR/ranking threshold gap was explained by an unverified Gaussian heuristic;
    (iii) b=1 circuits cannot be guaranteed to converge to Haar statistics regardless of depth
    (no entangling gate exists to scramble a single qubit toward 2-design); (iv) end-to-end
    downstream ML task design (attempts 1–2) failed for reasons unrelated to concentration and
    was abandoned in favor of the task-agnostic ranking metric.

No equation above is to be modified to improve agreement with simulation. If disagreement is
found, Part X of the protocol requires investigation and honest reporting, not adjustment.
