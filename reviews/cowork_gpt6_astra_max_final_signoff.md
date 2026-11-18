# Frontier Cowork Final Sign-Off Audit (`GPT 6 Astra, Max`)

- **App / Model:** Microsoft 365 Copilot (`Cowork`, `Model and generation settings: GPT 6 Astra, Max`, `Worked for 3m 2s`)
- **Thread:** `Optimization Audit and Release Readiness` (`cf1e8666-94d8-4a8c-b75c-2e3e8a486dba`, Final Turn)

---

## Verbatim `GPT 6 Astra, Max` Audit

**Conditional sign-off: a credible audit campaign, not unconditional statistical certification.**

Based solely on your summary:

### (a) `v3.4` Statistical & Governance Critique

- **Directionally sensible, not proven calibrated:** Removing the Gaussian cap avoids an additional normal approximation; its negative LCB was not itself proof of a bug. Call this **“bootstrap-based”** (`percentile-bootstrap`), not **“skewness-equivariant.”** Eight clusters and passing tests do not establish unconditional coverage.
- **Sign-flip exactness is the key issue:** It requires independent clusters and null sign-exchangeability or justified randomization—not simply $G < 20$. Arbitrarily skewed mean-zero gains need not satisfy this. A sign test instead targets win probability/median, not mean improvement. ETT cells/horizons are not automatically independent. AND-ing gates needs no independence, but cannot manufacture calibration.
- **External noise:** Quadrature needs a justified independence/noise model; adding variances does not generally produce a valid percentile-bootstrap quantile. **Model external noise jointly.**
- **Check p-values:** For 16 strictly positive gains, zero margin and exhaustive one-sided mean-statistic sign flips, $p = 1/65{,}536 \approx 0.00001526$, not $0.000125$. Explain any adjustment, nonzero margin or Monte Carlo computation.
- **Prospective validity:** Changing the gate after `TreeHFD` makes that Confirm decision retrospective. Untouched Holdout remains useful, but seals cannot retroactively preregister rules or cure adaptive Confirm reuse. Document candidate/look multiplicity and policy versions. Zero looks must include prior campaigns and manual access, not just this ledger.
- **Actionable synthesis:** Require null calibration under skew, dependence and adaptive looks, **with both gates aligned to the promotion margin**.

### (b) Defensible Headlines & Necessary Scientific Boundaries

> **Defensible headline:** *“An 11-benchmark governed audit reported five held-out improvements and six non-promotions without Holdout access, while exposing privacy, compute-budget and comparison-protocol failures.”*
> *(Note: That is a campaign outcome—not a general success probability or proof governance caused the gains.)*

| Track / Benchmark | Necessary Scientific Boundary |
|---|---|
| **`A2` (`TropicalAttention`)** | Strongest controlled headline: gains against an equal-runtime comparator within tested shifts. |
| **`A4` (`Paper #96 M3SVM-DP`)** | “Exact DP” requires end-to-end accounting including adaptive selection; statistical claims also need selection-aware evaluation. |
| **`B1` (`Paper #88 ChannelNorm`)** | “85% attributable to sequence length” requires a controlled ablation (`RMLP L=96 -> RMLP L=336 -> RMLP+CN L=336`, all verified on identical splits), and note cross-horizon ETT correlation across the 16 cells. |
| **`B3` (`Paper #104 BalancedActiveInf`)** | Width superiority requires uncertainty-aware, per-stratum coverage verification (`1,600` MC trials/stratum, binomial SE $\approx 0.0075$ at $90\%$ nominal coverage). |
| **`B5` (`Paper #69 TreeHFD`)** | Lead with improvement over `AutoSOTA` (`-7.91%` MSE on unseen shifted Holdout); distinguish stability rescue of `depth=6` from superiority over a functioning regularized baseline, and disclose that `v3.4` removed the Gaussian LCB cap after observing `B5` Confirm skew. |

**Release recommendation:** Reproducible audit/case-study paper plus a focused guard PR; withhold claims of exact general calibration, universal SOTA, or guaranteed correctness.

---

## Follow-Up Actions Applied Immediately in `v3.5` (`governed_loop_guard.py`)

1. **Exact exhaustive sign-flip raised to $G \le 16$ (`65,536` flips) + explicit `signflip_method`:** Fixed why $G = 16$ (`B1`) previously fell into the `n_perm = 8,000` Phipson-Smyth Monte-Carlo branch ($1/8{,}000 = 0.000125$); $G = 16$ now computes exact $1/65{,}536 \approx 0.00001526$ and records `"signflip_method": "exact" | "monte_carlo_8000"`.
2. **Both gates aligned to the promotion margin (`null_margin = min_improvement`):** Cluster deltas in the paired Rademacher sign-flip test are now shifted by `null_margin` ($d_g - \Delta_{\min}$) so both the percentile-bootstrap LCB and the paired sign-flip p-value test the identical margin-shifted null $H_0: \mu_\Delta \le \Delta_{\min}$.
3. **Joint external Gaussian noise sampling inside the bootstrap:** When `aa_noise_se > 0` (or `reported_se > 0`), independent $\mathcal{N}(0, \sigma_{\text{ext}}^2)$ draws are convolved directly inside each bootstrap replicate rather than subtracting an ad-hoc post-hoc variance gap from a skewed percentile, and comments/docs now refer accurately to **percentile-bootstrap LCB** rather than “skewness-equivariant.”
