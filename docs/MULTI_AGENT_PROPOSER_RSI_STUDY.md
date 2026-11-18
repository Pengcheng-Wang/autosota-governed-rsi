# Multi-Agent Iterative Proposal Study in Governed RSI (`Wave 4: D1–D9`)

This study directly tests two structural questions about Recursive Self-Improvement (`RSI`) under `/governed-loop-optimization` (`v3.6`):
1. **Does multi-agent proposer diversity (`4` independent proposer personas with orthogonal mathematical priors) materially outperform relying on a single proposer (`Coordinator` alone)?**
2. **Does feeding redacted Round-1 Search verdicts (`proposer_verdict.json`, zero Confirm/Holdout exposure) into a Round-2 Iterative Crossover/Refinement pass unlock confirmed single-touch Holdout breakthroughs that Round 1 missed?**

---

## 1. Experimental Design (`9` New `AutoSOTA` Benchmarks, `54` Standalone `python -I` Scripts)

Across `9` self-contained `AutoSOTA` benchmarks (`D1–D9`) spanning cooperative game theory, fairness/GroupDRO, certified monotonic learning, highway traffic sensor networks, financial graphical models, manifold MCMC, combinatorial Max-Cut, variational inference, and hard-constrained QP layers:
- **Every baseline, mandatory comparator, and candidate** executed as a standalone `.py` script across `governed_loop_guard.py run-isolated --mode predict` (subprocess `python -I -B`, scrubbed environment, static AST `audit-leaks` scan, and CLI `--forbid-paths immutable_eval/` check; note this provides process/interpreter isolation and static/CLI path guards rather than OS-kernel mandatory access control), binding its exact executable code `SHA-256` into `trial_ledger.jsonl` and `HOLDOUT_SEAL` (winning finalist `.py` scripts and tournament harnesses are archived in [`results/wave4_multiagent_d/finalists/`](../results/wave4_multiagent_d/finalists/) and [`results/wave4_multiagent_d/runners/`](../results/wave4_multiagent_d/runners/)).
- **Four Round-1 Proposers** operated **100% blind** to benchmark evaluation splits:
  - **`Proposer A` (`Coordinator-Math`):** Closed-form sufficient statistics, Kriging, WLS, convex-concave bases.
  - **`Proposer B` (`Fast-Explorer`):** Active 2-batch exploration, Halton QMC natural-gradient trust-region, multiple-imputation stability selection.
  - **`Proposer C` (`Convex-Spectral`):** Greedy D-optimal leverage designs, degree-2 minimax, GroupSort, Ballistic Simulated Bifurcation, ADMM Glasso.
  - **`Proposer D` (`Decision-Theory`):** Empirical-Bayes variance shrinkage, Polya-Gamma EM, active-set dual Newton, MAD neighbor shrinkage.
- **Two Round-2 Iterative Proposers (`Round2_Critic_Iterative` & `Round2_Explorer_Crossover`)** received **only redacted Round-1 Search metrics & tripwires** (`0` Confirm looks, `0` Holdout looks) to hybridize complementary Round-1 mechanisms and fix Search runtime/feasibility tripwires.
- *(Operational note on GUI Cowork delegation during overnight run: M365 Copilot `160.0.4409.0` was verified signed in with `Opus 5.5, Max` and `GPT 6.2 Astra` available prior to the `01:14 AM` Windows lock screen [`Windows desktop state: Locked, session_active=false`], which fail-closed GUI `computer-use` while headless multi-agent `task`/`rubber-duck` proposers executed uninterrupted across `24 × A100-80GB`.)*

---

## 2. Controlled Proposer Attribution & Multi-Round Ablation Across `D1–D9`

| Benchmark | Primary Metric | R0 Paper Baseline (Search) | Best Single-Agent (`Proposer A` R1) | Best 4-Agent R1 Winner & Agent | Best R2 Iterative Winner & Lineage | Final Gate & Single-Touch Sealed Holdout Outcome |
|---|---|---:|---:|---|---|---|
| **`D1` `Paper #48` `shapiq`** | `Precision@10` ($\uparrow$, `budget<=200`) | `0.2875` *(SVARM: `0.44`)* | `0.9063` | `0.9125` *(B: Explorer; A kept `0.9063` on margin)* | **`0.9500` (`R2 Crossover`: D-Opt + Order-3 Möbius)** | **SEALED WIN (`0.9375` Holdout, `+44.4 pp` vs Paper, `p=3.05e-5`)** |
| **`D2` `Paper #80` `ACS_AZ`** | `Worst-Group Acc` ($\uparrow$) | `0.5590` *(ERM: `0.4710`)* | `0.6477` | **`0.6584` (`C: Convex-Spectral` Poly2 Minimax)** | `0.6584` *(R1 C retained)* | Honest Block (`+7.8 pp` vs Paper `PROMOTE`, `-1.0 pp` vs `GroupDRO`, `0` Holdout looks) |
| **`D3` `Paper #77` `CMNN`** | `Std RMSE` ($\downarrow$, `0%` viol) | `0.8632` | **`0.7725`** | **`0.7725` (`A: Coordinator` Convex-Concave)** | `0.7717` *(within `0.004` margin)* | Honest Block (`-9.8%` vs Paper, lost to `Monotone-Spline-Ridge` `0.7401`, `0` Holdout looks) |
| **`D4` `Paper #90` `PEMS04`** | `Traffic MAE` ($\downarrow$) | `66.34` | `39.29` | **`39.29` (`A: Coordinator` Spatial Kriging)** | **`31.07` (`R2 Crossover`: Kriging + Spike Interp + SVD Factors)** | **SEALED WIN (`31.55` Holdout, `-52.9%` vs Paper, `-23.7%` vs Spatial-MLP, `p=1.53e-5`)** |
| **`D5` `Paper #9` `S&P 500`** | `Edge AUPRC` ($\uparrow$) | `0.4421` | `0.5117` | **`0.5286` (`B: Fast-Explorer` Multi-Imp Stab Glasso)** | `0.5349` *(within `0.008` margin; R1 B promoted)* | **SEALED WIN (`0.6321` Holdout, `+11.7 pp` [`+22.7%`] vs Paper `MSM`, `p=1.53e-5`)** |
| **`D6` `Paper #99` `OLLA`** | `Sliced W1` ($\downarrow$, `<=400` calls) | `0.0625` | `0.0696` | `0.0625` *(none beat margin; B got `0.0620`)* | **`0.0580` (`R2 Crossover`: Stein + Equil Langevin, `5.4x` faster)** | Honest Block (`-8.2%` Confirm mean, `p=0.305 > 0.0125` across heterogeneous manifolds, `0` Holdout looks) |
| **`D7` `Paper #73` `RLD`** | `Max-Cut` ($\uparrow$) | `486.96` | `496.78` | **`497.62` (`C: Convex-Spectral` Ballistic SB)** | **`499.40` (`R2 Crossover`: dSB + SDP + Memetic + 2-Flip)** | **SEALED WIN (`497.64` Holdout, `+13.34` vs Paper, `+12.29` vs `AutoSOTA`, `p=1.53e-5`)** |
| **`D8` `Paper #74` `LSVI`** | `Exact-MC ELBO` ($\uparrow$) | `-74.836` | `-74.984` *(lost)* | **`-74.783` (`B: Fast-Explorer` QMC NatGrad TR)** | `-74.783` *(R1 B retained; R2 BFGS hit `2.2x` cap)* | **SEALED WIN (`-77.348` Holdout, `+0.046` vs `LSVI`, `+0.252` vs `BBVI`, `+1.552` vs `Laplace`, `p=1.53e-5`)** |
| **`D9` `Paper #2` `PINet`** | `Opt RMSE` ($\downarrow$, `feas<=1e-4`) | `0.3040` | `3.6e-5` *(infeas)* | `0.3040` *(`C` was feas `6.7e-15` at `0.531` RMSE & `1.36x`)* | `3.16e-8` *(100% feas `2.3e-6`, `0.2555s / 0.2043s = 1.2506 > 1.25`)* | Halted at Search (`0` Confirm looks, `0` Holdout looks; exposed `AutoSOTA` `dr10` `1.1e-3` infeasibility) |

---

## 3. Three Quantitative Conclusions on Multi-Agent Iterative RSI

1. **`5 / 9` (`55.6%`) Confirmed Holdout Promotions in Wave 4 (Descriptively vs `7 / 20` [`35.0%`] Across Earlier Single-Agent Waves 0–3, which Tested Different Benchmark Subsets), with `100%` (`5/5`) Holdout Replication:**
   - Every promoted Wave-4 finalist (`D1, D4, D5, D7, D8`) replicated on untouched Protected Holdout (`15/16` positive margin-shifted clusters on `D1` [$p = 3.05 \times 10^{-5}$] and `16/16` positive clusters on `D4, D5, D7, D8` [$p = 1.53 \times 10^{-5}$]) with **zero post-Holdout retractions**.
2. **Controlled Within-Wave-4 Paired Ablation (`Same 9 Benchmarks & Harnesses`): Single-Agent (`Proposer A` Alone) Would Have Missed or Degraded `4 of the 5` Wave-4 Sealed Wins:**
   - On **`D8` (`LSVI`)**, `Proposer A` (`Coordinator`) failed to beat baseline (`-74.984 < -74.836`); **`Proposer B` (`Fast-Explorer`)** was the *sole* Round-1 agent to beat `LSVI` within budget and delivered a `16/16`-cluster Holdout win!
   - On **`D5` (`S&P 500`)** and **`D7` (`RLD`)**, `Proposer B` and `Proposer C` (`Convex-Spectral`) beat `Proposer A` cleanly in Round 1.
   - Across Round 1 (`D1–D9`), leadership split almost evenly across orthogonal minds: **`Proposer A` led 2, `Proposer B` led 3 (`D1` raw, `D5`, `D8`), and `Proposer C` led 2 (`D2`, `D7`)**.
3. **Round-2 Iterative Crossover (Using Redacted Search Verdicts Only) Delivered Dramatic Compounding Gains on `4 / 9` Benchmarks:**
   - **`D1` (`shapiq`):** `0.9063` (R1) $\to$ **`0.9500`** (R2, cutting MSE `8x` via Order-3 Möbius residualization).
   - **`D4` (`PEMS04`):** `39.29` (R1) $\to$ **`31.07`** (R2, an *additional* `-20.9%` MAE reduction by fusing `Proposer A` Kriging with `Proposer B` highway SVD factors and adjacent-step spike interpolation).
   - **`D6` (`OLLALanding`):** `0.0625` (R1 stuck at baseline) $\to$ **`0.0580`** (R2, `5.4x` speedup via batched linear algebra + Stein-equilibrium two-stage schedule).
   - **`D7` (`RLD Max-Cut`):** `497.62` (R1) $\to$ **`499.40`** (R2, hybridizing `Proposer C` Simulated Bifurcation + `Proposer A` SDP + `Proposer B` memetic crossover + `Critic` 2-flip escape).
