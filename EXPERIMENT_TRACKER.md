# Master Experiment Tracker (Successful & Unsuccessful Governed RSI Runs)

Every campaign run under `/governed-loop-optimization` is indexed below without omission.

---

## 1. Confirmed & Single-Touch Sealed-Holdout Wins

| ID | Paper / Benchmark | Domain | `AutoSOTA` Flaw Caught | Confirm Gate | Holdout Seal (`SHA-256`) | Sealed Holdout Result (`95% CI`) | Report |
|---|---|---|---|---|---|---|---|
| **`A2`** | `Paper #66` `TropicalAttention` | Algorithmic OOD | `L=16` train-aug leak + `3x` scale collapse | `PROMOTE` (3/3) | `1923f0a8...` | **`0.6682` Macro-F1** (`+6.52 pp` vs Tropical `[+5.66,+7.44]`, `+3.97 pp` vs 3L equal-runtime `[+2.50,+5.52]`) | [`A2_paper66_TropicalAttention.md`](./experiments/successful/A2_paper66_TropicalAttention.md) |
| **`A4`** | `Paper #96` `M3SVM` (`8` datasets) | $(\varepsilon=3, \delta=10^{-5})$-DP | `_base_coef = coef_ - noise` (`eps=21.21 > 3.0`) | `PROMOTE` (2/2) | `8a6421e8...` | **`43.77%` Acc** (`+13.09 pp` vs `pmsvm_dp` `[+8.44,+17.90]`; per-run exact DP) | [`A4_paper96_M3SVM_DP.md`](./experiments/successful/A4_paper96_M3SVM_DP.md) |
| **`B1`** | `Paper #88` `ChannelNorm` (`16` ETT) | Time-Series (`L=336`) | `85.1%` of raw drop came from `seq_len 96->336` | `PROMOTE` (2/2) | `983ee190...` | **`0.3568` MSE** (`-4.45%` vs `RMLP-L336`, `-3.23%` vs `AutoSOTA-CN-L336`, `-1.38%` vs `Ridge`) | [`B1_paper88_ChannelNorm.md`](./experiments/successful/B1_paper88_ChannelNorm.md) |
| **`B3`** | `Paper #104` `BalancedActiveInf` | Active Inference | `AutoSOTA` under-covered `2/4` Holdout DGPs (`0.8625`) | `PROMOTE` (2/2) | `12c2987c...` | **`0.7950` CI width** (`-26.73%` vs Paper, `-4.58%` vs `AutoSOTA`) + valid `0.885–0.948` coverage on all 4 DGPs | [`B3_paper104_BalancedActiveInf.md`](./experiments/successful/B3_paper104_BalancedActiveInf.md) |
| **`B5`** | `Paper #69` `TreeHFD` | Functional ANOVA | Paper `depth=6` unstable; `AutoSOTA` `depth=3` valid | `PROMOTE` *(v3.4)* | `537eae9f...` | **`0.1012` MSE** (`-7.91%` vs `AutoSOTA` `[+0.0028,+0.0153]`, `-93.78%` vs Paper `depth=6`) | [`B5_paper69_TreeHFD.md`](./experiments/successful/B5_paper69_TreeHFD.md) |

---

## 2. Unsuccessful, Baseline-Tied & Governance-Blocked Runs (`0` Holdout Looks Consumed)

| ID | Paper / Benchmark | Domain | Why It Did Not Promote | Gate Verdict | Holdout Looks | Key RSI Learning | Report |
|---|---|---|---|---|---:|---|---|
| **`P0`** | `packing32` / `NAS` / raw `xPatch` | Prior Reproduction | Non-convex basin saturation, flat NAS space, lookback confound | `NO_NEW_GAIN` | `N/A` | Motivated Step-0 `check-preconditions` & `audit-leaks` | [`P0_Prior_AutoSOTA_Packing32_and_NAS.md`](./experiments/unsuccessful/P0_Prior_AutoSOTA_Packing32_and_NAS.md) |
| **`A1`** | `Paper #42` `xPatch` (`16` ETT) | Time-Series (`L=336`) | Beat `xPatch` & `AutoSOTA`, tied `Ridge` (`0.4400` vs `0.4394`) | `BLOCKED_NOT_CONFIRMED` | `0` | Mandatory linear `Ridge` comparator blocks illusory DL SOTA claims | [`A1_paper42_xPatch_RidgeParity.md`](./experiments/unsuccessful/A1_paper42_xPatch_RidgeParity.md) |
| **`A3`** | `FunSearch` Bin Packing | Online Heuristics | Beat `Best-Fit`, lost to `FunSearch-OR` (`0.0647` vs `0.0624`) | `BLOCKED_NOT_CONFIRMED` | `0` | `causal_online` stdio prevents lookahead; SOTA heuristic held firm | [`A3_FunSearch_CausalOnline_BinPacking.md`](./experiments/unsuccessful/A3_FunSearch_CausalOnline_BinPacking.md) |
| **`A5`** | `Paper #95` `FastFeatureCP` | Conformal Prediction | 2-seed Search winner (`1.976`) regressed on 5-seed Confirm (`2.175` vs `2.166`) | `BLOCKED_NOT_CONFIRMED` | `0` | Two-stage Search vs. Confirm caught CQR seed selection bias | [`A5_paper95_FastFeatureCP_2SeedOverfit.md`](./experiments/unsuccessful/A5_paper95_FastFeatureCP_2SeedOverfit.md) |
| **`B2`** | `Paper #92` `KmeansLocalOpt` | Clustering | Blocked `AutoSOTA` `15.79x` runtime; Hartigan lost to equal-compute `3-restart kmeans++` | `BLOCKED_NOT_CONFIRMED` | `0` | Equal-compute multi-restart baselines beat local single-basin swaps | [`B2_paper92_KmeansLocalOpt_EqualComputeLoss.md`](./experiments/unsuccessful/B2_paper92_KmeansLocalOpt_EqualComputeLoss.md) |
| **`B4`** | `Paper #65` `SKEMPI` $\Delta\Delta G$ | Protein Binding | `+6.96 pp` vs `StaB-ddG`, but `LCB = -0.0031` vs `FoldX` across `8` disjoint `#Pdb` buckets | `BLOCKED_NOT_CONFIRMED` | `0` | Complex-disjoint clustering prevents row-split homology pseudoreplication | [`B4_paper65_SKEMPI_DisjointPDB_FoldXGate.md`](./experiments/unsuccessful/B4_paper65_SKEMPI_DisjointPDB_FoldXGate.md) |
| **`B6`** | `Paper #105` `BinaryClassEval` | Label-Shift Eval | PhysioNet CSV stripped upstream; `BinormalGenerator` already at Bayes floor | `NO_SEARCH_GAIN` | `0` | Halt immediately at Search when `S2` headroom is zero | [`B6_paper105_BinaryClassEval_BayesFloor.md`](./experiments/unsuccessful/B6_paper105_BinaryClassEval_BayesFloor.md) |
