# Automated Step-0 Gate-1 Triage of All `612` `AutoSOTA` Repositories

Before spending multi-node A100 GPU hours, we parsed all **`612` repositories** in `AutoSOTA` (`triage/gate1_612_summary_stats.json`):

- **`105` `improved-papers` repositories** across 7 ML domains (`24` time-series, `24` general ML/optimization, `19` conformal/DP/stats/causal, `17` LLM/NLP/reasoning, `11` vision/multimodal/audio, `6` combinatorial/RL/routing, `4` graph/bio/physics).
- **`455` `ICML-2026` repositories** & **`52` `CVPR-2026` repositories** (slimmed code trees with upstream datasets, checkpoints, and `.git` histories stripped).

## Key Static Audit Finding (`42 / 105 = 40.0%` Risk Rate)

Static Gate-1 inspection flagged **`42 / 105` (`40.0%`)** of `improved-papers` with at least one protocol-confound risk before running a single GPU job:

| Static Protocol Risk Category | Flagged Count (`/ 105`) | Representative Examples Verified Empirically |
|---|---:|---|
| **Compute / Ensemble / Multi-Restart Inflation** | `22` (`21.0%`) | `Paper #92 KmeansLocalOpt` (`20` restarts = `15.79x` runtime), `Paper #50 CausalPFN` (`42x` boot $\times$ temp ensemble), `Paper #100 InvMissingData` (`n_pairs 2 -> 8`) |
| **Sub-`0.5%` Unconfirmed Deltas Within Seed Noise** | `11` (`10.5%`) | `Paper #68 SEMPO` (`0.12%`), `Paper #74 LSVI` (`0.02%`), `Paper #78 ConformalAnomaly` (`0.11%`) |
| **Test-Time Solver / Resolution / Lookback Overrides** | `9` (`8.6%`) | `Paper #42 xPatch` & `Paper #88 ChannelNorm` (`seq_len 96 -> 336`), `Paper #2 PINet` (`n_iter_test 50 -> 10`) |
| **LLM-Judge / Proxy-Interface Threshold Shifts** | `2` (`1.9%`) | `Paper #5 TSAE` (`act_threshold_frac 0.01 -> 0.05` changing judge mask density without changing the SAE) |
| **Direct DGP / Privacy / Coverage Violations Caught Deeper in Step-0/Code Audit** | Multiple | `Paper #96 M3SVM` (`_base_coef = coef_ - noise`, `eps=21.21 > 3.0`), `Paper #79 DPFKMeans` (`variance=0.49` DGP tweak + `kmeans_cost_train`), `Paper #103 QDCP` (`epsilon_0: 0.1 -> 0.001` safety margin cut) |

## Throughput Scaling Law for Governed RSI

Across `24 x NVIDIA A100-80GB`, pure GPU `Search + Confirm + Holdout` takes only **3–25 minutes per tractable repository**, whereas **Step-0 dataset hydration, split de-confounding, and out-of-process adapter binding takes ~30–60 minutes per repository** (and dominates completely on the `507` slimmed `ICML/CVPR` repos). Automating static diff/AST leak scanning (`audit-leaks`) and symmetric `run-isolated` wrappers is therefore the primary multiplier on valid autonomous discoveries per GPU-hour.
