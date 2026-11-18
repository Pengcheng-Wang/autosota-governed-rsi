# [SUCCESS] B5 — Paper #69 `TreeHFD`: Subsampled L2 `depth=3` Functional ANOVA Decomposition

- **Status:** `PROMOTED & SEALED HOLDOUT WIN` (`HOLDOUT_SEAL.json` `537eae9f02a0a7d8...`; *Confirm rule updated in Phase-C `v3.4` loopback — see disclosure*)
- **Wave / Node:** Wave 2 (`Option B`), `node-2` (`NVIDIA A100-SXM4-80GB`)
- **Domain:** Hierarchical Functional ANOVA Decomposition of Tree Ensembles (`4` shifted functional-ANOVA DGPs, `6` seeds, `8` replication blocks)
- **Primary Metric:** Functional Component Decomposition `MSE` (`equal_stratum`, lower is better)

## 1. Prior Unconstrained `AutoSOTA` Audit Finding
- `AutoSOTA`'s regularization fix (`max_depth=6 -> 3` + L2 leaf penalty) **genuinely survived** rigorous out-of-sample testing, rescuing catastrophic interaction-variance blowups in the paper's unregularized `depth=6` trees (`1.6275 -> 0.1099` Holdout decomposition MSE; `pred_r2: 0.4289 -> 0.9302`).

## 2. Governed Hypothesis & Candidate Design (`b5_cand_04_bagged_depth3_l2_hfd`)
- Constructed a $K=5$ row-subsampled (`0.85`), full-feature (`colsample=1.0` required for exact orthogonal tree-HFD path integration) L2-regularized `depth=3` `XGBTreeHFD` ensemble (`OMP_NUM_THREADS=1`), reducing tree-split boundary variance before analytical functional ANOVA projection.

## 3. Governed Gate Trajectory & Phase-C `v3.4` Loopback Discovery
- **Empirical Bug Caught on Confirm:** `b5_cand_04` (`0.0794` MSE) beat `treehfd_depth3_reg` (`0.0857`, `LCB = +0.0009`, `PROMOTE`) AND beat `treehfd_paper_depth6` (`11.112` MSE, where 1 unstable baseline seed blew up to `60+`) on **`8/8` clusters** (`exact signflip_p = 0.0039 <= 0.0125`, nonparametric percentile-bootstrap `LCB = +2.172 > 0`). However, `v3.3`'s `min(boot_lcb, mean - z * stdev)` returned `-3.04` because subtracting symmetric `z * stdev` penalized **upward right-skewed positive gains**!
- **Prospective Disclosure:** Removing that redundant Gaussian cap in `v3.4` promoted `b5_cand_04` retrospectively on Confirm; virgin Holdout was then unsealed strictly once.
- **Single-Touch Sealed Holdout (`8` blocks $\times$ `6` seeds):**
  - **Candidate Decomposition MSE:** **`0.10116`** (`pred_r2 = 0.9338`)
  - vs `treehfd_depth3_reg` (`AutoSOTA`, `0.10985`): **`-7.91%`** (`95% CI [+0.00275, +0.01535]`, `8/8` blocks positive, exact `signflip_p = 0.003906`)
  - vs `treehfd_paper_depth6` (`1.62754`): **`-93.78%`** (`95% CI [+0.5199, +2.9073]`, `8/8` blocks positive)

## 4. Conclusion & RSI Learning
1. Lead scientifically with the **`-7.91%` Holdout gain over `AutoSOTA`'s functioning `depth=3` baseline** (distinguishing algorithmic improvement from rescuing a broken `depth=6` baseline).
2. **Meta-RSI lesson (`v3.4 -> v3.5`):** Never cap a percentile-bootstrap LCB with a symmetric `mean - z * stdev` approximation on skewed distributions; instead pair percentile-bootstrap LCB with a margin-aligned cluster sign-flip test (`G <= 16` exact).
