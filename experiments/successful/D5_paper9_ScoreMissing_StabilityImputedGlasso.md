# [SUCCESS] D5 — Paper #9 `ScoreMissing`: EM-Warmed Conditional Multiple Imputation + Subsample Stability-Selected ADMM Graphical Lasso on Real `S&P 500`

- **Status:** `PROMOTED & SEALED HOLDOUT WIN` (`HOLDOUT_SEAL_d5.json` `60ac248f800136d1...`, `holdout_guardrails_passed: true`)
- **Wave / Node:** Wave 4 (`Multi-Agent Iterative RSI`), `node-1` (`NVIDIA A100-SXM4-80GB`)
- **Winning Proposer Lineage:** **`Proposer_B_Fast_Explorer`** (`d5_r1_explore_multi_imp_stab`; Round-2 `d5_r2_crossover_em_multiimp_consensus` pushed Search AUPRC even higher to `0.5349` within the `0.008` step margin)
- **Domain:** Missing-Data Graphical Model / Precision-Matrix Edge Recovery on Real `S&P 500` 5-Year Daily Log-Returns (`snp_500_allstocks_5yr.csv`, `3` non-overlapping temporal market eras `[0:410]`, `[420:830]`, `[840:1258]` $\times$ `16` random `36`-stock sub-universes per split under volatility-dependent `~32%` MAR missingness)
- **Primary Metric:** Off-Diagonal Precision Edge-Recovery `AUPRC` (`edge_auprc`, `maximize`, `max_ratio_vs_baseline: eval_wall_clock_sec <= 2.5`)
- **Mandatory Comparators (`required_comparators`):** `d5_paper_msm_score_matching` (`Paper #9` Variational Imputed Score Matching), `d5_base_mean_impute_ridge`, `d5_comp_pairwise_cov_shrink` (PSD-projected pairwise-complete Ledoit-Wolf shrinkage)

## 1. Multi-Agent Round-1 -> Round-2 Trajectory (`Search`)
- **Round-0 Baseline (`d5_paper_msm_score_matching`):** `edge_auprc = 0.4421` (`AUROC = 0.7259`, `9.82s`); `AutoSOTA` (`5x` ensemble `d5_autosota_msm_5x_ens`): `0.4681` (`21.35s = 2.17x` runtime).
- **Round-1 (`4` Independent Agents):**
  - `Proposer D (6-Step Dense EM Inverse)`: `0.4209` (`5.85s` — unpenalized $\Sigma_{\text{EM}}^{-1}$ lacks L1 sparsity)
  - `Proposer C (IPW Horvitz-Thompson PSD ADMM Glasso)`: `0.4384` (`0.72s` — IPW weights increase variance at `32%` missingness)
  - `Proposer A (EM Second-Moment + Multi-Lambda Meinshausen-Bühlmann)`: `0.5117` (`9.17s`, `PROVISIONAL_KEEP`)
  - **`Proposer B (Fast-Explorer: EM-Warmed 5-Draw Conditional Multiple Imputation + `85%`-Subsample Stability-Selected Eigendecomposition ADMM Graphical Lasso)` (`d5_r1_explore_multi_imp_stab`):** **`0.5286` (`AUROC = 0.8160`, `6.58s = 0.67x` of baseline runtime!)** -> **`PROVISIONAL_KEEP` (Finalist)**!

## 2. Confirmation Gate & Single-Touch Sealed Holdout (`HOLDOUT_SEAL_d5.json`)
- **Confirm Gate (`G = 16` equity blocks in `[420:830]`):** `PROMOTE` across all 3 mandatory comparators (`0.6686` vs `0.5284` `MSM`, `0.5106` `Mean-Impute-Ridge`, `0.4182` `Pairwise-Shrinkage`; exact $2^{16}$ `signflip_p = 1.53e-5` on all 3!).
- **Single-Touch Sealed Holdout (`16` unseen equity blocks in `[840:1258]`, `HOLDOUT_SEAL_d5.json` `60ac248f...`):**
  - **Candidate Holdout Edge AUPRC:** **`0.6321`** (`33%` faster than `MSM` baseline!)
  - vs `d5_paper_msm_score_matching` (`0.5152`): **`+11.68 pp` (`+22.68%` relative)** (`95% CI [+10.05, +13.45 pp]`, **`16/16` positive blocks**, exact `signflip_p = 1.53e-5`)
  - vs `d5_base_mean_impute_ridge` (`0.5044`): **`+12.76 pp`** (`95% CI [+11.21, +14.55 pp]`, **`16/16` positive blocks**, exact `signflip_p = 1.53e-5`)
  - vs `d5_comp_pairwise_cov_shrink` (`0.4265`): **`+20.55 pp`** (`95% CI [+18.45, +22.83 pp]`, **`16/16` positive blocks**, exact `signflip_p = 1.53e-5`)
- *(Inference note: Each stage's 16 blocks draw distinct random 36-stock sub-universes from a non-overlapping 418-day market era `[840:1258]`; block-level resampling treats sub-universe blocks within that era as exchangeable replicates.)*

## 3. RSI Learning
1. **Why `Proposer B` (`Fast-Explorer`) beat single-imputation `MSM` and dense `EM`:** Under `32%` volatility-driven MAR missingness, single conditional imputation creates spurious partial correlations between co-imputed stocks; multiplying exact eigendecomposition ADMM Graphical Lasso magnitudes by **cross-imputation + subsample stability frequency** (`(0.35 + 0.65 * sel_freq) * mag`) suppresses imputation-induced false edges across all `16/16` Holdout blocks.
