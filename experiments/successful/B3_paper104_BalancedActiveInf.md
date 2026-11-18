# [SUCCESS] B3 — Paper #104 `BalancedActiveInf`: Coverage-Safe OOF-PPI++ & Cube-Projected Active Inference

- **Status:** `PROMOTED & SEALED HOLDOUT WIN` (`HOLDOUT_SEAL.json` `12c2987c0f5060c8...`)
- **Wave / Node:** Wave 2 (`Option B`), `node-1` (`NVIDIA A100-SXM4-80GB`)
- **Domain:** Active Prediction-Powered Statistical Inference (`4` nonlinear/heteroskedastic/heavy-tailed Friedman DGPs, `1,600` Monte Carlo replications per DGP)
- **Primary Metric:** `90%` Confidence Interval Width (`equal_stratum`, lower is better) subject to `per_stratum_hard_floors: {"coverage": 0.885}` (nominal `0.90`, MC binomial SE `0.0075`)

## 1. Prior Unconstrained `AutoSOTA` Audit Finding
- `AutoSOTA`'s `XGBoost lr=0.001 -> 0.1` surrogate fix was **genuinely leak-free on Confirm** (`1.1940 -> 0.9157` width), **but on unseen Holdout seeds (`1,600` MC trials/DGP) it under-covered on 2 of 4 DGPs**:
  - `friedman_std`: coverage **`0.8625 < 0.885`** (nominal `0.90`)
  - `friedman_heavytail`: coverage **`0.8788 < 0.885`**
- Greedy active sampling proportional to in-sample XGBoost residuals created extreme inverse-propensity weights under heavy-tailed noise, eroding finite-sample coverage.

## 2. Governed Hypothesis & Candidate Design (`b3_cand_03_defensive_ppipp_cube`)
- **2-Fold Out-of-Fold `PPI++` Power Tuning ($\hat{\lambda}^* \in [0, 1]$):** Eliminated in-sample surrogate overfitting bias and optimally scaled control-variate subtraction.
- **Owen-Zhou `0.35` Defensive Sampling Floor + Deville-Tillé Cube Variance Projection:** Bounded maximum IPW leverage while preserving covariate balance.

## 3. Governed Gate Trajectory (`Search -> Confirm -> Atomic Holdout`)
- **Confirm Gate:** `PROMOTE` vs `cube_paper_base` (`0.8744` vs `1.1940`, `LCB = +0.3070`) and vs `autosota_xgb_lr01` (`0.9157`, `LCB = +0.0312`).
- **Single-Touch Sealed Holdout (`10` clusters $\times$ `8` seeds $\times$ `4` DGPs = `6,400` MC trials):**
  - **Candidate Mean CI Width:** **`0.7950`** (classical SRS width `3.6307`)
  - vs `cube_paper_base` (`1.0850`): **`-26.73%`** (`95% CI [+0.2834, +0.2966]`, `10/10` blocks, exact `signflip_p = 0.000977`)
  - vs `autosota_xgb_lr01` (`0.8332`): **`-4.58%`** narrower **AND fixed `AutoSOTA`'s coverage failure** (`0.9031, 0.9481, 0.8850, 0.8988` across all 4 DGPs!)

## 4. Conclusion & RSI Learning
1. Without high-replication per-stratum coverage floors (`1,600` MC trials/stratum), active inference optimizers silently trade valid coverage for narrower intervals.
