# [UNSUCCESSFUL / BLOCKED] C5 — Paper #100 `InvMissingData`: Beat `OT-Sinkhorn` & `MICE`, Blocked on `KNN` Margin by Exact `G=16` Sign-Flip (`p = 0.0228 > 0.0125`)

- **Status:** `BLOCKED_NOT_CONFIRMED` (`0` Holdout looks consumed — Holdout kept 100% virgin)
- **Wave / Node:** Wave 3 (`Option C`), `node-2` (`NVIDIA A100-SXM4-80GB`)
- **Domain:** Tabular Missing-Data Imputation (`16` bundled real UCI datasets, disjoint row blocks across `MCAR -> MAR -> MNAR` shift)
- **Primary Metric:** Masked Imputation MAE (`impute_mae`, `equal_stratum` across `16` UCI datasets, lower is better)
- **Mandatory Comparators:** `ot_sinkhorn_paper_np2` (Paper Optimal-Transport Sinkhorn imputer), `iterative_ridge_mice`, `knn_distance_base`

## 1. Prior Unconstrained `AutoSOTA` Audit Finding (`3.66x` Compute Inflation for `0.0005%` Gain!)
- `AutoSOTA` increased `n_pairs = 2 -> 8` Sinkhorn mini-batch pairs per iteration plus cosine LR scheduling.
- On our Search audit across all `16` real UCI datasets:
  - `ot_sinkhorn_paper_np2` (`n_pairs = 2`): `57.58s` wall-clock
  - `ot_sinkhorn_autosota_np8` (`n_pairs = 8`): **`210.83s` wall-clock (`3.662x > 1.60x` ceiling -> `ROLLBACK`)**, improving standardized MAE by only `0.0125`!

## 2. Governed Candidate (`c5_cand_02_masked_cv_mice_knn_ot_synthesis`) & Confirm Gate Verdict
- Designed a **Self-Supervised Masked-CV Synthesis** imputer (`79.29s = 1.38x <= 1.60x` runtime ceiling) that masks an extra `10%` of observed entries inside each test batch (using zero ground-truth missing values!) to convexly weight `Iterative-Ridge-MICE`, `Distance-KNN`, and `OT-Sinkhorn` per dataset.
- **Confirm Gate (`MAR` missingness across `G = 16` real UCI datasets, `alpha = 0.025, K = 2 -> eff_alpha = 0.0125`, `min_improvement = 0.005`):**
  - vs `ot_sinkhorn_paper_np2`: **`gain = +0.0689`**, `LCB = +0.0182`, exact $2^{16}$ `signflip_p = 0.00385 <= 0.0125` -> **`PROMOTE`**!
  - vs `iterative_ridge_mice`: **`gain = +0.0643`**, `LCB = +0.0380`, exact $2^{16}$ `signflip_p = 0.00024 <= 0.0125` -> **`PROMOTE`**!
  - vs `knn_distance_base`: **`gain = +0.0456`**, **`LCB = +0.00167 > 0`**, but margin-aligned exact $2^{16} = 65,536$ Rademacher sign-flip gave **`signflip_p = 0.02283 > eff_alpha = 0.0125`** -> **`ROLLBACK`**!
- *(Technical post-mortem note on scaler hygiene: `79 / 80` dataset-seed items had normal standardized MAE `<= 5.0`, whereas `1 / 80` item (`drug_s14`) had a sparse constant-zero observed column where `nanstd + 1e-8` preceded the `< 1e-8` fallback, inflating that item's raw level to `~58,434.54..58,434.56` with a `0.0194` inter-method residual difference. Keeping `C5` as `BLOCKED_NOT_CONFIRMED` with `0` Holdout looks ensures zero promotion relies on an uncleaned sparse-column scaler.)*
- **`gate-holdout --phase reserve` Verdict:** `BLOCKED_NOT_CONFIRMED` (`missing_or_failed_comparators: ["knn_distance_base:ROLLBACK"]`, `0` Holdout looks).

## 3. Conclusion & RSI Learning
1. **Textbook validation of `v3.5`'s exact `G = 16` margin-aligned Rademacher sign-flip (`test_18`):** Even when a candidate achieves **`PROMOTE` over the published paper (`p = 0.0038`)**, **`PROMOTE` over MICE (`p = 0.00024`)**, AND a **positive bootstrap LCB (`+0.0017`) over `KNN`**, `v3.5`'s exact $2^{16}$ margin-shifted sign-flip (`p = 0.0228 > 0.0125` under `K = 2` multiplicity adjustment) refused promotion rather than relaxing predeclared statistical standards.
2. **Distance-weighted `KNN` is a remarkably strong baseline on small/medium nonlinear UCI tables**, and `required_comparators` ensured our loop evaluated against `MICE` + `KNN` rather than only `OT-Sinkhorn`.
