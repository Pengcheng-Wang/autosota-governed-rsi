# [UNSUCCESSFUL / BLOCKED] A5 — Paper #95 `FastFeatureCP`: Caught 2-Seed Search Overfitting on Confirm

- **Status:** `BLOCKED_NOT_CONFIRMED` (`0` Holdout looks consumed — Holdout kept 100% virgin)
- **Wave / Node:** Wave 1 (`Option A`), `node-2` (`NVIDIA A100-SXM4-80GB`)
- **Domain:** Conformal Prediction Interval Efficiency (`6` tabular UCI/MEPS datasets, `coverage >= 0.88` per-stratum floor)
- **Mandatory Comparators:** `split_cp_base`, `ffcp_taylor_base`

## 1. Search vs. Confirm Divergence (`a5_cand_02_cqr_pinball`)
- **Search Stage (`2` seeds):** Pinball CQR appeared to shrink mean conformal interval width (`1.976` vs `2.078`, `PROVISIONAL_KEEP`).
- **Confirm Stage (`5` fresh seeds):** Candidate width regressed to **`2.1750`** vs `split_cp_base` **`2.1662`** (`signed_gain = -0.0088`, `LCB = -0.9731`, **`ROLLBACK`**) and `ffcp_taylor_base` **`2.1662`** (**`ROLLBACK`**).
- **`gate-holdout --phase reserve` Verdict:** `BLOCKED_NOT_CONFIRMED`.

## 2. Root Cause & RSI Learning
- Quantile-regression neural heads on small/medium tabular datasets have substantially higher seed-to-seed calibration variance than conditional-mean + residual Split-CP; optimizing over 2 Search seeds selected a lucky quantile draw that vanished on 5 independent Confirm seeds.
- **RSI Learning:** Textbook empirical proof of why `Search` (`PROVISIONAL_KEEP`) MUST be separated from multi-seed `Confirm` (`PROMOTE`) before touching Holdout!
