# [UNSUCCESSFUL / BLOCKED] A1 — Paper #42 `xPatch`: Beat `xPatch` & `AutoSOTA`, Blocked by `Ridge` Parity

- **Status:** `BLOCKED_NOT_CONFIRMED` (`0` Holdout looks consumed — Holdout kept 100% virgin)
- **Wave / Node:** Wave 1 (`Option A`), `node-0` (`NVIDIA A100-SXM4-80GB`)
- **Domain:** Multivariate Time-Series Forecasting (`16` ETT cells, `seq_len=336` locked)
- **Mandatory Comparators:** `xpatch`, `autosota_l336`, `ridge` (`min_improvement = 0.002`)

## 1. Confirm Gate Results (`a1_cand_02_phase1_ms` & `a1_cand_01_linear_last`)
- vs `xpatch` (`0.44816`): Candidate achieved `0.43999` (`LCB = +0.00307`, **`PROMOTE`**)
- vs `autosota_l336` (`0.45258`): Candidate achieved `0.43999` (`LCB = +0.00403`, **`PROMOTE`**)
- vs `ridge` (`0.43943`): Candidate achieved `0.43999` (`signed_gain = -0.00056`, `LCB = -0.00919`, **`ROLLBACK`**); `a1_cand_01` got `0.43894` (`signed_gain = +0.00049 < 0.002` margin, **`ROLLBACK`**)
- **`gate-holdout --phase reserve` Verdict:** `BLOCKED_NOT_CONFIRMED` (`missing_or_failed_comparators: ["ridge:ROLLBACK"]`).

## 2. Root Cause & RSI Learning
- On linear-dominant ETT time-series windows (`[12960, 14400)`), a well-regularized multi-horizon `L=336` Ridge regression (`0.43943`) sits near the Bayes linear predictability ceiling; `xPatch`'s EMA/seasonal-trend nonlinear stream overfits regime drift.
- **Why this non-promotion is a governance triumph:** Without `ridge` in `required_comparators`, an unconstrained RSI loop would have celebrated a `"statistically significant +2.8% SOTA win over xPatch & AutoSOTA"`! Mandatory simple strong baselines prevent illusory deep-learning SOTA claims.
