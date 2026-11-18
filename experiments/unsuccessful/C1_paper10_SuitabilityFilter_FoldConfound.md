# [UNSUCCESSFUL / BLOCKED] C1 — Paper #10 `SuitabilityFilter`: Exposed `num_folds=5->10` Target-Mutation Confound; Tied Calibrated GBDT on WILDS

- **Status:** `BLOCKED_NOT_CONFIRMED` (`0` Holdout looks consumed — Holdout kept 100% virgin)
- **Wave / Node:** Wave 3 (`Option C`), `node-0` (`NVIDIA A100-SXM4-80GB`)
- **Domain:** Unlabeled OOD Model Suitability & Accuracy Estimation (`WILDS`: `fmow`, `civilcomments`, `rxrx1`)
- **Primary Metric:** Shift Accuracy Estimation MAE (`acc_est_mae`, `equal_stratum`, lower is better)
- **Mandatory Comparators:** `sf_paper_base` (Logistic suitability filter), `sf_autosota_10fold`, `equal_compute_calibrated_gbdt`

## 1. Prior Unconstrained `AutoSOTA` Audit Finding (Ground-Truth Label Mutation!)
- In `Paper #10` (`SuitabilityFilter`), each evaluation row tests binary target `ground_truth = (target_shift_acc >= source_fold_j_acc)` across `num_folds` source CV folds.
- `AutoSOTA` mutated `num_folds = 5 -> 10` directly inside the evaluation loop, which simultaneously **doubled the evaluation dataset size** and **inflated `source_fold_j_acc` standard deviation by $\approx \sqrt{2}$** (`fmow` fold-accuracy std jumped from `0.00815` to `0.01130`), altering the ground-truth binary suitability labels themselves!

## 2. Confirm Gate Results (`c1_cand_01_oof_isotonic_logit_blend`)
- Under an invariant evaluation target (predicting true unlabeled OOD accuracy across disjoint WILDS regime shifts):
  - vs `sf_paper_base` (`0.03580` MAE): Candidate achieved `0.03481` (`gain = +0.00099`, `LCB = -0.00004`, `signflip_p = 0.1326`, **`ROLLBACK`**)
  - vs `sf_autosota_10fold` (`0.03476` MAE): Candidate achieved `0.03481` (`gain = -0.00005`, `LCB = -0.00171`, **`ROLLBACK`**)
  - vs `equal_compute_calibrated_gbdt` (`0.03468` MAE): Candidate achieved `0.03481` (`gain = -0.00013`, `LCB = -0.00148`, **`ROLLBACK`**)
  - Furthermore, multi-fold GBDT + isotonic calibration exceeded the `2.5x` wall-clock ratio relative to single-pass logistic `sf_paper_base` (`10.34x > 2.5x`).
- **`gate-holdout --phase reserve` Verdict:** `BLOCKED_NOT_CONFIRMED` (`0` Holdout looks).

## 3. Root Cause & RSI Learning
1. **Once the `num_folds=5->10` label-definition confound is removed, `AutoSOTA`'s `10`-fold GBDT (`0.03476`) has zero advantage over a standard calibrated GBDT (`0.03468`).**
2. **Empirical plateau under severe covariate + concept shift (`WILDS`):** Without target-domain labels or an explicit domain-invariance anchor, confidence-margin histograms (`msp`, `energy`, `entropy`) plateaued around `~3.47 pp` MAE across these unseen geographic/demographic shifts.
