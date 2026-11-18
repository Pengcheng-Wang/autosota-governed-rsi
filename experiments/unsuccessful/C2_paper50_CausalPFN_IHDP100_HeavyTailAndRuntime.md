# [UNSUCCESSFUL / BLOCKED] C2 — Paper #50 `CausalPFN`: Blocked `AutoSOTA` `19.30x` Ensemble Compute; Analytic MKL-TAR Beat All Point Means on `IHDP-100` but Blocked by Heavy-Tail Variance & `2.62x` Runtime Cap

- **Status:** `BLOCKED_NOT_CONFIRMED` (`0` Holdout looks consumed — Holdout kept 100% virgin)
- **Wave / Node:** Wave 3 (`Option C`), `node-0` (`NVIDIA A100-SXM4-80GB`)
- **Domain:** Observational Causal Inference / CATE Estimation (`IHDP-100` real benchmark `.npz` realizations: `25` Search [`r=0..24`] / `35` Confirm [`r=25..59`] / `40` Sealed Holdout [`r=60..99`], strict `run-isolated --mode predict` blindness to counterfactual `mu0, mu1`)
- **Primary Metric:** $\sqrt{\text{PEHE}}$ (`sqrt_pehe`, lower is better, `max_ratio_vs_baseline: eval_wall_clock_sec <= 2.0`)
- **Mandatory Comparators:** `t_learner_gbr_base`, `dr_learner_base` (5-fold cross-fitted Doubly Robust Learner), `equal_compute_t_ens`

## 1. Prior Unconstrained `AutoSOTA` Audit Finding (`19.30x` Inference Inflation!)
- `AutoSOTA` replaced `CausalPFN`'s single forward pass with a `42x` propensity-stratified bootstrap/temperature ensemble (`6` resamples $\times$ `7` temperatures/learning rates = `84` potential-outcome networks per realization).
- Under symmetric `run-isolated --mode predict` measurement (`OMP_NUM_THREADS=1`, `python -I -B`):
  - Single-pass `t_learner_gbr_base`: `14.18s` (`sqrt_pehe = 1.9341`)
  - `autosota_42x_propensity_ens`: **`273.59s` (`19.30x > 2.0x` ceiling -> `ROLLBACK`!)** (`sqrt_pehe = 1.4897`).

## 2. Governed Candidate (`c2_cand_02_loo_mkl_tar_synthesis`) & Confirm Gate Verdict
- Designed an analytic **Closed-Form Leave-One-Out Multi-Kernel + Cross-Fitted Propensity TAR Synthesis** (`c2_cand_02`) running strictly inside `python -I -B` (`numpy` + `torch` only, zero access to `immutable_eval/mu0, mu1`).
- **Confirm Gate (`G = 35` disjoint IHDP realizations, `r=25..59`):**
  - **Candidate achieved the lowest point-mean $\sqrt{\text{PEHE}}$ (`1.5594`) across every baseline:**
    - vs `dr_learner_base` (`2.1588`): **`gain = +0.5994` (`-27.77%`)**, **`LCB = +0.1224 > 0`** (statistically significant!)
    - vs `t_learner_gbr_base` (`1.8748`): **`gain = +0.3154` (`-16.82%`)**, `LCB = -0.1614 < 0`
    - vs `equal_compute_t_ens` (`1.7129`): **`gain = +0.1535` (`-8.96%`)**, `LCB = -0.3292 < 0`
  - **Runtime Tripwire:** Evaluating `24` $N \times N$ kernel matrix inversions across both raw and propensity-augmented arms took `2.616x` of `t_learner_gbr_base` (`> 2.0x` cap -> `baseline_ratio_too_high:eval_wall_clock_sec_ratio=2.6155>2.0`).
- **`gate-holdout --phase reserve` Verdict:** `BLOCKED_NOT_CONFIRMED` (`0` Holdout looks).

## 3. Conclusion & RSI Learning
1. **Why ungoverned `IHDP-100` papers routinely overclaim SOTA:** `IHDP-100` has notorious heavy-tailed realization skewness (a handful of realizations have response surface scales $10\times$ larger than the median). Reporting point-mean $\sqrt{\text{PEHE}}$ (`1.559` vs `1.875`, a `-16.8%` drop!) looks like a dramatic causal-inference breakthrough, yet paired bootstrap across 35 Confirm realizations reveals `LCB = -0.161`!
2. **Systems lesson (`python -I` site-packages & OpenMP contention):** `python -I` implies `-s` (hiding `~/.local/lib/python3.12/site-packages`), and default PyTorch CPU OpenMP spawning on 96-vCPU EPYC nodes slows small `(600, 64)` MLPs by `>15x` unless `OMP_NUM_THREADS=1` / `torch.set_num_threads(1)` is pinned symmetrically across all workers.
