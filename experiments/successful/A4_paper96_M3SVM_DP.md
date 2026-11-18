# [SUCCESS] A4 — Paper #96 `M3SVM`: $(\varepsilon=3.0, \delta=10^{-5})$-DP Multi-Class Classification

- **Status:** `PROMOTED & SEALED HOLDOUT WIN` (`HOLDOUT_SEAL.json` `8a6421e8bd3150fa...`)
- **Wave / Node:** Wave 1 (`Option A`), `node-2` (`NVIDIA A100-SXM4-80GB`)
- **Domain:** Differentially Private Machine Learning (`8` benchmark `.mat` datasets: `Cornell, Dermatology, Glass, HHAR, ISOLET, ORL, USPS, Vehicle`)
- **Primary Metric:** Multi-Class Classification Accuracy (`equal_stratum` across 8 datasets, `crossed_seed_cluster`, higher is better) subject to `per_stratum_hard_ceilings: {"eps_spent": 3.0}`

## 1. Prior Unconstrained `AutoSOTA` Audit Finding (Critical Privacy Leak Caught)
- `governed_loop_guard.py audit-leaks` statically flagged two `CRITICAL` violations (`DP_NOISE_SUBTRACTION_OR_CLEAN_CACHE`) in `paper-96-M3SVM/main-sklearn.py`:
  - Line 62: `self._base_coef = self.coef_ - noise  # clean weights before noise`
  - Line 111: `noisy_coef = self._base_coef + np.random.normal(0, self._sigma, ...)` (`K=50` ensemble draws)
- By subtracting the DP noise vector and drawing 50 fresh Gaussian perturbations around the clean weights without splitting $\varepsilon$, actual privacy spend blew up to **`\varepsilon_{\text{spent}} = 21.213 > 3.0`** (mechanically blocked at runtime across all 8 strata).
- When properly budgeted at $\varepsilon / \sqrt{50}$ per ensemble member (`autosota_k50_budgeted_eps3`), `AutoSOTA`'s ensemble collapsed to `31.66%` Holdout accuracy (statistically tied with single-model `30.68%`).

## 2. Governed Hypothesis & Candidate Design (`a4_cand_04_adaptive_linear_multinomial`)
- Replaced $K$ independent One-vs-Rest output-perturbed binary SVMs (which split privacy budget $\varepsilon / \sqrt{K}$ across classes) with a single joint multinomial cross-entropy objective whose vector gradient has global $\ell_2$-sensitivity bounded by $\sqrt{2}/(n \lambda)$ regardless of class count $K$, paired with analytical public $(n, d, K, \varepsilon)$ ridge calibration.

## 3. Governed Gate Trajectory (`Search -> Confirm -> Atomic Holdout`)
- **Confirm Gate:** `PROMOTE` vs `pmsvm_dp_base` (`43.57%` vs `30.96%`, `+12.61 pp`, `LCB = +7.00 pp`) and `opera_advcom_dp` (`16.90%`, `LCB = +18.92 pp`), with `eps_spent = 3.000 <= 3.0` on all 8 datasets.
- **Single-Touch Sealed Holdout (`8` datasets $\times$ `8` seeds):**
  - **Candidate Accuracy:** **`43.77%`**
  - vs `pmsvm_dp_base` (`30.68%`): **`+13.09 pp`** (`95% CI [+8.44, +17.90 pp]`)
  - vs `opera_advcom_dp` (`17.75%`): **`+26.03 pp`** (`95% CI [+21.35, +30.97 pp]`)
  - vs `autosota_k50_budgeted_eps3` (`31.66%`): **`+12.11 pp`**

## 4. Conclusion, Caveat & RSI Learning
1. **Scientific caveat (Frontier-Audited):** Each training run satisfies exact $(\varepsilon=3.0, \delta=10^{-5})$-DP and Holdout evaluates the single released mechanism on untouched splits; however, end-to-end DP deployment when selecting among $M$ candidates on private validation splits requires adaptive selection accounting (`Liu & Talwar 2019`; `Papernot & Steinke 2022`).
2. **Meta-RSI lesson:** Formal domain invariants (`eps_spent <= 3.0`) MUST be enforced both statically (`audit-leaks`) and per-stratum at runtime (`per_stratum_hard_ceilings`).
