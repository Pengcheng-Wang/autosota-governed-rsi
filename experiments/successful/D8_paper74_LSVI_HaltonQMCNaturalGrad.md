# [SUCCESS] D8 — Paper #74 `LSVI`: Antithetic Quasi-Monte Carlo Natural-Gradient Trust-Region Full-Covariance Variational Inference

- **Status:** `PROMOTED & SEALED HOLDOUT WIN` (`HOLDOUT_SEAL_d8.json` `560feb8fa0252ba3...`, `holdout_guardrails_passed: true`)
- **Wave / Node:** Wave 4 (`Multi-Agent Iterative RSI`), `node-2` (`NVIDIA A100-SXM4-80GB`)
- **Winning Proposer Lineage:** **`Proposer_B_Fast_Explorer`** (`d8_r1_explore_natgrad_tr`)
- **Domain:** Bayesian Variational Inference (`Full-Covariance Gaussian VI` $q(\theta) = \mathcal{N}(\mu, L L^\top)$ across `16` disjoint finite-sample binary logistic regression posteriors per split drawn from real `ACS_income_AZ` and `COMPAS` cohorts)
- **Primary Metric:** Common-Random-Number `8,192`-Draw Antithetic Monte Carlo `ELBO` Estimate (`maximize`, low-variance CRN estimate of $-\text{KL}(q \| p) + \log p(y)$, `equal_stratum`, `max_ratio_vs_baseline: eval_wall_clock_sec <= 2.2`)
- **Mandatory Comparators (`required_comparators`):** `d8_paper_lsvi` (`Paper #74` Least-Squares / Stein Variational Inference), `d8_comp_laplace_newton` (Newton MAP + inverse Hessian Laplace), `d8_comp_bbvi_reparam_1x` (equal-budget Adam Reparameterization BBVI)

## 1. Prior Unconstrained `AutoSOTA` Audit Finding (`28.7x` Compute Inflation for `0.016%` KL Shift!)
- In `Paper #74` (`LSVI`), `AutoSOTA` replaced fast fixed-sample `LSVI` with an `8`-seed, long two-phase Adam reparameterization BBVI + best-iterate selection, moving KL by `0.016%` (`367.79 -> 367.73`).
- Under symmetric `run-isolated` timing, `d8_autosota_bbvi_8seed_long` took **`56.51s` vs `1.97s` baseline (`28.74x > 2.2x` -> `ROLLBACK`)**.

## 2. Multi-Agent Round-1 -> Round-2 Trajectory (`Search`)
- **Round-0 Baseline (`d8_paper_lsvi`):** `ELBO = -74.8358` (`1.97s`)
- **Round-1 (`4` Independent Agents):**
  - `Proposer D (Jaakkola-Jordan Polya-Gamma EM)`: `-75.7127` (`1.95s`, quadratic bound slack under-disperses skewed posteriors)
  - `Proposer A (Coordinator 11-Node Gauss-Hermite)`: `-74.9841` (`1.97s`)
  - `Proposer C (Halton QMC PyTorch L-BFGS)`: `-74.8517` (`9.27s > 4.33s` runtime cap)
  - **`Proposer B (Fast-Explorer Antithetic Halton QMC Natural-Gradient + Armijo Trust-Region)` (`d8_r1_explore_natgrad_tr`):** **`-74.7834`** (`3.44s = 1.75x <= 2.2x` -> **`PROVISIONAL_KEEP` — only `Proposer B` beat `LSVI` within the `2.2x` runtime budget!**)
- **Round-2:** Full-Cholesky $O(d^4)$ BFGS variants (`-74.7975`, `8.11s`) exceeded the `2.2x` wall-clock cap due to $105 \times 105$ dense Hessian updates, leaving `Proposer B`'s $O(d^3)$ precision natural-gradient trust-region solver as the overall Search winner!

## 3. Confirmation Gate & Single-Touch Sealed Holdout (`HOLDOUT_SEAL_d8.json`)
- **Confirm Gate (`G = 16` disjoint posteriors):** `PROMOTE` across all 3 mandatory comparators (`-76.6172` vs `-76.6711` `LSVI`, `-76.8624` `BBVI-1x`, `-78.3901` `Laplace`, exact $2^{16}$ `signflip_p = 1.53e-5` on all 3!).
- **Single-Touch Sealed Holdout (`16` fresh `ACS` + `COMPAS` posteriors, `HOLDOUT_SEAL_d8.json` `560feb8f...`):**
  - **Candidate Holdout ELBO:** **`-77.3484`**
  - vs `d8_paper_lsvi` (`-77.3947`): **`+0.0463` nats** (`95% CI [+0.0356, +0.0587]`, **`16/16` positive clusters**, exact `signflip_p = 1.53e-5`)
  - vs `d8_comp_bbvi_reparam_1x` (`-77.6005`): **`+0.2522` nats** (`95% CI [+0.2193, +0.2858]`, **`16/16` positive clusters**, exact `signflip_p = 1.53e-5`)
  - vs `d8_comp_laplace_newton` (`-78.9008`): **`+1.5524` nats** (`95% CI [+1.0885, +2.0299]`, **`16/16` positive clusters**, exact `signflip_p = 1.53e-5`)

## 4. RSI Learning
1. **Why Multi-Agent Proposer Diversity was decisive:** On `D8`, the Coordinator's damped Gauss-Hermite fixed-point (`-74.984`) slightly over-stepped without line search, while `Proposer B` (`Fast-Explorer`) paired **deterministic antithetic Halton QMC draws** with **Armijo trust-region line search on natural-gradient precision updates** (`Prec <- (1 - beta) Prec + beta H_QMC`), winning `16/16` Confirm and `16/16` Holdout posteriors inside `1.75x` runtime.
