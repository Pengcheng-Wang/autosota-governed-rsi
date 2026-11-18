# [UNSUCCESSFUL / BLOCKED] D6 — Paper #99 `OLLALanding`: Caught `AutoSOTA` `4x` Step-Count Cheat; Round-2 Crossover Beat `OLLA` on Search & Point Confirm (`-8.2%`) at `5.4x` Speedup, Blocked by `crossed_seed_cluster` Gate (`p = 0.305`)

- **Status:** `BLOCKED_NOT_CONFIRMED` (`0` Holdout looks consumed — Holdout kept 100% virgin)
- **Wave / Node:** Wave 4 (`Multi-Agent Iterative RSI`), `node-1` (`NVIDIA A100-SXM4-80GB`)
- **Finalist & Proposer:** `d6_r2_crossover_qmc_stein_equil` (`Round2_Explorer_Crossover`)
- **Domain:** Equality-Constrained Non-Log-Concave Manifold Sampling ($\{x \in \mathbb{R}^d : h(x) = 0\}$ across `sphere_mog`, `stiefel_2frame`, and `fair_bayes_logit`; `12` manifold tasks $\times$ `3` seeds, hard-capped `max_grad_calls <= 400.0` and `max_manifold_err <= 1e-3`)
- **Mandatory Comparators:** `d6_paper_olla_400`, `d6_comp_clangevin_newton_400`, `d6_comp_rattle_hmc_400`

## 1. Prior Unconstrained `AutoSOTA` Audit Finding (`4x`–`5x` Chain-Length Inflation!)
- `AutoSOTA` claimed `-4.83%` NLL improvement by multiplying MCMC chain steps (`n_steps`).
- Our hard-counted manifold oracle (`BudgetedManifoldProblem`) immediately caught and rolled back `d6_autosota_olla_2000steps_cheat` (`hard_ceiling_failed:max_grad_calls=1600.0>400.0`, `eval_wall_clock_sec_ratio=3.99x > 2.2x`), showing `1,600` steps only lowered `sliced_w1_dist` from `0.06247` to `0.05846`.

## 2. Multi-Agent Round-1 -> Round-2 Trajectory & Confirm Gate Verdict
- **Round-1:** None of the 4 Round-1 proposers beat `d6_paper_olla_400` (`0.06247`) by the `0.004` margin (`Proposer B` got `0.06198`).
- **Round-2 (`d6_r2_crossover_qmc_stein_equil`):** Combining wide manifold initialization + early-phase Stein-repulsive cross-mode exploration (`prog < 0.35`) + vectorized exact unbiased tangent-projected Langevin (`prog >= 0.35`) achieved **`0.05796` at strict `400` oracle calls in `7.69s` (`5.37x` faster than `OLLA-400` and beating even `AutoSOTA`'s illegal `1,600`-step score `0.05846`!)** -> `PROVISIONAL_KEEP`!
- **Confirm Gate (`12` manifold clusters $\times$ `3` seeds):**
  - vs `d6_comp_rattle_hmc_400` (`0.2686`): **`0.05604`** (`gain = +0.2126`, `signflip_p = 0.000244`, **`PROMOTE`**)
  - vs `d6_paper_olla_400` (`0.06105`): **`0.05604`** (`-8.21%` point-mean improvement, `gain = +0.00501`), **but `LCB = -0.00292`, `signflip_p = 0.3052 > 0.0125` (`ROLLBACK`)**
  - vs `d6_comp_clangevin_newton_400` (`0.06092`): **`0.05604`** (`-8.02%` point-mean improvement, `gain = +0.00488`), **but `LCB = -0.00056`, `signflip_p = 0.3057 > 0.0125` (`ROLLBACK`)**
- **`gate-holdout --phase reserve` Verdict:** `BLOCKED_NOT_CONFIRMED` (`0` Holdout looks).

## 3. Root Cause & RSI Learning
- Stein-repulsive early annealing yielded large gains on disconnected multimodal `sphere_mog` tasks while adding slight dispersion on unimodal `fair_bayes_logit` and compact `stiefel_2frame` tasks, leaving cluster-level sign-flip non-significant (`p = 0.305`) across all 12 heterogeneous manifolds despite a `-8.2%` aggregate mean drop. Strict `equal_stratum` + `crossed_seed_cluster` prevented single-family dominance from masquerading as universal manifold-MCMC superiority.
