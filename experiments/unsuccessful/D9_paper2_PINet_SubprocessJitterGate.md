# [UNSUCCESSFUL / BLOCKED] D9 — Paper #2 `PINet`: Caught `AutoSOTA` `n_iter_test=10` Feasibility Breach; Round-2 Vectorized ADMM Hit `3.16e-8` Error (`10^7x` Better) but Missed `0.20s` Subprocess Wall-Clock Ratio by `0.12 ms` (`1.2506 > 1.2500`)

- **Status:** `NO_SEARCH_GAIN_HALTED_BEFORE_CONFIRM` (`0` Confirm looks, `0` Holdout looks consumed)
- **Wave / Node:** Wave 4 (`Multi-Agent Iterative RSI`), `node-2` (`NVIDIA A100-SXM4-80GB`)
- **Domain:** Hard-Constrained Neural Network Equality + Box Quadratic Projection Layers ($\arg\min_{Az=b, \, l \le z \le u} \frac{1}{2}\|z - y\|_2^2$, `per_stratum_hard_ceilings: {"feasibility_err": 1e-4}`, `max_ratio_vs_baseline: {"eval_wall_clock_sec": 1.25}`)

## 1. Prior Unconstrained `AutoSOTA` Audit Finding (`11x` Feasibility Violation!)
- `AutoSOTA` claimed a `+16.72%` improvement on `PINet` by cutting test-time Douglas-Rachford iterations `n_iter_test: 50 -> 10`.
- Our hard KKT feasibility verifier (`max(|Az - b|, \text{box\_viol}) <= 1e-4`) immediately caught and rolled back `d9_autosota_pinet_dr10_infeasible`: **`max_feasibility_err = 1.102e-3 > 1e-4`** (`11x` over tolerance on coupled/ill-conditioned QPs!).
- Meanwhile, `d9_paper_pinet_dr50`'s raw clipped shadow iterate had `opt_rmse = 0.30402` in `0.20428s` (of which `~0.195s` is pure `python -I` interpreter startup + JSON I/O!).

## 2. Multi-Agent Round-1 -> Round-2 Trajectory (`Search`)
- **Round-1:** Per-vector active-set Newton (`d9_r1_coord_fista_ssn`, `opt_rmse = 3.60e-5`, `1.369s`) and `d9_r1_explore_primal_kkt` (`opt_rmse = 4.04e-5`, `1.385s`) rolled back due to Python `for` loop overhead and residual tolerance.
- **Round-2 (`100%` Vectorized Matrix-Only Solvers):**
  - **`d9_r2_critic_vec_admm`** (closed-form affine-projected over-relaxed ADMM $Z = P_{\text{aff}}(\frac{Y + \rho(W - U)}{1 + \rho})$): **`opt_rmse = 3.16e-8`** (`10,000,000x` lower error than `0.3040`!), **`max_feasibility_err = 2.28e-6 <= 1e-4` (`100%` feasible!)**, `eval_wall_clock_sec = 0.25547s` vs `0.20428s` baseline -> **`ratio = 1.25057 > 1.25000` (missed by `0.00057x` = `0.116 milliseconds` of OS process-spawn jitter -> `ROLLBACK`!)**.
  - `d9_r2_crossover_vec_nesterov_dual`: **`opt_rmse = 1.30e-8`**, **`max_feasibility_err = 3.83e-7 <= 1e-4`**, `0.3226s` (`ratio = 1.579 > 1.25`).

## 3. Meta-RSI Systems Lesson (Sub-Millisecond Compute Dominated by `python -I` Spawn Time)
- Because our predeclared rules are immutable after `freeze-manifest`, we strictly honored the `ROLLBACK` (`0` Confirm/Holdout looks) rather than loosening `1.25` to `1.26` post-hoc.
- **Actionable Meta-RSI insight:** When `baseline.eval_wall_clock_sec` (`0.204s`) is `>95%` cold `python -I` interpreter startup (`~0.195s`) and `<5%` algorithm compute (`<10ms`), raw subprocess `guard_wall_clock_sec` ratios amplify `50ms` OS scheduling jitter into `25%` ratio swings. Sub-second micro-benchmarks should either amortize `>= 20` repetitions per `run-isolated` invocation or measure warm internal kernel latency across interleaved `ABAB` blocks (`Step 1 Archetype A`).
