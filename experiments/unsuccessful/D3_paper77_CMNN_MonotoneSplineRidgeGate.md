# [UNSUCCESSFUL / BLOCKED] D3 — Paper #77 `CMNN`: Beat `CMNN` (`-9.75%`) & `AutoSOTA` (`-9.09%`) with `0.0%` Violations, Blocked by Non-Neural `Monotone-Spline-Ridge`

- **Status:** `BLOCKED_NOT_CONFIRMED` (`0` Holdout looks consumed — Holdout kept 100% virgin)
- **Wave / Node:** Wave 4 (`Multi-Agent Iterative RSI`), `node-0` (`NVIDIA A100-SXM4-80GB`)
- **Finalist & Proposer:** `d3_r1_coord_convex_concave` (`Proposer_A_Coordinator_Math`; `d3_r2_crossover_ridge_warm_hybrid` improved Search further to `0.7717` within `0.0008` of the `0.004` step margin)
- **Domain:** Certified Partial-Order Monotonic Tabular Learning across `4` real bundled datasets (`auto`, `heart`, `compas`, `blog`) with anonymous shuffled monotone-perturbation probes (`per_stratum_hard_ceilings: {"monotonicity_viol_rate": 0.0}`)
- **Mandatory Comparators:** `d3_paper_cmnn_100ep`, `d3_autosota_cmnn_150ep`, `d3_comp_monotone_ridge_spline`

## 1. Search & Confirm Gate Results
- **Every candidate achieved `0.0000%` monotonicity violations across 100% of anonymous monotone probe pairs.**
- **Search (`std_rmse`, lower is better):** `Paper #77 CMNN`: `0.8632`; `AutoSOTA (150ep)`: `0.8591` (`-0.47%`); `Proposer C (GroupSort)`: `0.7842`; `Proposer A (Convex-Concave)`: **`0.7725` (`-10.51%`)**; `Round-2 Ridge-Warm Hybrid`: `0.7717`.
- **Confirm Gate (`12` clusters $\times$ `3` seeds across `auto, heart, compas, blog`):**
  - vs `d3_paper_cmnn_100ep` (`0.8410`): **`0.7590`** (`-9.75%`, `gain = +0.0819`, `signflip_p = 0.0364 > 0.0125`, **`ROLLBACK`**)
  - vs `d3_autosota_cmnn_150ep` (`0.8349`): **`0.7590`** (`-9.09%`, `gain = +0.0759`, `signflip_p = 0.0308 > 0.0125`, **`ROLLBACK`**)
  - vs `d3_comp_monotone_ridge_spline` (**`0.7401`**): **`0.7590`** (`gain = -0.0189`, **`ROLLBACK`**)
- **`gate-holdout --phase reserve` Verdict:** `BLOCKED_NOT_CONFIRMED` (`0` Holdout looks).

## 2. Root Cause & RSI Learning
- **Identical lesson to `A1` (`xPatch` vs `Ridge`) in a new domain:** `Paper #77`'s `MonotonicLinear` (`W.clamp(min=0)`) both starves gradients and forces unconstrained `free_net` embeddings through nonnegative first-layer weights (`0.8410` RMSE). Fixing that expressive bottleneck (`Convex-Concave` & `GroupSort`, `0.7590`) beat `CMNN` and `AutoSOTA` by `>9%`, **yet a clean non-neural Projected Non-Negative I-Spline/Hinge Ridge baseline (`0.7401`, `0.0%` violation) beat every neural model on these tabular sample sizes!**
