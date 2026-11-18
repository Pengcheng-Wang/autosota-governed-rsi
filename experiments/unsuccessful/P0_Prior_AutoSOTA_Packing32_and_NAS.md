# [UNSUCCESSFUL / SATURATED] P0 — Prior `AutoSOTA` Reproduction: `packing32` Basin Saturation & `NAS` Carrier Bottlenecks

- **Status:** `REPRODUCED STORED ARTIFACTS ONLY — NO NEW VALIDATED GAIN`
- **Wave:** Prior Baseline Investigation (`research/lab_cycle_results.md`)
- **Domain:** Circle Packing (`packing32`), Neural Architecture Search (`NAS`), and Unconstrained `xPatch`

## 1. What Happened & Why Optimization Failed
1. **`packing32` Local Basin Saturation:** Starting from published high-density `32`-circle configurations, local LLM/SLM perturbations and coordinate polishing were trapped in a narrow non-convex basin with zero validated improvement over multi-start SLSQP/L-BFGS-B.
2. **`NAS` Flat Config Space:** Narrow JSON hyperparameter/cell tweaks lacked structural operator expressiveness and were swamped by training seed variance.
3. **`xPatch` Lookback & TTA Confounding:** Stored `AutoSOTA` gains relied on overriding `seq_len=96 -> 336` and test-time lookback tweaks rather than beating a matched-lookback linear Ridge baseline.

## 2. RSI Governance Learning
- Directly motivated **Step 0 (`check-preconditions` + `audit-leaks`)** in `/governed-loop-optimization`: screen out saturated toy basins (`S2`), enforce equal-lookback/equal-budget baselines (`B2/B5`), and audit diffs statically before spending GPU hours.
