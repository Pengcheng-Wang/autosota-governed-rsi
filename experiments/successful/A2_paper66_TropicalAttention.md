# [SUCCESS] A2 — Paper #66 `TropicalAttention`: Log-Domain Semiring GEMM + Causal Scale Normalization

- **Status:** `PROMOTED & SEALED HOLDOUT WIN` (`HOLDOUT_SEAL.json` `1923f0a873368ea4...`)
- **Wave / Node:** Wave 1 (`Option A`), `node-1` (`NVIDIA A100-SXM4-80GB`)
- **Domain:** Neural Algorithmic Reasoning / Out-of-Distribution Generalization (`PrefixSum`, `RangeMinQuery`, `ShortestPathBackup`)
- **Primary Metric:** Pooled-Count `Macro-F1` (`macro_f1_from_counts`, `equal_stratum`, `crossed_seed_cluster`, higher is better)
- **Mandatory Comparators (`required_comparators`):** `tropical` (paper baseline), `adaptive_softmax`, `tropical_equal_runtime` (3-layer equal-runtime Tropical baseline)

## 1. Prior Unconstrained `AutoSOTA` Audit Finding
- `AutoSOTA`'s stored artifact achieved in-distribution/mild-shift gains partly by augmenting training sequences up to OOD length `L=16` (violating strict length-extrapolation blindness) and collapsed to `0.4717` Macro-F1 under `3x` value-range scaling (`V=30`).
- Furthermore, naive PyTorch TropicalAttention materializes an $O(B \cdot H \cdot S^2 \cdot d_h)$ broadcast tensor, starving Tensor Cores and inflating wall-clock latency when multi-step semiring layers are stacked.

## 2. Governed Hypothesis & Candidate Design (`a2_cand_04_tri3step_scalenorm`)
- **Exact Log-Domain Tensor-Core GEMM Semiring Factorization:** Rewrote Smooth Max/Min-Plus tropical attention via stabilized log-domain matrix multiplications (`torch.matmul` on Tensor Cores, `0.757x` parameter count), satisfying the symmetric `run-isolated` `1.80x` runtime ceiling without loosening rules.
- **Causal Per-Instance Scale-Ratio Normalization:** Normalized token magnitude ratios causally prior to tropical routing so `V=1..10 -> V=30` value-scale extrapolation remains invariant.

## 3. Governed Gate Trajectory (`Search -> Confirm -> Atomic Holdout`)
- **Train Split:** Strictly `L=8, V=1..10` (`5` seeds).
- **Confirm Gate (`L=16, V=20`):** `PROMOTE` across all 3 comparators:
  - vs `tropical`: `0.7551` vs `0.6601` (`+9.50 pp`, `LCB = +8.29 pp`)
  - vs `adaptive_softmax`: `0.7551` vs `0.6804` (`+7.48 pp`, `LCB = +6.14 pp`)
  - vs `tropical_equal_runtime`: `0.7551` vs `0.6877` (`+6.74 pp`, `LCB = +6.07 pp`)
- **Single-Touch Sealed Holdout (`L=24`, `V=30`, `Noisy-L20`, fresh seeds `7101..7105`, `12` clusters $\times$ `5` seeds):**
  - **Candidate Macro-F1:** **`0.6682`**
  - vs `tropical` (`0.6030`): **`+6.52 pp`** (`95% CI [+5.66, +7.44 pp]`, `12/12` positive clusters)
  - vs `adaptive_softmax` (`0.6067`): **`+6.15 pp`** (`95% CI [+4.68, +7.93 pp]`, `12/12` positive clusters)
  - vs `tropical_equal_runtime` (`0.6285`): **`+3.97 pp`** (`95% CI [+2.50, +5.52 pp]`, `12/12` positive clusters)

## 4. Conclusion & RSI Learning
1. **Strongest controlled architectural headline:** Even against a budget-matched 3-layer equal-runtime Tropical comparator, causal scale normalization + 3-step tropical semiring composition yields `+3.97 pp` OOD Macro-F1.
2. **Meta-RSI lesson (`v3.1 -> v3.2`):** Measuring candidate wall-clock inside `python -I` subprocesses while running baselines in-process biased `wall_clock_sec` by `~2.1s` `import torch` cold-start overhead; symmetric out-of-process execution (`B3`) is mandatory for fair runtime ceilings.
