# [SUCCESS] B1 — Paper #88 `ChannelNorm`: Lookback De-Confounding & Multi-Horizon Drift Synthesis

- **Status:** `PROMOTED & SEALED HOLDOUT WIN` (`HOLDOUT_SEAL.json` `983ee190319a8cdf...`)
- **Wave / Node:** Wave 2 (`Option B`), `node-0` (`NVIDIA A100-SXM4-80GB`)
- **Domain:** Multivariate Long-Horizon Time-Series Forecasting (`16` ETT cells: `ETTh1, ETTh2, ETTm1, ETTm2` $\times$ `{96, 192, 336, 720}`)
- **Primary Metric:** `MSE` (`equal_stratum` across 16 cells, `crossed_seed_cluster`, lower is better, `seq_len=336` locked across ALL arms)

## 1. Controlled Ablation of `AutoSOTA`'s Claimed `-15.2%` Gain
- Evaluating on identical Search splits revealed a major protocol confound:
  - `RMLP (seq_len=96)`: `0.28681` MSE
  - `RMLP (seq_len=336)`: `0.26722` MSE (`-6.83%` from lookback extension alone)
  - `RMLP + AutoSOTA ChannelNorm (seq_len=336)`: `0.26381` MSE (`-1.28%` incremental)
- **85.1% of the raw MSE reduction came from overriding `seq_len=96 -> 336`**, while `ChannelNorm` itself contributed a real but much smaller `-1.28%` improvement at matched `L=336`.

## 2. Governed Hypothesis & Candidate Design (`b1_cand_02_cn_drift_blend`)
- Combined `L=336` `RMLP + ChannelNorm` residual representations with validation-shrunk multi-horizon Ridge linear drift anchors (`train`-only scaler, strictly disjoint physical-time target blocks across `Search`, `Confirm`, and `Holdout`).

## 3. Governed Gate Trajectory (`Search -> Confirm -> Atomic Holdout`)
- **Confirm Gate:** `PROMOTE` vs `rmlp_base_l336` (`0.4389` vs `0.4532`, `LCB = +0.0096`) and vs `rmlp_cn_autosota_l336` (`0.4521`, `LCB = +0.0079`).
- **Single-Touch Sealed Holdout (`16` strata $\times$ `5` seeds):**
  - **Candidate Holdout MSE:** **`0.35679`** (vs `RMLP L=96` diagnostic `0.40830`)
  - vs `rmlp_base_l336` (`0.37342`): **`-4.45%`** (`+0.01663`, `95% CI [+0.01427, +0.01927]`, `16/16` cells positive)
  - vs `rmlp_cn_autosota_l336` (`0.36870`): **`-3.23%`** (`+0.01191`, `95% CI [+0.00970, +0.01407]`, `16/16` cells positive)
  - vs `ridge_l336` (`0.36179`): **`-1.38%`** (`+0.00500`, `95% CI [+0.00171, +0.00899]`, `14/16` cells positive)

## 4. Conclusion & RSI Learning
1. Locking `seq_len=336` across every baseline stripped away lookback inflation while uncovering a genuine `-3.23%` improvement over `AutoSOTA` at matched `L=336` (and beating `Ridge` on `B1`, unlike `xPatch` on `A1`!).
2. **Statistical boundary:** Note that the 16 ETT cells span 4 underlying physical series $\times$ 4 horizons.
