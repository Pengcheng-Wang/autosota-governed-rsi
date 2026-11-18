# [SUCCESS] D4 — Paper #90 `RSTIB`: Spatial Kriging + Adjacent-Spike Interpolation + Highway Factor Gated Residual Net on Real `PEMS04`

- **Status:** `PROMOTED & SEALED HOLDOUT WIN` (`HOLDOUT_SEAL_d4.json` `6a8bb07a36b81833...`, `holdout_guardrails_passed: true`)
- **Wave / Node:** Wave 4 (`Multi-Agent Iterative RSI`), `node-1` (`NVIDIA A100-SXM4-80GB`)
- **Winning Proposer Lineage:** **`Round2_Explorer_Crossover`** (`d4_r2_crossover_kriging_spectral_factor`, combining `Proposer A` closed-form spatial Ridge-Kriging sensor reconstruction + `Proposer B` top-16 global highway SVD factor trajectory + Round-2 adjacent-step spike interpolation and last-step velocity anchoring)
- **Domain:** Spatial-Temporal Highway Sensor Traffic Forecasting under Realistic Sensor Dropout (`12%`) & Impulse Glitches (`3%`) on Real `PEMS04.npz` (`96` active highway sensors $\times$ `3` channels `[flow, occupancy, speed]`, strict physical-time non-overlapping split across `Train [0:9200]`, `Search [9300:11620]`, `Confirm [11700:14020]`, `Holdout [14100:16420]`, `16` contiguous physical-time blocks per split)
- **Primary Metric:** 12-Horizon Traffic Flow `MAE` (`traffic_mae`, `minimize`, `max_ratio_vs_baseline: eval_wall_clock_sec <= 1.8`)
- **Mandatory Comparators (`required_comparators`):** `d4_paper_rstib_ns12` (`Paper #90` Information-Bottleneck MLP), `d4_base_huber_ridge` (clipped per-sensor linear), `d4_comp_spatial_mlp` (joint 96-sensor cross-spatial bottleneck MLP)

## 1. Prior Unconstrained `AutoSOTA` Audit Finding (`-0.45%` Gain from Disabling IB + `4.17x` MC Sampling!)
- `AutoSOTA` achieved `-0.38%` MAE (`18.49 -> 18.42`) by setting `info_beta = 0.0` (disabling the paper's Information Bottleneck) and inflating test-time MC samples `n_sample: 12 -> 50`.
- On our Search audit under realistic sensor dropout + impulse spikes, `d4_autosota_rstib_ns50_zeroib` improved MAE by only `-0.45%` (`66.34 -> 66.05`) at `1.69x` wall-clock cost (`21.54s` vs `12.78s`), because a per-sensor encoder cannot reconstruct a dropped sensor without spatial conditional inference!

## 2. Multi-Agent Round-1 -> Round-2 Iterative Trajectory (`Search`)
- **Round-0 Baseline (`d4_paper_rstib_ns12`):** `traffic_mae = 66.34` (`12.78s`)
- **Round-1 (`4` Independent Agents):**
  - `Proposer D (MAD-Clipped Top-8 Neighbor Affine)`: `50.29` (`11.60s`)
  - `Proposer C (Graph Laplacian Low-Pass + Gated Huber)`: `43.97` (`17.84s`)
  - `Proposer B (Harmonic Graph Diffusion + PCA Factor AR)`: `41.82` (`23.38s > 22.99s` `1.8x` cap)
  - **`Proposer A (Closed-Form Spatial Ridge-Kriging + Graph ResNet)` (`d4_r1_coord_kriging_resnet`):** **`39.29`** (`16.04s`, **`PROVISIONAL_KEEP` — `-40.78%` MAE!**)
- **Round-2 (`Iterative Crossover`):**
  - `d4_r2_critic_fast_kriging_ms`: **`33.63`** (`15.05s`, **`PROVISIONAL_KEEP`**)
  - **`d4_r2_crossover_kriging_spectral_factor`** (replaces flat window-median spike clamping with adjacent-step $(t-1, t+1)$ interpolation, reconstructs dropped sensors via exact spatial Kriging $\Sigma_{mo}\Sigma_{oo}^{-1} X_o$, anchors output on last-step flow + 3-step velocity, and fuses top-16 global highway SVD factors): **`31.07` (`-53.17%` vs `RSTIB` in `18.32s = 1.43x <= 1.8x`!)** -> **`PROVISIONAL_KEEP` (Finalist)**!

## 3. Confirmation Gate & Single-Touch Sealed Holdout (`HOLDOUT_SEAL_d4.json`)
- **Confirm Gate (`G = 16` contiguous physical-time blocks):** `PROMOTE` across all 3 mandatory comparators (`30.85` vs `64.23` `RSTIB`, `67.33` `Clipped-Ridge`, and `39.50` `Joint-Spatial-MLP`; exact $2^{16}$ `signflip_p = 1.53e-5` on all 3!).
- **Single-Touch Sealed Holdout (`16` unseen future physical-time blocks `[14100:16420]`, `HOLDOUT_SEAL_d4.json` `6a8bb07a...`):**
  - **Candidate Holdout Traffic MAE:** **`31.55`**
  - vs `d4_paper_rstib_ns12` (`66.99`): **`-52.90%` MAE** (`gain = +35.44`, `95% CI [+31.48, +39.43]`, **`16/16` positive blocks**, exact `signflip_p = 1.53e-5`)
  - vs `d4_comp_spatial_mlp` (`41.34`): **`-23.68%` MAE** (`gain = +9.79`, `95% CI [+8.42, +11.40]`, **`16/16` positive blocks**, exact `signflip_p = 1.53e-5`)
  - vs `d4_base_huber_ridge` (`67.97`): **`-53.58%` MAE** (`gain = +36.42`, `95% CI [+33.03, +39.96]`, **`16/16` positive blocks**, exact `signflip_p = 1.53e-5`)
- *(Inference note: Holdout blocks are separated by 10-step buffers across `[14100:16420]` so target horizons never overlap; block-level sign-flip/bootstrap treats 145-step blocks as approximately exchangeable units.)*
