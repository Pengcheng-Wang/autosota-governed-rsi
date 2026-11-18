# [SUCCESS] D7 — Paper #73 `RLD`: Discrete Simulated Bifurcation + Burer-Monteiro SDP Memetic Crossover + 2-Flip Escape for Graph Max-Cut

- **Status:** `PROMOTED & SEALED HOLDOUT WIN` (`HOLDOUT_SEAL_d7.json` `5fda27e31da2a14b...`, `holdout_guardrails_passed: true`)
- **Wave / Node:** Wave 4 (`Multi-Agent Iterative RSI`), `node-2` (`NVIDIA A100-SXM4-80GB`)
- **Winning Proposer Lineage:** **`Round2_Explorer_Crossover`** (`d7_r2_crossover_dsb_memetic`, combining `Proposer C` Simulated Bifurcation + `Proposer A` rank-10 Burer-Monteiro SDP rounding + `Proposer B` $\mathbb{Z}_2$-aligned backbone memetic crossover + `Proposer_R2_Critic` vectorized 2-flip local-minimum escape)
- **Domain:** Combinatorial Optimization (`Graph Max-Cut` across `Barabási-Albert`, `Erdős-Rényi`, `6-Regular`, and `Weighted Frustrated` graphs, $N \in \{140, 180\}$, `16` graph clusters $\times$ `3` seeds per split)
- **Primary Metric:** Exact Cut Weight (`maxcut_value`, `maximize`, `equal_stratum`, `crossed_seed_cluster`, `max_ratio_vs_baseline: eval_wall_clock_sec <= 1.6`)
- **Mandatory Comparators (`required_comparators`):** `d7_paper_rlsa_langevin` (`Paper #73` Regularized Langevin Simulated Annealing), `d7_autosota_rlsa_leap` (`AutoSOTA` `LEAP` chain-copy mechanism), `d7_comp_spectral_greedy`

## 1. Multi-Agent Round-1 -> Round-2 Iterative Trajectory (`Search`)
- **Round-0 Baseline (`d7_paper_rlsa_langevin`):** `486.96` (`1.68s`); `AutoSOTA` (`d7_autosota_rlsa_leap`): `487.85` (`+0.18%`, `1.72s` — reproducing `AutoSOTA`'s reported `+0.24%` `LEAP` gain!).
- **Round-1 (`4` Independent Agents):**
  - `Proposer B (Fast-Explorer Spectral Backbone)` (`d7_r1_explore_elite_crossover`): `494.56` (`0.60s`)
  - `Proposer D (Decision-Theory Incremental SA)` (`d7_r1_decision_inc_sa`): `494.90` (`1.67s`)
  - `Proposer A (Coordinator Burer-Monteiro SDP + ILS)` (`d7_r1_coord_bm_sdp_ils`): `496.78` (`0.55s`, `PROVISIONAL_KEEP`)
  - **`Proposer C (Convex-Spectral Ballistic Simulated Bifurcation)` (`d7_r1_spectral_sim_bifurcation`):** **`497.62`** (`0.87s`, **`PROVISIONAL_KEEP` — beat `Coordinator`!**)
- **Round-2 (`Iterative Crossover`):**
  - **`d7_r2_crossover_dsb_memetic`** (transitions ballistic `bSB` into discrete `dSB` $\text{sign}(X) A$ coupling to suppress analog amplitude-heterogeneity traps, pools top cuts with rank-10 Burer-Monteiro SDP hyperplanes, runs $\mathbb{Z}_2$ spin-flip-aligned backbone crossover, and escapes 1-flip local optima via vectorized pairwise `2-flip` gain matrices $\Delta_{uv} = g_u + g_v - 2 s_u s_v A_{uv}$): **`499.40`** (`1.11s = 0.66x` of baseline runtime!) -> **`PROVISIONAL_KEEP` (Finalist)**!

## 2. Confirmation Gate & Single-Touch Sealed Holdout (`HOLDOUT_SEAL_d7.json`)
- **Confirm Gate (`16` graphs $\times$ `3` seeds = `48` runs):** `PROMOTE` across all 3 mandatory comparators (`496.86` vs `485.12` `RLSA`, `485.86` `AutoSOTA-LEAP`, `490.95` `Spectral-Greedy`, exact $2^{16}$ `signflip_p = 1.53e-5` on all 3!).
- **Single-Touch Sealed Holdout (`16` fresh graphs $\times$ `3` seeds = `48` runs, `HOLDOUT_SEAL_d7.json` `5fda27e3...`):**
  - **Candidate Holdout Cut Value:** **`497.64`**
  - vs `d7_paper_rlsa_langevin` (`484.30`): **`+13.34` (`+2.75%`)** (`95% CI [+11.31, +15.50]`, **`16/16` positive clusters**, exact `signflip_p = 1.53e-5`)
  - vs `d7_autosota_rlsa_leap` (`485.35`): **`+12.29` (`+2.53%`)** (`95% CI [+10.34, +14.36]`, **`16/16` positive clusters**, exact `signflip_p = 1.53e-5`)
  - vs `d7_comp_spectral_greedy` (`491.99`): **`+5.64` (`+1.15%`)** (`95% CI [+4.10, +7.43]`, **`16/16` positive clusters**, exact `signflip_p = 1.53e-5`)

## 3. RSI Learning
1. **Local hyperparameter/population tweaks (`AutoSOTA`'s `LEAP`: `+0.22%`) stay trapped inside `RLSA`'s noisy first-order Langevin basin**, whereas switching to symplectic Kerr/Simulated-Bifurcation dynamics (`Proposer C`) + SDP-memetic 2-flip crossover (`Round 2`) achieved **`11x` larger improvement (`+2.75%` vs `+0.22%`) in `34%` less wall-clock time**.
