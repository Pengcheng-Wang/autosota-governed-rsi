# [UNSUCCESSFUL / BLOCKED] D2 — Paper #80 `LatentScoreReweight`: Beat `ERM` (`+18.54 pp`) & `Paper #80` (`+7.77 pp`), Blocked by Mandatory `GroupDRO` Comparator

- **Status:** `BLOCKED_NOT_CONFIRMED` (`0` Holdout looks consumed — Holdout kept 100% virgin)
- **Wave / Node:** Wave 4 (`Multi-Agent Iterative RSI`), `node-0` (`NVIDIA A100-SXM4-80GB`)
- **Finalist & Proposer:** `d2_r1_spectral_poly_minimax` (`Proposer_C_Convex_Spectral`)
- **Domain:** Subpopulation Shift / Worst-Group Classification on Real `ACS_income_AZ` (`14,000` Train rows with bundled `weight.csv`, `18,240` Evaluation rows across `3` disjoint splits $\times$ `16` blocks, `8` `(SEX, minority_race, PINCP)` groups)
- **Mandatory Comparators:** `d2_base_erm_mlp`, `d2_paper_latent_score_reweight` (`weight.csv`), `d2_comp_group_dro`

## 1. Search & Confirm Gate Results
- **Search (`16` blocks):** `Proposer C` (`d2_r1_spectral_poly_minimax`, degree-2 polynomial features + primal-dual minimax group weighting) won Search at **`65.84%`** worst-group accuracy (`2.93s`) vs `47.10%` ERM and `55.90%` `Paper #80 LatentScoreReweight` (`d2_r2_crossover_poly2_mlp_stack` hit `65.63%`).
- **Confirm Gate (`G = 16` disjoint blocks, `6,080` rows):**
  - vs `d2_base_erm_mlp` (`43.85%`): **`62.39%`** (`+18.54 pp`, exact `signflip_p = 3.05e-5`, **`PROMOTE`**)
  - vs `d2_paper_latent_score_reweight` (`54.61%`): **`62.39%`** (`+7.77 pp`, exact `signflip_p = 0.00218 <= 0.0125`, **`PROMOTE`**)
  - vs `d2_comp_group_dro` (`63.37%`): **`62.39%`** (`-0.98 pp`, **`ROLLBACK`**)
- **`gate-holdout --phase reserve` Verdict:** `BLOCKED_NOT_CONFIRMED` (`missing_or_failed_comparators: ["d2_comp_group_dro:ROLLBACK"]`, `0` Holdout looks).

## 2. Root Cause & RSI Learning
- `Paper #80`'s unsupervised diffusion latent-score weights (`weight.csv`) target unknown latent subgroups (`54.61%`), whereas when demographic attributes `(SEX, race)` are observed inside `ACS_income_AZ`, **online exponentiated-gradient `GroupDRO` on a nonlinear MLP (`63.37%`)** is the gold-standard oracle comparator. Including `GroupDRO` in `required_comparators` prevented an illusory `+7.77 pp` SOTA claim over `Paper #80`.
