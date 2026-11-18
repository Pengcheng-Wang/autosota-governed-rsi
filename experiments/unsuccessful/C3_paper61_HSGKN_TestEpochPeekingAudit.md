# [UNSUCCESSFUL / BLOCKED] C3 — Paper #61 `HSGKN`: Proved `+7.60 pp` Test-Epoch Peeking Inflation; Honest Nested Splits Blocked Noise Promotion

- **Status:** `BLOCKED_NOT_CONFIRMED` (`0` Holdout looks consumed — Holdout kept 100% virgin)
- **Wave / Node:** Wave 3 (`Option C`), `node-1` (`NVIDIA A100-SXM4-80GB`)
- **Domain:** Graph Classification (`TUDataset`: `IMDB-BINARY` + `IMDB-MULTI`, `16` disjoint stratified graph blocks $\times$ `5` seeds)
- **Primary Metric:** Graph Classification Accuracy (`graph_acc`, `equal_stratum`, `crossed_seed_cluster`, higher is better)
- **Mandatory Comparators:** `hsgkn_paper_honest`, `hsgkn_autosota_se_honest`, `sp_graph_kernel_svm` (Shortest-Path Weisfeiler-Lehman Graph Kernel SVM)

## 1. Controlled Pool-Only Audit of `AutoSOTA` Test-Epoch Peeking (`+7.60 pp` Phantom Gain!)
- In `Paper #61` (`HSGKN` / `GIN` TUDataset convention), `main.py` evaluates test-fold accuracy across training epochs and reports $\max_{e} \text{Acc}_{\text{test}}(e)$. `AutoSOTA` tripled training epochs from `epochs = 500 -> 1500` plus added Squeeze-and-Excitation (`SE`).
- Using **only the 500-graph training pool** (zero access to Confirm or Holdout; evaluating every 5th epoch up to `1500` across 50-graph inner folds), we measured `max-over-eval-epochs` peeking vs. honest inner-validation checkpoint selection:
  - **`500` epochs:** `peek_test_max = 74.60%` vs `honest_inner_val = 68.00%` (**`+6.60 pp`** artificial peeking inflation!)
  - **`1500` epochs (`AutoSOTA`):** `peek_test_max = 75.20%` vs `honest_inner_val = 67.60%` (**`+7.60 pp`** artificial peeking inflation — tripling epochs `500 -> 1500` raised test-peeking by `+0.60 pp` while **degrading** honest inner-validation accuracy by **`-0.40 pp`** due to overfitting!).

## 2. Confirm Gate Results under Honest Inner-Val Stopping (`c3_cand_02_hellinger_kernel_hsgkn_blend`)
- Across `16` disjoint stratified graph blocks $\times$ `5` seeds (`80` evaluations):
  - vs `hsgkn_paper_honest` (`55.76%`): Candidate achieved `57.67%` (`gain = +1.91 pp`, `LCB = -4.87 pp`, `signflip_p = 0.3086`, **`ROLLBACK`**)
  - vs `hsgkn_autosota_se_honest` (`57.42%`): Candidate achieved `57.67%` (`gain = +0.26 pp`, `LCB = -4.60 pp`, **`ROLLBACK`**)
  - vs `sp_graph_kernel_svm` (`57.33%`): Candidate achieved `57.67%` (`gain = +0.34 pp`, `LCB = -2.81 pp`, **`ROLLBACK`**)
- **`gate-holdout --phase reserve` Verdict:** `BLOCKED_NOT_CONFIRMED` (`0` Holdout looks).

## 3. Root Cause & RSI Learning
1. **Small ego-network social graphs (`IMDB-BINARY`, `1,000` unlabeled graphs; `IMDB-MULTI`, `1,500` unlabeled graphs) have high block-to-block variance (`SE ~ 3.5 pp`) and low structural signal beyond degree/shortest-path histograms (`SP-Kernel SVM = 57.33%`).**
2. **Why TUDataset `max-over-epochs` must be banned in RSI:** Tripling epochs `500 -> 1500` improved `max-over-epochs` peeking (`+0.60 pp`) while hurting honest inner-validation generalization (`-0.40 pp`), with a `+7.60 pp` total gap between test-max peeking and honest inner-val checkpointing. Honest inner-validation stopping + `crossed_seed_cluster` gating prevented false SOTA claims.
