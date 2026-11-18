# [UNSUCCESSFUL / BLOCKED] A3 — `FunSearch` Online Bin Packing: Beat `Best-Fit`, Lost to `FunSearch-OR`

- **Status:** `BLOCKED_NOT_CONFIRMED` (`0` Holdout looks consumed — Holdout kept 100% virgin)
- **Wave / Node:** Wave 1 (`Option A`), `node-2` (`causal_online` JSONL stdio isolation)
- **Domain:** Online Combinatorial Optimization (`OR-Library` / Weibull / Shifted streams)
- **Mandatory Comparators:** `best_fit`, `funsearch_or`

## 1. Confirm Gate Results (`a3_cand_03_causal_quantile_match`)
- Executed strictly step-by-step via `governed_loop_guard.py run-isolated --mode causal_online` (preventing any future item lookahead).
- vs `best_fit` (`0.06665` excess bin ratio): Candidate achieved `0.06466` (`LCB = +0.00070`, **`PROMOTE`**)
- vs `funsearch_or` (`0.06241` excess bin ratio): Candidate achieved `0.06466` (`signed_gain = -0.00225`, `LCB = -0.00419`, **`ROLLBACK`**)
- **`gate-holdout --phase reserve` Verdict:** `BLOCKED_NOT_CONFIRMED` (`missing_or_failed_comparators: ["funsearch_or:ROLLBACK"]`).

## 2. Root Cause & RSI Learning
- Hand-crafted / small-budget rational dead-zone templates easily beat classical `Best-Fit` (`+0.20 pp`), but published `FunSearch-OR` evolved non-obvious piecewise polynomial bin-slack penalties across thousands of LLM islands tailored to Weibull item distributions.
- **RSI Learning:** Validates `run-isolated --mode causal_online` (zero lookahead cheating possible) and proves `gate-holdout` blocks partial wins when a stronger domain SOTA heuristic exists.
