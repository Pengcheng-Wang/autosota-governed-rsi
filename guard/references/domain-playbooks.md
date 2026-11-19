# Domain Playbooks, `gate_rules.json` Schemas & Empirical Post-Mortems (v3.7)

## Table of Contents
1. [Archetype A: Coding Repos, Systems, Kernel, Service & SQL Loops (`build_test_bench`)](#1-archetype-a-coding-repos-systems-kernel-service--sql-loops-build_test_bench)
2. [Archetype B: Ranking, RAG, Prompt & Agent Workflow Loops](#2-archetype-b-ranking-rag-prompt--agent-workflow-loops)
3. [Archetype C: Distributed AI/ML Training (`train_lock_score`) & Constrained Scientific ML](#3-archetype-c-distributed-aiml-training-train_lock_score--constrained-scientific-ml)
4. [Archetype D: Quantization, Distillation & Serving Efficiency (`noninferiority_efficiency`)](#4-archetype-d-quantization-distillation--serving-efficiency-noninferiority_efficiency)
5. [Archetype E: Causal Online Heuristics & Sequential Policies (`causal_online`)](#5-archetype-e-causal-online-heuristics--sequential-policies-causal_online)
6. [Archetype F: Verified Weight-Space Self-Training (`ReST^EM` / RLVR / Lean 4)](#6-archetype-f-verified-weight-space-self-training-restem--rlvr--lean-4)
7. [Canonical `gate_rules.json` Templates (v3.7)](#7-canonical-gate_rulesjson-templates-v37)
8. [Empirical Failure Post-Mortems (Systems, Weight-RSI & `AutoSOTA` Waves 1–5)](#8-empirical-failure-post-mortems-systems-weight-rsi--autosota-waves-15)

---

## 1. Archetype A: Coding Repos, Systems, Kernel, Service & SQL Loops (`build_test_bench`)

- **`.git`-Free Snapshot + Mutable Allowlist (`run-isolated --mode build_test_bench`):** `git worktree` shares `.git/objects` and `.git/logs`, allowing agents to inspect other branches/commits via `git log --all`. Always run inside `build_test_bench` (which strips `.git`, blocks `.pyc/.so/.dll/.pth` & build-hook escapes, enforces `--allowlist` against `--base-dir`, overlays `--trusted-tests-dir`, and checks `--min-tests-executed`).
- **Test the Exact Benchmarked Binary & Include Null-Rebuild `se_AA`:** Release compiler flags (`-O3`, `NDEBUG`, `-ffast-math`) alter semantics relative to debug builds. Hash `--artifact-glob` once after `--build-cmd` and run both `--test-cmd` (including randomized differential tests vs $M_0$) and `--bench-cmd` against that **identical** artifact. Include **null-change rebuilds** when calibrating `se_AA` to capture code-layout variance (Mytkowicz et al. 2009).
- **Replication Unit Is the Interleaved Run (`ABAB`), Never the Request:** Run `>= 10–20` interleaved `ABAB` blocks (`cluster_id = "block_k"`, `require_env_fingerprint_match: true`).
- **Open-Loop Load & Reference Oracles:** Benchmark under open-loop arrivals separating cold vs. warm cache; verify `fp64`/ULP numerical bounds or `NULL`-aware SQL multiset equality (`IS NOT DISTINCT FROM`) across multi-shape workloads.

---

## 2. Archetype B: Ranking, RAG, Prompt & Agent Workflow Loops

- **Offline-to-Online Trust Region:** Enforce `topk_overlap_vs_baseline >= 0.60`, `unjudged_rate_at_k` ceiling, and `IPS_ESS_ratio` floor.
- **Temporal Splits & Proposer Redaction:** Split chronologically, cluster by `user_id`/`session_id`, and pass `--proposer-out` so per-item Dev scores never leak into prompts or few-shot stores.

---

## 3. Archetype C: Distributed AI/ML Training (`train_lock_score`) & Constrained Scientific ML

- **3-Phase Training Isolation (`run-isolated --mode train_lock_score`):**
  1. Launch `torchrun` DDP/FSDP or trainer inside a `.git`-free snapshot with Confirm/Holdout & `immutable_eval` stripped (`--forbid-paths`) and offline env (`HF_HUB_OFFLINE=1`).
  2. Emit **data-only checkpoints** (`.safetensors`, `.npz`, `.json`, `.onnx` — `.pt`/`.pkl` pickle checkpoints are rejected because `pickle` executes arbitrary code during scoring!) and verify `checkpoint_meta.json` `selected_on_split` is a train-carved inner validation split (never Confirm/Holdout, blocking `+7.60 pp` epoch-peeking inflation!).
  3. Lock `checkpoint_sha256`, **delete candidate code**, and run `--score-cmd` in a clean isolated process.
- **Deduplicate & Audit Splits Before Round 1 (`audit-splits`):** Verify zero shared feature vectors (`EXACT_FEATURE_ROW_LEAKAGE`), disjoint `user_id`/`entity_id` groups, and chronological ordering (`--timestamp-key --embargo-gap`). Fit scalers, tokenizers, and ranking popularity/history features strictly on Train; convert cross-tokenizer losses to **bits-per-byte (`bpb`)**.
- **Short-Horizon Proxy Rank Calibration:** Aggressive learning rates, warmups, and cosine schedules often win short Search horizons and lose at full token budgets (Wu et al. 2018; Bjorck et al. 2024). Always validate proxy-to-full-horizon rank correlation first, and run Confirmation at the **full target horizon** with external token/FLOPs/VRAM/latency caps.
- **2-Way Crossed `(seed x block)` Satterthwaite Bound (`claim_scope: "training_recipe"`):** With $S = 3\text{–}5$ affordable training seeds across $G$ blocks, cluster-only sign-flips ignore shared seed variance while seed-only sign-flips hit $2^{-3} = 0.125$. Passing both `seed_id` and `cluster_id` activates unbiased 2-way ANOVA + Welch-Satterthwaite Student-$t$ bounds on leave-one-cell-out jackknife pseudovalues (`macro_f1_from_counts`, `ratio_from_sums`, `equal_stratum` supported).
- **Enforce Mathematical Domain Guarantees Per Stratum & Auto-Require Incumbent:**
  - Set `"current_incumbent_id"` so `gate-holdout` automatically requires `PROMOTE` against the active champion alongside `"required_comparators"`.
  - Use `"margin_mode": "relative_headroom"` or `"log_ratio"` whenever baselines approach `0` or `1` bounds.
  - Conformal prediction (`"per_stratum_hard_floors": {"coverage": 0.885}`) and Differential Privacy (`"per_stratum_hard_ceilings": {"eps_spent": 3.0}`) are enforced on both Confirm and Holdout.

---

## 4. Archetype D: Quantization, Distillation & Serving Efficiency (`noninferiority_efficiency`)

- **Compound $M_0$-Anchored Non-Inferiority + Log-Speedup IUT Gate:** Set `"decision_mode": "noninferiority_efficiency"`, `"max_quality_drop_vs_baseline": 0.005` (anchored strictly to initial `baseline` $M_0$ so multi-round promotions can **never** accumulate "biocreep" quality erosion!), `"efficiency_metric": "latency_ms"`, and `"min_speedup_ratio": 1.20`, paired with `"hard_ceilings"` on `"prediction_flip_rate"` and `"kl_divergence"`.

---

## 5. Archetype E: Causal Online Heuristics & Sequential Policies (`causal_online`)

- **Enforce Step-by-Step Causal Arrival (`run-isolated --mode causal_online`):** Supports `"protocol": "bin_packing"` and `"protocol": "generic_step"` (which pipes only `obs` and `valid_actions` at step $t$ while keeping `action_values` and future steps $t+1 \dots T$ inside the guard).

---

## 6. Archetype F: Verified Weight-Space Self-Training (`ReST^EM` / RLVR / Lean 4)

1. Verify `learnable_fraction >= 0.15` (`0 < successes < k`) before Round 1.
2. Decouple strict SFT format admission from lenient semantic test scoring.
3. Run `audit-leaks` on `.lean` outputs to block uncommented `sorry`, `admit`, custom `axiom`, `native_decide`, and `@[implemented_by`/`extern`.
4. Set `max_tokens >= 3x–5x` median CoT length; admit **only** `finish_reason == "stop"`; sample `<= 2–4` verified traces/task **uniformly** (never `min(len)`).
5. Retrain fresh from base `M_0` every `ReST^EM` round (`FP32` LoRA merge parity `<= 1 ULP`), tracking both `avg@k` and `pass@k` plus external rollout FLOPs.

---

## 7. Canonical `gate_rules.json` Templates (v3.7)

### Multi-Cell OOD / F1 Confirmation Gate (`crossed_seed_cluster` + `macro_f1_from_counts` + `gate-holdout`)
```json
{
  "primary_metric": "ood_macro_f1",
  "primary_statistic": "macro_f1_from_counts",
  "aggregation_weighting": "equal_stratum",
  "bootstrap_mode": "auto",
  "direction": "maximize",
  "decision_stage": "confirm",
  "alpha_one_sided": 0.025,
  "max_confirm_looks": 2,
  "min_improvement": 0.015,
  "required_comparators": ["tropical", "adaptive_softmax", "tropical_equal_runtime"],
  "hard_floors": {
    "id_f1": 0.80
  },
  "max_ratio_vs_baseline": {
    "mean_params": 1.20,
    "wall_clock_sec": 1.50
  },
  "streak_reset_policy": "confirm_only",
  "max_consecutive_rollbacks": 3
}
```

### Constrained Conformal / DP Confirmation Gate (`per_stratum_hard_floors` & `per_stratum_hard_ceilings`)
```json
{
  "primary_metric": "iqr_norm_band_length",
  "primary_statistic": "mean",
  "aggregation_weighting": "equal_stratum",
  "bootstrap_mode": "auto",
  "expected_strata": ["CASP", "STAR", "bike", "communities", "meps_19", "blogData"],
  "direction": "minimize",
  "decision_stage": "confirm",
  "alpha_one_sided": 0.025,
  "max_confirm_looks": 2,
  "min_improvement": 0.015,
  "required_comparators": ["split_cp_base", "ffcp_taylor_base"],
  "hard_floors": {
    "mean_coverage": 0.895
  },
  "per_stratum_hard_floors": {
    "coverage": 0.880
  },
  "per_stratum_hard_ceilings": {
    "eps_spent": 3.0
  },
  "max_consecutive_rollbacks": 3
}
```

### Quantization / Distillation / Kernel $M_0$-Anchored Non-Inferiority + Speedup Gate (`v3.7`)
```json
{
  "primary_metric": "eval_accuracy",
  "primary_statistic": "mean",
  "decision_stage": "confirm",
  "decision_mode": "noninferiority_efficiency",
  "direction": "maximize",
  "alpha_one_sided": 0.025,
  "max_confirm_looks": 2,
  "max_quality_drop_vs_baseline": 0.005,
  "efficiency_metric": "p99_latency_ms",
  "efficiency_direction": "minimize",
  "min_speedup_ratio": 1.25,
  "current_incumbent_id": "prod_int8_v1",
  "required_comparators": ["fp16_base_m0", "prod_int8_v1"],
  "hard_ceilings": {
    "prediction_flip_rate": 0.04,
    "kl_divergence": 0.05
  }
}
```

---

## 8. Empirical Failure Post-Mortems (Systems, Weight-RSI & `AutoSOTA` Waves 1–5)

| Case Study | Violated Principle | Failure Mechanism | Governed v3.7 Remediation |
| :--- | :--- | :--- | :--- |
| **`AutoSOTA` Paper #42 (`xPatch`) & #88 (`ChannelNorm`)** | Silent Protocol Override (`B3`) | Changed `--seq_len 96 -> 336` in shell/config while comparing against `L=96` baseline; under locked `L=336`, `AutoSOTA` model was `2.3%` *worse* than vanilla `xPatch` | `audit-leaks --locked-constant seq_len=336` + frozen out-of-process `eval_locked.py` + `Ridge` comparator |
| **`AutoSOTA` Paper #61 (`HSGKN`, Wave 3)** | Test-Epoch Checkpoint Peeking (`B2`/`B3`) | Selected best epoch across `1,500` test evaluations, inflating accuracy by `+7.60 pp` (`75.20%` vs `67.60%` honest inner-val) | `train_lock_score` enforces `selected_on_split` $\notin$ `{confirm, holdout, test}` + deletes trainer code before scoring |
| **Fairness `COMPAS` & `ACS` (`Wave 5 D2`)** | Pre-Split Feature-Row Duplication (`B2`) | Raw `COMPAS` had `46%` duplicate feature vectors (`4,937 -> 2,665`), leaking identical features across random splits | `audit-splits` blocks `EXACT_FEATURE_ROW_LEAKAGE`, `GROUP_ENTITY_LEAKAGE`, and `TEMPORAL_LOOKAHEAD_LEAKAGE` |
| **Traffic Imputation (`Wave 5 D4`) & Zero-Floor Projection (`D9`)** | Missing Incumbent & Zero-Floor Margin (`S2`/`S5`) | Static 3-comparator gate promoted `24.82` while prior `W4` champion scored `23.17`; `D9` additive `5e-4` margin deadlocked against `Dykstra-50 = 5.22e-4` even at `1e-10` RMSE | `current_incumbent_id` auto-required in `gate-holdout`; `margin_mode: "relative_headroom"` / `"log_ratio"` + Step-0 `S3_ADDITIVE_MARGIN_NEAR_BOUNDARY` |
| **Manifold Sampling (`Wave 5 D6`)** | Off-Manifold Drift & Sequential Search Bias (`B5`) | Euclidean support-point herding lowered SW1 to `0.0555` while drifting off-manifold (`0.0565 > 1e-4`); arrival order penalized `0.0592` behind `0.0611` | Hard `manifold_err <= 1e-4` tripwire caught herding; `select-search-winner` ranks batches order-invariantly |
| **`AutoSOTA` Paper #96 (`M3SVM` DP) & #95 (`FFCP`)** | Formal Invariant Bypass (`B5`) | Cached `_base_coef = coef_ - noise` (`7.1x` $\varepsilon$ leak) and clipped conformal scores on test batch | `audit-leaks` + scorer-owned calibration/DP release + `per_stratum_hard_floors/ceilings` verified on Holdout |
| **GSM8K Parser Asymmetry & STaR Truncation Ratchet** | Verifier & Length Hygiene (`Step 2` / `F`) | Strict regex mis-scored `299/1,319` baseline answers; `max_tokens=192` + `min(len)` caused `-20.08 pp` collapse | 200-sample preflight audit + `max_tokens=768` + `finish_reason=="stop"` + `M_0` reset (`+26.73 pp` turnaround) |
