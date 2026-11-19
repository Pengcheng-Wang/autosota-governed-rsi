# `program.md` — Governed Optimization Loop Specification (v3.7)

> **Usage:** Fill every `<placeholder>` before Round 1, run `governed_loop_guard.py check-preconditions` + `audit-splits` + `audit-leaks --allowlist ...` + `freeze-manifest`, record `manifest_root_sha256` outside writable scope, run isolated trials via `run-isolated` (`predict` | `causal_online` | `build_test_bench` | `train_lock_score`), select batch Search winners via `select-search-winner`, confirm via `gate-decision`, and unseal Holdout strictly once via `gate-holdout`.

---

## 1. Objective, Claim Scope, Tier & Go/No-Go Admission Audit

- **Problem Statement:** `<What artifact/system/model is being optimized and why>`
- **Loop Archetype & Isolated Runner:** `<A: Repo/Systems (build_test_bench) | B: Pipeline/Ranking (predict) | C: Distributed ML Training (train_lock_score) | D: Quantization/Efficiency (noninferiority_efficiency) | E: Causal Online (causal_online) | F: Weight ReST^EM/RLVR/Lean>`
- **Claim Scope (`claim_scope`):** `<training_recipe (2-way seed x cluster Satterthwaite t-bound) | fixed_checkpoint | repo_build>`
- **Loop Scale Tier:** `<Tier 1: 10–30 trials | Tier 2: 50+ trials with --proposer-out redaction>`
- **Primary Metric, Statistic & Weighting:** `<metric_name>` (`statistic`: `<mean | median | p95 | p99 | macro_f1_from_counts | ratio_from_sums>`, `aggregation_weighting`: `<equal_stratum | equal_item>`, `direction`: `<maximize | minimize>`)
- **Decision & Margin Mode:** `decision_mode`: `<superiority | noninferiority_efficiency>`, `margin_mode`: `<additive | relative_headroom | log_ratio>`
- **Deployed Incumbent & Mandatory Comparators:** `current_incumbent_id`: `<active_champion_id>`, `required_comparators`: `<["baseline_m0", "strong_simple_baseline", "equal_compute_baseline", "active_champion_id"]>`
- **Confirmation Multiplicity (`max_confirm_looks`):** `<K = 1..3; IUT across M comparators at effective_alpha = alpha / K; verify 2^-G <= alpha / K when G < 20>`
- **Minimum Worthwhile Effect (`eps`) vs. 80%-Power MDE:** `<expected_gain >= eps + (z_{1 - alpha/K} + 0.84) * max(se_boot, se_AA)>`
- **Hard Admission Blockers Verified (`NO_GO` if any unchecked):**
  - [ ] **B1 Uncheatable Verifier:** Deterministic / reference-grounded oracle (+ differential tests vs $M_0$ or `0` Lean `sorry`/`axiom`); zero self-judging LLM on subjective output.
  - [ ] **B2 Disjoint & Deduplicated Split Hierarchy:** Verified via `audit-splits` across Train/Inner-Val, Search (`<N_s>`), Confirmation (`<N_c>`), and Sealed Single-Touch Holdout (`<N_h>`) for zero duplicate feature rows, disjoint entity groups, and chronological cutoffs; Train-only scaler/tokenizer/feature fitting.
  - [ ] **B3 Workload-Native Out-of-Process Scorer:** `IMMUTABLE_MANIFEST.json` frozen (`manifest_root_sha256: <PINNED_HASH>`), baseline & candidates run via `run-isolated` (`.safetensors`/non-pickle checkpoints, trusted tests + `--min-tests-executed`, external FLOPs/VRAM/latency accounting).
  - [ ] **B4 Hermetic `.git`-Free Reset:** `.git`-stripped snapshot (`run-isolated`) + offline env + reproducible reset to `M_0`.
  - [ ] **B5 Proxy Horizon Calibration & Domain Constraints:** `<Short-to-full horizon LR/schedule rank check | per_stratum_hard_floors/ceilings | M0-anchored NI + flip_rate/kl_div>`.

---

## 2. Mutable Surface (`A_t` — Enforced via `audit-leaks --allowlist`)

The optimization loop **may ONLY modify**:
1. `<path/or/glob/in/allowlist>` (verified via `audit-leaks --allowlist ... --base-dir ...`)
2. `<Locked protocol constants verified via audit-leaks --locked-constant KEY=VALUE>`

---

## 3. Immutable Protected Surface (`E_t` — Strictly Read-Only)

Locked via `governed_loop_guard.py freeze-manifest` (`verify-manifest --expected-root-sha256 <PINNED_HASH>`):
1. Out-of-process scorer, accountants & test suites: `<immutable_eval/, tests/>`
2. Split manifests & feature/label stores: `<splits/*>`
3. Governance guard & rules: `<gate_rules_search.json, gate_rules_confirm.json, scripts/governed_loop_guard.py>`

---

## 4. Replication, Crossed Pairing & Anti-Survivorship Protocol

- **Bootstrap Mode:** `<crossed_seed_cluster (seed_id x aligned physical/instance cluster_id) | cluster_1d (ABAB blocks / user sessions)>`
- **Frozen Stratum Weighting:** `<expected_strata listed explicitly; aggregation_weighting = "equal_stratum" so unequal cell sample sizes never reweight benchmark>`
- **Metadata & ID Parity:** Candidate items must match incumbent/baseline `id`, `cluster_id`, `seed_id`, `stratum_id`, and `class_id` 1-for-1 (`missing/altered metadata = immediate fail-closed ROLLBACK`).

---

## 5. Pre-Flight Calibration & Protocol Hygiene Invariants

1. **20–50 Sample Verifier & Leak Audit:** Zero parser false negatives; `audit-leaks` confirms zero `sys.modules`/`validate()` monkey-patching, zero `_base_coef` DP noise subtraction, zero test-split `stratify`/normalization leakage, and zero silent `seq_len`/epoch overrides.
2. **Amdahl / Frontier Headroom Verified:** `<Target module >= X% of e2e latency | Weight RSI learnable fraction (0 < pass < k) >= 15%>`.
3. **Compute & Parameter Parity:** Guardrail `max_ratio_vs_baseline` enforced on guard-measured `wall_clock_sec` and `mean_params`; if a candidate uses `>1.2x` compute, include a compute-matched baseline in `required_comparators`.

---

## 6. Two-Stage Promotion Gate & Fail-Closed Safety Tripwires

- **Search (`decision_stage: "search"`):** Evaluate parallel candidates against the frozen round baseline and select the batch winner order-invariantly via `select-search-winner` (`streak_reset_policy: "confirm_only"` keeps streak pending until Confirm).
- **Confirmation (`decision_stage: "confirm"`):** `PROMOTE` iff `margin_excess_lcb_one_sided > 0` (`LCB > eps` under `margin_mode`, or `M_0`-anchored non-inferiority + `log_speedup` LCB under `noninferiority_efficiency`) at IUT `alpha / max_confirm_looks` across **every** comparator in `required_comparators` + `current_incumbent_id` (`Holdout cannot overturn a failed Confirm gate!`).
- **Fail-Closed Safety Tripwires (Missing / `NaN` / Violated -> Immediate `ROLLBACK`):**
  1. **Global Hard Floors / Ceilings:** `<unit_pass_rate == 1.0, id_f1 >= floor, max_ulp_error <= ceil, prediction_flip_rate <= ceil>`
  2. **Per-Stratum Hard Floors / Ceilings:** `<per_stratum_hard_floors: {"coverage": 0.89}, per_stratum_hard_ceilings: {"eps_spent": 3.0}>`
  3. **Resource / Parity Ratios:** `<max_ratio_vs_baseline: {"wall_clock_sec": 1.50, "mean_params": 1.20, "train_tokens": 1.05, "peak_vram_mb": 1.25}>`

---

## 7. Budget & Automatic Stopping Criteria

- **Budget Caps:** `max_dev_looks = <B>`, `max_confirm_looks = <K>`, `max_consecutive_rollbacks = <3–5>`
- **Halt Rule:** Stop immediately when `halt_recommended == true` or frontier saturation `>95%`.

---

## 8. Hash-Chained Ledger & Atomic Single-Touch Holdout (`gate-holdout`)

- **Ledger (`trial_ledger.jsonl`):** Verified via `verify-ledger`; binds `candidate_artifact_sha256` on every trial.
- **Protected Holdout Seal (`HOLDOUT_SEAL.json`):** Reserved atomically via `gate-holdout --phase reserve` (`O_CREAT | O_EXCL`) **only** after `PROMOTE` is verified for every `required_comparators` + `current_incumbent_id`, then finalized once via `gate-holdout --phase finalize` (storing `holdout_primary_replicated` + per-comparator replication).
