---
name: governed-loop-optimization
description: >-
  Design, audit, and execute governed autonomous optimization loops (program.md
  + executable evaluator gates + out-of-process isolation + sealed holdouts)
  across systems/kernel/SQL latency, ranking/RAG/data pipelines, constrained ML
  (conformal, differential privacy, OOD, time-series), agentic code workflows,
  and verified LLM/SLM post-training (ReST-EM, STaR, RLVR). Use whenever the
  user mentions RSI (recursive self-improvement), program.md, autoresearch,
  vibe training, loop engineering, self-improving loops, autonomous
  kernel/query/prompt/config/model tuning, or multi-round benchmark optimization
  — and proactively BEFORE starting any iterative propose-evaluate-keep loop so
  you run the Go/No-Go admission, protocol-leak scan, and sizing audit first
  instead of wasting compute on a gameable or noisy loop.
---

# Governed Loop Optimization (`program.md` + Hardened Executable Guard v3.7)

Autonomous optimization loops ("loop engineering", `autoresearch`, "Vibe Training", or Recursive Self-Improvement) create compounding gains **only when the verifier is harder to game than the target is to improve, candidates run across a hermetic `.git`-free data/build/train boundary, search selection is order-invariant and separated from statistically confirmed promotions, and the final Holdout is cryptographically sealed for single-touch evaluation**. Without mechanical enforcement, multi-round loops silently override benchmark protocols (`seq_len`, test splits, TTA hooks), peek at validation/test checkpoints, leak `.git` history or duplicate feature rows across splits, regress against the deployed incumbent, violate domain invariants (differential privacy, conformal coverage), or shop across Holdout evaluations.

Use this skill in six steps:
1. **Step 0 — Go/No-Go Admission, Split-Hygiene, Leak Audit & Sizing:** Run `check-preconditions` + `audit-splits` + `audit-leaks` to block invalid/leaking loops early (`NO_GO`) and verify `2^-G <= alpha/K` & `MDE_80% = delta + (z_{1 - alpha/K} + 0.84) * SE`.
2. **Step 1 — Archetype, Runner, Resampling & Gate Routing:** Match the target to Repo Coding/Systems (`build_test_bench` + `ABAB` blocks), Distributed ML Training (`train_lock_score` + 2-way Satterthwaite `crossed_seed_cluster`), Pipeline/Ranking, Constrained Scientific ML, Online Causal Policies (`causal_online`), Efficiency/Quantization (`noninferiority_efficiency`), or Weight-Space `ReST^EM`/RLVR.
3. **Step 2 — Pre-Flight Calibration & Immutable Surface Lock:** Audit 20–50 baseline passes/failures, calibrate short-horizon proxy rank correlation (for ML training) or null-rebuild `se_AA` (for compiled repos), and lock evaluators/splits/rules with `freeze-manifest` (pinning `manifest_root_sha256` outside writable scope).
4. **Step 3 — Author `program.md` + `gate_rules.json`:** Instantiate `assets/program_template.md`, predeclaring mutable `--allowlist`, `max_confirm_looks` (`alpha / K`), `current_incumbent_id` + `required_comparators`, `margin_mode`, and fail-closed global + per-stratum safety gates.
5. **Step 4 — Governed Search & Confirmation Loop:** Execute `audit-leaks -> run-isolated -> verify-manifest -> select-search-winner / gate-decision` with an append-only SHA-256 hash-chained `trial_ledger.jsonl` (using `--proposer-out` redaction across rounds).
6. **Step 5 — Atomic Single-Touch Holdout (`gate-holdout`):** Unseal Protected Holdout **strictly once** via `gate-holdout` (`O_CREAT | O_EXCL`), which mechanically verifies `PROMOTE` across **every** mandatory comparator **plus `current_incumbent_id`** and re-checks safety guardrails on Holdout before sealing.

---

## Step 0: Go/No-Go Admission Blockers, Leak Audit & Sizing

Before writing `program.md` or spending compute, run:
```bash
python <skill-dir>/scripts/governed_loop_guard.py check-preconditions --config preflight.json
python <skill-dir>/scripts/governed_loop_guard.py audit-splits --train splits/train.jsonl --confirm splits/confirm.jsonl --holdout splits/holdout.jsonl --group-key user_id --timestamp-key ts
python <skill-dir>/scripts/governed_loop_guard.py audit-leaks --paths <repo_or_diff> --allowlist "src/*.py" --locked-constant seq_len=336
```

### Part A — Hard Admission Blockers (Any Violation = Mandatory `NO_GO`)

| ID | Hard Precondition | Requirement | If Violated -> Mandatory Action |
| :--- | :--- | :--- | :--- |
| **B1** | **Asymmetric, Uncheatable Verifier (`E_t`)** | Deterministic or reference-grounded check (`Cost(Verify) << Cost(Solve)`): compiler/kernel, locked test suite + differential fuzzing vs $M_0$, `fp64`/ULP oracle, Lean 4 kernel (`0` `sorry`/`admit`/custom `axiom`/`native_decide`), mathematical DP/conformal accountant, or locked offline replay. **Self-judging LLMs on subjective prose/style/UX do NOT qualify.** | **NO-GO.** Build deterministic checks/fixtures first, or keep human evaluation in the loop. |
| **B2** | **Disjoint, Deduplicated Split Hierarchy** | Disjoint **Train/Inner-Val** $\to$ **Search** $\to$ **Confirmation Gate** $\to$ **Sealed Single-Touch Holdout**, verified via `audit-splits` for **zero duplicate feature rows** (`COMPAS`-style `46%` row overlap leaks splits!), disjoint entity/user groups, and strict chronological cutoffs ($\max t_{\text{train}} + \text{embargo} \le \min t_{\text{confirm}}$). Scalers, tokenizers, and point-in-time ranking features must fit strictly on Train. | **NO-GO until split & deduplicated.** Selecting checkpoints or reporting on leaked splits manufactures phantom gains. |
| **B3** | **Immutable + Workload-Native Out-of-Process Scorer** | Evaluators, trusted tests, splits, `gate_rules.json`, and `governed_loop_guard.py` are `lstat` SHA-256 locked (`freeze-manifest`). **Both baselines/incumbents and candidates** run through `run-isolated` (`predict`, `causal_online`, `build_test_bench`, or `train_lock_score`): offline env, zero Confirm/Holdout access, `.safetensors`/non-pickle checkpoints (candidate code deleted before scoring!), and external wall-clock/FLOPs/VRAM/token accounting. | **NO-GO until isolated & symmetric.** Never score `.pt`/`.pkl` checkpoints in-process or trust candidate-reported timers/FLOPs. |
| **B4** | **Hermetic `.git`-Free State Reset** | Failed trials leave zero residue and **cannot read git history**: `run-isolated` snapshots workspaces **stripping `.git` objects/logs** (`git worktree` shares `.git/objects`, enabling `git log --all` benchmark exploits!); ML training resets to base `M_0` with fresh compile/temp caches. | **NO-GO until `.git`-free & hermetic.** |
| **B5** | **Proxy Validity, Horizon Calibration & Domain Constraints** | Short-horizon ML Search proxies must verify rank correlation with full-horizon training (aggressive LRs/warmups often win short runs and lose at full token horizon!). Enforce trust-region tripwires (`topk_overlap`, `flip_rate`, `kl_div`), `bits-per-byte` (`bpb`) if tokenizers change, and `per_stratum_hard_floors/ceilings` for conformal/DP invariants. | **Calibrate & enforce inside scorer + `gate_rules.json` before searching.** |

### Part B — Sizing, Seed/Cluster Resolution & Power Diagnostics (`S1–S5`)

1. **`S1` One-Shot Filter & Trial Budget:** Fix deterministic bugs (broken parsers, chat-template flags, CPU DataLoader stalls) in 1 shot first. Require `budget / single_trial_time >= 2` (`>= 8` typical for code/pipeline loops unless a rank-verified fast proxy exists).
2. **`S2` Headroom & Boundary-Margin Check:** Verify Amdahl share (`>= 10%` of e2e latency for systems) or Weight-RSI learnable frontier (`0 < successes < k` on `>= 15%` of tasks). Reject additive `min_improvement` values `>= 50%` of comparator headroom near `0`/`1` bounds (`S3_ADDITIVE_MARGIN_NEAR_BOUNDARY`) — use `margin_mode: "relative_headroom"` or `"log_ratio"`.
3. **`S3` A/A Noise Floor, Discrete `2^-G` Floor & Margin-Adjusted `MDE_80%`:** Measure `se_AA` across `>= 5–10` interleaved `A/A` runs (including **null-change rebuilds** for compiled C++/Rust/CUDA repos to capture binary layout variance!). Predeclare `max_confirm_looks = K`, verify discrete cluster sign-flip resolution $2^{-G} \le \alpha / K$ when $G < 20$ (`B8`), verify `planned_training_seeds >= 2` (`3–5+` recommended) for retraining claims (`B9`), and check:
   $$\text{MDE}_{80\%} = \delta + \left(z_{1 - \alpha/K} + 0.8416\right) \cdot \max(\text{se}_{\text{boot}}, \text{se}_{AA}) \le \text{expected\_total\_gain}$$
4. **`S4` Valley-Crossing Macro-Trials:** Allow predeclared `<= M` coupled refactor steps gated on correctness tests before taking **one** Search/Dev read.
5. **`S5` Mandatory Deployed Incumbent (`current_incumbent_id`):** Always declare the current deployed/prior-wave champion as `current_incumbent_id` so `gate-holdout` automatically requires `PROMOTE` against it alongside static baselines (preventing promotion of candidates that beat `M_0` but regress vs the active champion).

---

## Step 1: Select Archetype, Resampling Engine & Two-Tier Gate Policy

**Critical ordering rule — Outer Harness Before Inner Weights:** Calibrate $h_t$ and freeze $E_t$ *before* training model weights $M_t$.

| Archetype | Runner, Resampling & Gate Mode (`gate_rules.json`) | Mandatory Guardrails |
| :--- | :--- | :--- |
| **A. Coding Repo / Systems / Kernel / SQL** | `run-isolated --mode build_test_bench` (`.git`-free snapshot, binary SHA-256 pin, `--trusted-tests-dir`, `--min-tests-executed`), `1D cluster` over `>= 10–20` interleaved `ABAB` blocks | `100%` tests + executed test-count parity; differential fuzzing vs $M_0$; `fp64`/ULP or SQL `NULL`-aware parity; test the **exact** benchmarked binary |
| **B. Pipeline / Ranking / Agent** | `run-isolated --mode predict` (`--workspace-root`), `cluster_id = user_id / session_id`, `"mean"` or `"ratio_from_sums"`, `audit-splits` temporal cutoffs | `topk_overlap`, `unjudged_at_k`, point-in-time Train-only features, tail-slice `per_stratum_hard_floors`, P95 cost/latency caps |
| **C. Distributed AI/ML Training & Constrained ML** | `run-isolated --mode train_lock_score` (`torchrun` DDP/FSDP $\to$ `.safetensors` lock $\to$ isolated scorer), `crossed_seed_cluster` (**2-way Satterthwaite $t$-bound** across $S$ seeds $\times$ $G$ blocks), `"equal_stratum"`, `"macro_f1_from_counts"` | Train-carved inner-val checkpoint rule (`selected_on_split != confirm/holdout`); full-horizon Confirm for LR/schedule edits; external FLOPs/tokens/VRAM caps; `per_stratum_hard_floors/ceilings` |
| **D. Quantization / Distillation / Serving Efficiency** | `decision_mode: "noninferiority_efficiency"`, $M_0$-anchored `max_quality_drop_vs_baseline`, `efficiency_metric` (`log_ratio` paired LCB $> \log s_{\min}$) | Anchored to base `M_0` (blocks multi-round biocreep!); `prediction_flip_rate` & `kl_divergence` ceilings |
| **E. Online Sequential / Heuristics** | `run-isolated --mode causal_online` (`bin_packing` or hidden-reward `generic_step` stdio), family-stratified `equal_stratum` | Runtime ratio ceiling (`max_ratio_vs_baseline`); distribution-shift holdout slice |
| **F. Verified Weight RSI (`ReST^EM` / RLVR / Lean 4)** | Problem-clustered paired rollouts ($k=8$), `decision_stage: "frontier_explore"`, `audit-leaks` on `.lean` (`0` `sorry`/`admit`/`axiom`/`native_decide`) | `avg@k` + `pass@k` + solved frontier; `finish_reason == "stop"`; uniform `<= 2–4` traces/task; reset to `M_0` |

### Two-Stage Search vs. Confirmation State Machine (`governed_loop_guard.py v3.7`)

- **Search (`decision_stage: "search"` & `select-search-winner`):**
  - For parallel/multi-agent rounds, use **`select-search-winner`** to evaluate all candidates against the **frozen round-start incumbent** and rank tripwire-clean passers **order-invariantly** by shrunk `lcb` (with deterministic hash tie-break), eliminating arrival-order bias.
  - Under `streak_reset_policy: "confirm_only"` (default), `PROVISIONAL_KEEP` stays *pending*—only a confirmed `PROMOTE` resets `unresolved_rollbacks_since_promote` to `0`.
- **Confirmation (`decision_stage: "confirm"`):**
  - **Intersection-Union Test (IUT) Across $M$ Comparators $\times$ Bonferroni Across $K$ Looks:** Requiring simultaneous `PROMOTE` across all $M$ entries in `required_comparators` (plus `current_incumbent_id`) is an IUT that controls single-look Type-I error at $\alpha$ **without dividing by $M$** (Berger 1982), while `effective_alpha = alpha / max_confirm_looks` protects adaptive sequential Confirm looks under arbitrary dependence.
  - **Consistent Margin Enforcement (`margin_excess_lcb_one_sided > 0` Across All $G$):** Requires paired one-sided `(1 - alpha/K)` `LCB > eff_min_effect` for **all** cluster counts $G$ (fixing the $G \ge 20$ boundary hole), paired with an exact $2^G$ ($G \le 16$) or Phipson-Smyth `(1+b)/(1+8000)` ($G > 16$) margin-shifted Rademacher sign-flip gate when $G < 20$. Supports `margin_mode: "additive" | "relative_headroom" | "log_ratio"` and compound $M_0$-anchored `"noninferiority_efficiency"`.
  - **Seed-Aware Crossed Random-Effects Satterthwaite Bound (`crossed_seed_cluster`):** When items carry both `seed_id` ($S \ge 2$ training seeds) and `cluster_id` ($G \ge 2$ blocks) under `claim_scope: "training_recipe"` (default), computes unbiased 2-way ANOVA variance $V_{\text{seed}}/S + V_{\text{cluster}}/G + V_{\text{inter}}/(SG)$ and Welch-Satterthwaite $\nu_{\text{Satt}}$ on leave-one-cell-out jackknife pseudovalues—blocking lucky 3-seed lotteries ($t_{\max} \le 2.0 < t_{0.95, 2} = 2.92$) while allowing consistent 3-seed wins without deadlocking on $2^{-3} = 0.125$. (Set `claim_scope: "fixed_checkpoint"` only when reporting conditional on a specific frozen checkpoint.)

---

## Step 2–5: Canonical CLI Workflow

### 1. Freeze & Verify Immutable Surface
```bash
python <skill-dir>/scripts/governed_loop_guard.py freeze-manifest \
  --manifest IMMUTABLE_MANIFEST.json --root . \
  --paths immutable_eval/ tests/ splits/ gate_rules_search.json gate_rules_confirm.json

python <skill-dir>/scripts/governed_loop_guard.py verify-manifest \
  --manifest IMMUTABLE_MANIFEST.json --expected-root-sha256 <PINNED_ROOT_SHA256>
```

### 2. Allowlist Leak Audit & Workload-Native Isolated Execution
```bash
# Multi-language / Lean / binary-hook / mutable-allowlist check
python <skill-dir>/scripts/governed_loop_guard.py audit-leaks \
  --paths candidate_patch.diff --allowlist "src/model/*.py" --locked-constant seq_len=336

# Option 2A: Single-file or multi-file repo prediction (--workspace-root strips .git & holdout)
python <skill-dir>/scripts/governed_loop_guard.py run-isolated \
  --mode predict --workspace-root . --candidate-script src/worker.py \
  --input-json runs/features_only.json --output-json runs/cand_preds.json \
  --expected-ids-json splits/dev_ids.json --forbid-paths immutable_eval/ splits/holdout/

# Option 2B: Multi-file coding / systems project (build -> pin binary -> trusted tests -> bench)
python <skill-dir>/scripts/governed_loop_guard.py run-isolated \
  --mode build_test_bench --workspace-root . --allowlist "src/*.rs" --base-dir ../clean_base \
  --build-cmd "cargo build --release --offline" --artifact-glob "target/release/my_bin*" \
  --trusted-tests-dir ../immutable_eval/tests --trusted-tests-relpath tests \
  --test-cmd "cargo test --release --offline" --min-tests-executed 42 \
  --bench-cmd "./target/release/my_bin --bench-json {output_json}" \
  --output-json runs/cand_bench.json --forbid-paths ../immutable_eval/

# Option 2C: Distributed AI/ML training (torchrun -> .safetensors lock -> delete code -> isolated scorer)
python <skill-dir>/scripts/governed_loop_guard.py run-isolated \
  --mode train_lock_score --workspace-root . --allowlist "trainer/*.py" \
  --train-cmd "torchrun --nproc_per_node=8 trainer/train.py --out ckpt_out" \
  --checkpoint-relpath ckpt_out \
  --score-cmd "{python} /immutable_eval/score_safetensors.py --ckpt {checkpoint_path} --output-json {output_json}" \
  --output-json runs/cand_val.json --forbid-paths /immutable_eval /splits/confirm /splits/holdout
```

### 3. Batch Order-Invariant Search & Confirmation Gate Decision
```bash
# Rank a batch of parallel Search proposals order-invariantly against a frozen incumbent
python <skill-dir>/scripts/governed_loop_guard.py select-search-winner \
  --candidates runs/search_c1.json runs/search_c2.json runs/search_c3.json \
  --incumbent runs/search_inc.json --baseline runs/search_base.json \
  --rules immutable_eval/gate_rules_search.json --ledger trial_ledger.jsonl --round-id round_01

python <skill-dir>/scripts/governed_loop_guard.py gate-decision \
  --candidate runs/confirm_cand.json \
  --incumbent runs/confirm_comp.json \
  --baseline runs/confirm_base.json \
  --rules immutable_eval/gate_rules_confirm.json \
  --ledger trial_ledger.jsonl \
  --trial-id "confirm_cand04_vs_ridge" \
  --comparator-id "ridge" \
  --candidate-artifacts mutable/candidate.py \
  --proposer-out runs/proposer_verdict.json \
  --hypothesis "Val-shrunk multi-anchor synthesis beats Ridge under locked L=336"
```

### 4. Atomic Single-Touch Protected Holdout (`gate-holdout`)
```bash
# Step A: Atomically reserve write-once HOLDOUT_SEAL.json BEFORE running Holdout
# (Fails closed if ANY mandatory comparator in required_comparators lacks PROMOTE,
# if candidate artifacts changed by 1 byte, or if HOLDOUT_SEAL.json already exists!)
python <skill-dir>/scripts/governed_loop_guard.py gate-holdout \
  --phase reserve \
  --seal-file HOLDOUT_SEAL.json \
  --manifest IMMUTABLE_MANIFEST.json \
  --expected-root-sha256 <PINNED_ROOT_SHA256> \
  --ledger trial_ledger.jsonl \
  --rules immutable_eval/gate_rules_confirm.json \
  --candidate-id "cand_04" \
  --candidate-artifacts mutable/candidate.py

# Step B: Evaluate Holdout once, then finalize seal & append HOLDOUT_SEALED to ledger
python <skill-dir>/scripts/governed_loop_guard.py gate-holdout \
  --phase finalize \
  --seal-file HOLDOUT_SEAL.json \
  --manifest IMMUTABLE_MANIFEST.json \
  --expected-root-sha256 <PINNED_ROOT_SHA256> \
  --ledger trial_ledger.jsonl \
  --rules immutable_eval/gate_rules_confirm.json \
  --candidate-id "cand_04" \
  --candidate-artifacts mutable/candidate.py \
  --candidate-holdout runs/holdout_cand_04.json \
  --comparator-holdout xpatch=runs/holdout_xpatch.json \
  --comparator-holdout ridge=runs/holdout_ridge.json
```

---

## Multi-Agent Proposer Tournament & Iterative Crossover Pattern

While `governed_loop_guard.py` guarantees **zero false positives**, **true-positive discovery rate** scales with proposer diversity across iterations (within our 9-benchmark `AutoSOTA` Wave-4 paired ablation, 3 of the 5 confirmed Holdout wins required non-coordinator proposers or Round-2 crossover):
1. **Round 1 — Blind Orthogonal Proposer Ensemble:** Query 3–4 independent proposers with distinct inductive biases (e.g., *Sufficient-Statistics/Analytical*, *Explorative/Active/Quasi-MC*, *Convex-Duality/Spectral*, *Bayesian Shrinkage/Decision-Theory*, plus frontier Cowork models when available), passing **only** the `program.md` interface and invariant rules (`0` evaluation data access). Materialize each proposal as a standalone `.py` file executed via `run-isolated`.
2. **Round 2 — Redacted Search-Feedback Crossover:** Pass **only** redacted Search verdicts (`--proposer-out` + Search runtime/feasibility tripwires; **never** Confirm or Holdout) into a Round-2 synthesis pass that hybridizes complementary Round-1 mechanisms and fixes Search runtime/constraint bottlenecks before spending any of the `K` Confirmation looks.
3. **Sub-Second `run-isolated` Spawn Amortization:** Cold `python -I` startup adds `~0.18–0.22s`. Ensure each `run-isolated` batch performs `>= 1.0s` of real work (or repeat/batch queries inside the worker) so `~10–50ms` OS spawn jitter does not trigger spurious `max_ratio_vs_baseline` rollbacks on sub-millisecond kernels.

---

## Bundled Resources

- **`scripts/governed_loop_guard.py`** — Zero-dependency Python 3.9+ CLI (`v3.7`: `freeze-manifest`, `verify-manifest`, `check-preconditions`, `audit-leaks`, `audit-splits`, `run-isolated` [`predict`, `causal_online`, `build_test_bench`, `train_lock_score`], `select-search-winner`, `gate-decision`, `gate-holdout`, `verify-ledger`).
- **`tests/test_governed_loop_guard.py`** — 27-test adversarial & statistical regression suite (`A/A` null, `SE=0` fail-closed, rollback-streak replay, cluster-splitting tamper, `equal_stratum` + stratum multiplicity, 2-way crossed `macro_f1_from_counts` + 3-seed Satterthwaite lottery block, `G=24` `LCB > delta` & `MDE + delta` enforcement, auto-required `current_incumbent_id`, `relative_headroom` & `log_ratio` zero-floor margins, $M_0$-anchored `noninferiority_efficiency` biocreep block, order-invariant `select-search-winner`, `audit-leaks` v2 allowlist/Lean/binary/test-skip checks, `audit-splits` row/group/time leak checks, `.git`-free `build_test_bench` & `train_lock_score` `.safetensors`/checkpoint-peeking blocks, and atomic `gate-holdout` guardrail verification).
- **`assets/program_template.md`** — 8-section `program.md` contract template (`v3.7`).
- **`references/domain-playbooks.md`** — Domain configurations, coding & ML-training playbooks, `gate_rules.json` schemas, and empirical post-mortems.
