# Milestone-3 Alignment Audit & `v3.7` Upgrade: Repo-Scale Coding Projects & Distributed AI/ML Training

**Auditors:**
1. **Line-by-Line Guard & Schema Audit** (`governed_loop_guard.py`, `SKILL.md`, `domain-playbooks.md`, `program_template.md`)
2. **M365 Copilot `Cowork` — `Claude Opus 5.5, Max`** (Task `c3619346-da38-4226-9459-4713ba8d2806`, `Worked for 8m`)
3. **M365 Copilot `Cowork` — `GPT 6.2 Astra, Max`** (Tasks `1513df24-29fc-427f-b545-980faadacc3e` & `548f6916-6aed-417d-acaa-125f0394bf6e`)

---

## 1. Motivation & Executive Verdict

Following the completion of Waves 1–5 across `AutoSOTA`, we conducted a dedicated 4-stage alignment audit (`Proposal`, `Eval`, `Gating`, `Checking`) evaluating `/governed-loop-optimization` (`v3.6`) for two primary real-world engineering workloads:
- **Workload A — Distributed AI/ML Model Training & Post-Training** (multi-GPU `torchrun` DDP/FSDP, SFT/RLVR, ranking/retrieval, quantization/distillation)
- **Workload B — Multi-File Coding Projects & Systems Optimization** (C++/Rust/Go/TS/Python/C# repositories with build/test/benchmark pipelines)

All three independent audits converged on the same conclusion:
- **Statistical & Governance Spine = Sound** (`Search -> Confirm -> O_CREAT|O_EXCL Single-Touch Holdout`, Intersection-Union Test across $M$ comparators at $\alpha / K$ without dividing by $M$, `lstat` SHA-256 manifest + external pin, `confirm_only` rollback streak, Round-1 blind + Round-2 redacted-Search crossover).
- **Six High-Confidence Upgrades Required for `v3.7`**, all of which are now implemented in `guard/scripts/governed_loop_guard.py` (`v3.7`) and verified across **`27 / 27` regression tests (`test_01`–`test_27`)**.

---

## 2. Stage-by-Stage Findings & Implemented `v3.7` Upgrades

| Stage | Misalignment / Bug Found in `v3.6` | `v3.7` Fix Implemented in `governed_loop_guard.py` | Regression Test |
| :--- | :--- | :--- | :--- |
| **`Proposal` & `Search`** | Sequential `PROVISIONAL_KEEP` depended on candidate arrival order (observed in Wave-5 `D6`); short-horizon ML Search lacked horizon rank-correlation guidance | Added `select-search-winner` CLI for order-invariant batch Search ranking with deterministic SHA-256 tie-break; added short-to-full-horizon LR/schedule calibration rules in `SKILL.md` & `domain-playbooks.md` | `test_24` |
| **`Eval` (`Workload B`: Coding Repos)** | `python -I` dropped repo sibling imports and lacked multi-file build/test/benchmark execution; `git worktree` shared `.git/objects` & `.git/logs` across branches | Added `.git`-free snapshot materialization, `--workspace-root` repo predict, and **`run-isolated --mode build_test_bench`** (offline env -> build -> binary SHA-256 pin -> `--trusted-tests-dir` + `--min-tests-executed` -> benchmark on exact tested binary) | `test_26` |
| **`Eval` (`Workload A`: ML Training & Causal)** | No runner separated distributed training from scoring or blocked `.pt/.pkl` pickle execution & Confirm-epoch checkpoint peeking (`HSGKN` `+7.60 pp`); `causal_online` was bin-packing-specific | Added **`run-isolated --mode train_lock_score`** (isolated training -> reject pickle checkpoints -> verify `selected_on_split` is train-carved inner-val -> lock SHA-256 & delete candidate code -> isolated scorer) and `"protocol": "generic_step"` in `causal_online` | `test_27` |
| **`Gating` (Margin & Power Exactness)** | For $G \ge 20$, `gate-decision` checked `lcb > 0` instead of `lcb > min_effect` (~50% pass rate at boundary $\Delta = \delta$); `check-preconditions` omitted $\delta$ from `MDE_80%` and lacked $2^{-G} \le \alpha/K$ blocker | Enforced `margin_excess_lcb_one_sided > 0` (`LCB > eff_min_effect`) across **all** $G$; Phipson-Smyth `(1+b)/(1+8000)` MC p-value; `MDE_80% = delta + (z_{1-alpha/K} + 0.84)*SE`; Step-0 `B8` (`2^-G <= alpha/K`) & `B9` (`planned_training_seeds >= 2`) | `test_20` |
| **`Gating` (`3–5` Seed ML Variance)** | Cluster-only sign-flip ignored shared training-seed variance $V_{\text{seed}}/S$, while seed-only sign-flip hit $2^{-3} = 0.125$; naive 3-seed bootstrap passed `[+0.10, +0.12, -0.01]` | Added **unbiased 2-way crossed `(seed x cluster)` ANOVA + Welch-Satterthwaite Student-$t$ bound** on leave-one-cell-out jackknife pseudovalues (`t_max <= 2.0 < t_{0.95,2} = 2.92` whenever any of 3 seeds $\le \delta$, while consistent 3-seed gains promote cleanly) | `test_21` |
| **`Gating` (Incumbent, `0/1` Bounds & Efficiency)** | Missing incumbent check allowed Wave-5 `D4` (`24.82`) to trail prior W4 champion (`23.17`); Wave-5 `D9` additive `5e-4` margin deadlocked near `0.0`; iterative quantization risked multi-round "biocreep" | Auto-required `current_incumbent_id` in `gate-holdout` + Step-0 `S5`; added `margin_mode` (`relative_headroom` / `log_ratio`) + Step-0 `S3`; added $M_0$-anchored compound `decision_mode: "noninferiority_efficiency"` + flip-rate/KL guardrails | `test_22`, `test_23` |
| **`Checking` (`audit-leaks` & `audit-splits`)** | `audit-leaks` lacked mutable-allowlist enforcement, binary/build-hook/test-skip checks, and Lean 4 `sorry`/`axiom` scanning; split deduplication (`COMPAS` `46%` duplicates) was manual | Upgraded `audit-leaks` (`--allowlist --base-dir`, `.pyc/.so/.dll/.pth` & build-hook block, multi-language test-skip & comment-aware Lean 4 scanner) and added **`audit-splits`** (exact feature-row dedup, entity group disjointness, chronological cutoffs) | `test_25` |

---

## 3. Verbatim Frontier Audit Highlights

### A. `Claude Opus 5.5, Max` (`c3619346-da38-4226-9459-4713ba8d2806`)
- *"v3.6's statistical spine is sound for single-process JSON-in/JSON-out tasks. Top risks: (1) no eval mode actually runs Workload A or B, so B3/B4 guarantees silently lapse; (2) the gate omits training-seed variance and the incumbent; (3) audit-leaks is a regex tripwire treated as a security boundary."*
- Recommended `train_lock_score` (`.safetensors` + train-carved inner-val checkpoint rule + isolated scorer), `build_test_bench` (`.git`-free snapshots + exact benchmarked binary hash pin + test-count parity), `LCB > delta` across all $G$, `MDE_80% = delta + (z + 0.84)*SE`, $2^{-G} \le \alpha/K$, 2-way `seed x cluster` variance, $M_0$-anchored NI + log-speedup gates, and `audit-splits` deduplication.

### B. `GPT 6.2 Astra, Max` (`1513df24-29fc-427f-b545-980faadacc3e` & `548f6916-6aed-417d-acaa-125f0394bf6e`)
- *"Keep the Search -> Confirm -> sealed Holdout design. The release blocker is knowing exactly what passed: for A, evidence about a frozen checkpoint is not evidence about retraining; for B, a passing script is not the binary built from a repository."*
- Confirmed IUT across $M$ comparators at $a = \alpha/K$ (without dividing by $M$), Satterthwaite/Kenward-Roger $t$-bounds on $V_{\text{seed}}/S + V_{\text{cluster}}/G + V_{\text{inter}}/(SG)$ when $S = 3\text{–}5$, headroom/log-ratio margins near `0/1` boundaries, $M_0$-anchored non-inferiority + speedup gates, and full-horizon Confirmation for LR/schedule edits.
