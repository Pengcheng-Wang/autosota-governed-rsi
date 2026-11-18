# Frontier Cowork Pre-Release Audit of `governed_loop_guard.py` v3.2 & Option-B Protocol (`Opus 5.5, Max`)

- **App / Model:** Microsoft 365 Copilot (`Cowork`, `Model and generation settings: Opus 5.5, Max`)
- **Thread:** `Peer Review of 3-Track AutoSOTA Run` (`cd51eb38-e373-44cf-acb9-269c0a002934`, Turn 2)

---

## Verbatim `Opus 5.5, Max` Audit

**Verdict:** The six defects are fixed; A1/A3/A5 show the gate refusing correctly. Close these fail-open edges first, each with a fault-injection test.

### 1. Remaining Edge Cases Audited & Closed in `v3.3`

- **Budget scope & first-look rules hash:** Record `rules_sha256` in every ledger entry and verify `gate_rules_confirm.json` matches the first Confirm look's `rules_sha256` (blocking mid-run rule swapping); avoid `O_EXCL` on blobfuse-style mounts (use local POSIX/NTFS disk for seal + SHA-pinned ledger).
- **Pending chains when `K` Confirm looks are spent:** If Search outlives `max_confirm_looks = K`, new `PROVISIONAL_KEEP`s could keep zeroing `active_streak` without halting. **Fix (`v3.3`):** Once `confirm_looks_used >= max_confirm_looks` with no remaining Confirm budget, set `halt_recommended = true` terminally even on Search turns.
- **Non-vacuous comparator/stratum checks:** Reject empty `required_comparators`, reject `NaN`/`Infinity` via strict JSON parser (`parse_constant`), and enforce exact item ID + metadata parity.
- **Exact cluster-level sign-flip (arm-swap) test for `G < 20` clusters:** `G >= 2` is too low for percentile bootstrap (`2^-G > alpha/K` cannot reject even if every cluster wins!). **Fix (`v3.3`):** Require `0.5 ** G <= eff_alpha` on Confirm (e.g., `G >= 6` at `alpha=0.025`, `G >= 7` at `alpha=0.0125`) AND for `G < 20`, compute the paired cluster-level Rademacher sign-flip (`arm-swap`) one-sided p-value (`exact` over all `2^G` sign flips for `G <= 15`, `8,000` paired sign-flips for `16 <= G < 20`) and require `signflip_p_one_sided <= eff_alpha`!
- **Tiny float deltas:** Require cluster deltas `> max(1e-12, 0.1 * min_improvement)` for sign-test / constant-shift credit rather than bare `> 0.0` GPU float jitter.
- **Confirm redaction (`--proposer-out`):** On `confirm`, emit only `PROMOTE` / `ROLLBACK` + generic `"guardrail_or_stat_gate"` without leaking stratum names or exact deltas.
- **DP adaptive-selection note (`A4`):** Document adaptive composition / private selection caveat (`Liu-Talwar 2019`; `Papernot-Steinke 2022`) when selecting among multiple DP trials on the same private training split, while Sealed Holdout measures out-of-sample accuracy of the single released mechanism on fresh splits.
- **GitHub push hygiene:** Omit `vendor/AutoSOTA` and third-party datasets from `Pengcheng-Wang/ai-skills` (push only clean self-authored skill code, templates, playbooks, and synthetic unit tests).

### 2. Three Mandatory Protocol Rules for `Option B` Scaling

1. **Freeze the task contract; the guard owns evaluation:** Pin `seq_len`/horizon, splits, train-only scaling, `drop_last=False`, target prevalence, nominal coverage, and `(epsilon, delta)` at Step-0. Retune `ChannelNorm` baselines fairly at `L=336`; fix `BinaryClassEval` shifted-prior Holdout before Search.
2. **Leakage unit = split unit = resampling unit (`G >= 8` clusters):**
   - `SKEMPI` (`paper-65`): Group strictly by `#Pdb` protein complex (`>= 8` disjoint complex clusters per split).
   - `Time series` (`paper-88`): `16` contiguous aligned physical-time blocks.
   - `Simulation/active-sampling` (`paper-69 TreeHFD`, `paper-104 BalancedActiveInf`): Fresh unseen seeds & settings on Holdout with `>= 8` independent replication clusters.
3. **Test constraints as bounds & match compute budgets:** Every compute-sensitive claim (`paper-92 KmeansLocalOpt`) must face an **equal-compute `kmeans++` multi-restart comparator** (`equal_compute_kmeanspp`), and active-inference coverage (`paper-104`) must use sufficient Monte Carlo replications per stratum.
