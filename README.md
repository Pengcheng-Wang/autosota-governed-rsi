# Governed Loop Optimization (`RSI`) on `AutoSOTA`

Systematic benchmark audit, protocol de-confounding, and governed Recursive Self-Improvement (`RSI`) across **`AutoSOTA`** using [`/governed-loop-optimization`](https://github.com/Pengcheng-Wang/ai-skills) (`scripts/governed_loop_guard.py` v3.5) and dual Frontier Cowork peer review (`Opus 5.5, Max` & `GPT 6.2 Astra / GPT 6 Astra, Max`).

---

## Core Design Principle: Zero Survivorship Bias

Every experiment attempted — **both confirmed single-touch Holdout wins AND unsuccessful, baseline-tied, Search-overfit, or governance-blocked runs** — is permanently logged in structured Markdown with its preregistered split contract, mandatory comparators, SHA-256 ledger/seal status, root-cause analysis, and RSI learning.

- **Master Ledger of All Experiments:** [`EXPERIMENT_TRACKER.md`](./EXPERIMENT_TRACKER.md)
- **Confirmed & Sealed-Holdout Wins:** [`experiments/successful/`](./experiments/successful/)
- **Unsuccessful, Baseline-Tied & Governance-Blocked Runs:** [`experiments/unsuccessful/`](./experiments/unsuccessful/)
- **Iterative Meta-RSI Evolution (`v2.0 -> v3.5`):** [`docs/META_RSI_SKILL_EVOLUTION.md`](./docs/META_RSI_SKILL_EVOLUTION.md)
- **612-Repo Static Gate-1 Triage Audit:** [`triage/AUTOSOTA_612_TRIAGE_REPORT.md`](./triage/AUTOSOTA_612_TRIAGE_REPORT.md)
- **Frontier Cowork Verbatim Audits:** [`reviews/`](./reviews/)

---

## Campaign Summary (Waves 0–2 Completed; Wave 3 In Progress)

| Wave | Scope | Total Benchmarks | Confirmed & Sealed Holdout Wins | Honest Non-Promotions (`0` Holdout Looks) | Key Governance Failure Modes Caught in `AutoSOTA` |
|---|---|---:|---:|---:|---|
| **Wave 0** | Prior `AutoSOTA` Reproduction | `3` | `0` | `3` | `packing32` basin saturation, `NAS` flat carrier bottleneck, `xPatch` lookback/TTA confound |
| **Wave 1 (`Option A`)** | Meta-RSI Validation (`A1..A5`) | `5` | `2` (`A2`, `A4`) | `3` (`A1`, `A3`, `A5`) | `A4` `_base_coef = coef_ - noise` DP violation (`eps=21.21 > 3.0`), `A1` `Ridge` parity, `A5` 2-seed Search overfit |
| **Wave 2 (`Option B`)** | Multi-Domain Scale (`B1..B6`) | `6` | `3` (`B1`, `B3`, `B5`) | `3` (`B2`, `B4`, `B6`) | `B1` `85%` `seq_len 96->336` confound, `B2` `15.79x` 20-restart inflation + loss to equal-compute `kmeans++`, `B3` active-inference under-coverage (`0.8625 < 0.885`) |
| **Wave 3 (`Option C`)** | Extended Pool (`C1..C6`) | `6` *(running)* | *pending* | *pending* | `C1` WILDS OOD calibration, `C2` `42x` IHDP ensemble, `C3` `max`-over-1500-test-epochs peeking, `C4` DGP `variance=0.49` tamper, `C5` `n_pairs 2->8` inflation, `C6` `epsilon_0` conformal cut |

*Note: Zero third-party `AutoSOTA` source files, third-party datasets, or cloud credentials/secrets are stored in this repository.*
