# Governed Loop Optimization (`RSI`) on `AutoSOTA`

Systematic benchmark audit, protocol de-confounding, and governed Recursive Self-Improvement (`RSI`) across **`AutoSOTA`** using bundled [`guard/`](./guard/) (`/governed-loop-optimization v3.7`, `guard/scripts/governed_loop_guard.py`, **`27 / 27` adversarial, statistical & isolation regression tests**) and dual Frontier Cowork / `rubber-duck` peer review (`Opus 5.5, Max` & `GPT 6.2 Astra / GPT 6 Astra, Max`).

> **Companion Wave-5 Frontier `Cowork` Tournament Repository (`Opus 5.5, Max` + `GPT 6.2 Astra, Max` Multi-Round Crossover & Dual Milestone Audits):** [`Pengcheng-Wang/autosota-frontier-cowork-rsi`](https://github.com/Pengcheng-Wang/autosota-frontier-cowork-rsi)

---

## Core Design Principle: Zero Survivorship Bias

Every experiment attempted — **both confirmed single-touch Holdout wins AND unsuccessful, baseline-tied, Search-overfit, or governance-blocked runs** — is permanently logged in structured Markdown with its preregistered split contract, mandatory comparators, SHA-256 ledger/seal status, root-cause analysis, and RSI learning.

- **Master Ledger of All Experiments:** [`EXPERIMENT_TRACKER.md`](./EXPERIMENT_TRACKER.md)
- **Bundled Hardened Skill & Executable Guard (`v3.7`, `27/27` Tests):** [`guard/`](./guard/) (`SKILL.md`, `scripts/governed_loop_guard.py`, `tests/test_governed_loop_guard.py`, `references/domain-playbooks.md`, `assets/program_template.md`)
- **Confirmed & Sealed-Holdout Wins (`12`):** [`experiments/successful/`](./experiments/successful/)
- **Unsuccessful, Baseline-Tied & Governance-Blocked Runs (`15` reports / `17` benchmarks, `0` Holdout looks consumed):** [`experiments/unsuccessful/`](./experiments/unsuccessful/)
- **Multi-Agent Iterative Proposer Study (`Wave 4: D1–D9`):** [`docs/MULTI_AGENT_PROPOSER_RSI_STUDY.md`](./docs/MULTI_AGENT_PROPOSER_RSI_STUDY.md)
- **Complete `105 / 105` `AutoSOTA` Benchmark Paper Master Audit:** [`triage/ALL_105_IMPROVED_PAPERS_COMPLETE_AUDIT.md`](./triage/ALL_105_IMPROVED_PAPERS_COMPLETE_AUDIT.md)
- **Iterative Meta-RSI Evolution (`v2.0 -> v3.7`):** [`docs/META_RSI_SKILL_EVOLUTION.md`](./docs/META_RSI_SKILL_EVOLUTION.md)
- **612-Repo Static Gate-1 Triage Audit:** [`triage/AUTOSOTA_612_TRIAGE_REPORT.md`](./triage/AUTOSOTA_612_TRIAGE_REPORT.md)
- **Frontier Cowork Verbatim Audits (incl. `v3.7` Coding & ML Alignment):** [`reviews/`](./reviews/)

---

## Campaign Summary (Waves 0–4 Completed — `29` Live Benchmarks Executed + `105 / 105` Full-Pool Paper Audit)

| Wave | Scope | Total Benchmarks | Confirmed & Sealed Holdout Wins | Honest Non-Promotions (`0` Holdout Looks) | Key Governance Failure Modes Caught in `AutoSOTA` |
|---|---|---:|---:|---:|---|
| **Wave 0** | Prior `AutoSOTA` Reproduction | `3` (`P0`) | `0` | `3` | `packing32` basin saturation, `NAS` flat carrier bottleneck, `xPatch` lookback/TTA confound |
| **Wave 1 (`Option A`)** | Meta-RSI Validation (`A1..A5`) | `5` | `2` (`A2`, `A4`) | `3` (`A1`, `A3`, `A5`) | `A4` `_base_coef = coef_ - noise` DP violation (`eps=21.21 > 3.0`), `A1` `Ridge` parity, `A5` 2-seed Search overfit |
| **Wave 2 (`Option B`)** | Multi-Domain Scale (`B1..B6`) | `6` | `3` (`B1`, `B3`, `B5`) | `3` (`B2`, `B4`, `B6`) | `B1` `85%` `seq_len 96->336` confound, `B2` `15.79x` 20-restart inflation + loss to equal-compute `kmeans++`, `B3` active-inference under-coverage (`0.8625 < 0.885`) |
| **Wave 3 (`Option C`)** | Extended Pool (`C1..C6`) | `6` | `2` (`C4`, `C6`) | `4` (`C1`, `C2`, `C3`, `C5`) | `C1` `num_folds 5->10` label mutation, `C2` `19.30x` (`42x`-pass) IHDP ensemble, `C3` `+7.60 pp` `max`-over-1500-test-epochs peeking, `C4` DGP `variance=0.49` tamper (`-50.9%` floor), `C5` `3.66x` `n_pairs=8`, `C6` `eps_0=0.001` slow-graph under-coverage |
| **Wave 4 (`Option D`)** | **4-Agent Iterative RSI (`D1..D9`)** | **`9`** | **`5` (`D1, D4, D5, D7, D8` — `55.6%`)** | **`4` (`D2, D3, D6, D9`)** | `D1` `N_ENSEMBLE=100` (`20,000`-query leak on $2^{10}$ game), `D2` `GroupDRO` gate, `D3` `Monotone-Spline-Ridge` gate, `D6` `4x` MCMC steps, `D8` `28.7x` 8-seed BBVI, `D9` `n_iter_test=10` `11x` infeasibility |
| **Total** | **Waves 0–4 (`+ 105/105` Pool Audit)** | **`29`** | **`12` (`41.4%`)** | **`17` (`0` Holdout looks)** | **`100%` (`12 / 12`) of Holdout seals passed every Holdout accuracy & safety guardrail (`0` post-Holdout retractions)** |

*Note: Zero third-party `AutoSOTA` source files, third-party datasets, or cloud credentials/secrets are stored in this repository.*
