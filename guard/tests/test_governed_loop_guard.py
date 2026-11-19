#!/usr/bin/env python3
"""
Adversarial, Fault-Injection & Statistical Regression Suite for governed_loop_guard.py (v3.0).

Tests every defect surfaced in the 3-Track AutoSOTA Audit and Frontier Cowork Reviews
(GPT-6 Astra Max & Anthropic Opus 5.5 Max):
  1. Identical (A/A) candidate never promotes under any noise setting.
  2. Confirm stage fails closed when items are omitted, SE == 0, or replication < 2.
  3. Rollback-streak state machine neither halts a PROVISIONAL_KEEP winner prematurely
     nor lets unconfirmed PROVISIONAL_KEEP launder prior rollbacks when Confirm fails.
  4. Cluster/seed/stratum metadata tampering (cluster-splitting attack) fails closed.
  5. Frozen equal-stratum (`equal_stratum`) weighting prevents horizon window-count distortion.
  6. 2-way crossed `(seed x cluster)` bootstrap captures seed variance and computes exact
     nonlinear `macro_f1_from_counts` from pooled `(TP, FP, FN)`.
  7. Predeclared `max_confirm_looks` enforces `alpha / K` multiplicity & look cap, and
     `--proposer-out` redacts raw Dev/Confirm scores.
  8. Atomic `gate-holdout` requires `PROMOTE` across EVERY mandatory comparator, binds
     artifact SHA-256, and seals Holdout strictly once across all candidates.
  9. `audit-leaks` & `run-isolated` block planted label readers, `validate()` TTA hooks,
     `sys.modules` patches, protocol overrides, and illegal online actions while passing
     clean `predict` and `causal_online` workers.
 10. `freeze-manifest` / `verify-manifest` / `verify-ledger` catch 1-byte edits, added files,
     symlink escapes, and ledger tampering.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

GUARD_PATH = Path(__file__).resolve().parents[1] / "scripts" / "governed_loop_guard.py"


def run_guard(*args: str, check: bool = False) -> tuple[int, dict]:
    proc = subprocess.run(
        [sys.executable, str(GUARD_PATH), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    out_txt = (proc.stdout or "").strip() or (proc.stderr or "").strip()
    payload = json.loads(out_txt) if out_txt else {}
    if check and proc.returncode != 0:
        raise AssertionError(f"Guard failed ({proc.returncode}): {out_txt}\nSTDERR: {proc.stderr}")
    return proc.returncode, payload


def write_json(path: Path, data: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


class TestGovernedLoopGuardV3(unittest.TestCase):
    def test_01_aa_identical_candidate_never_promotes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            items = [{"id": f"item_{i}", "cluster_id": f"c_{i%8}", "value": 0.5 + 0.01 * i} for i in range(16)]
            arm = write_json(p / "arm.json", {"candidate_id": "aa_cand", "metrics": {"score": 0.575}, "items": items})
            rules_confirm = write_json(
                p / "rules_confirm.json",
                {"primary_metric": "score", "direction": "maximize", "decision_stage": "confirm", "alpha_one_sided": 0.025},
            )
            rc, res = run_guard(
                "gate-decision", "--candidate", str(arm), "--incumbent", str(arm), "--baseline", str(arm), "--rules", str(rules_confirm)
            )
            self.assertEqual(rc, 1)
            self.assertEqual(res["decision"], "ROLLBACK")

            rules_search = write_json(
                p / "rules_search.json",
                {"primary_metric": "score", "direction": "maximize", "decision_stage": "search"},
            )
            rc_s, res_s = run_guard(
                "gate-decision", "--candidate", str(arm), "--incumbent", str(arm), "--baseline", str(arm), "--rules", str(rules_search)
            )
            self.assertEqual(rc_s, 1)
            self.assertEqual(res_s["decision"], "ROLLBACK")

    def test_02_confirm_fails_closed_when_items_missing_or_se_zero(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            # 2a: Missing items even when aa_noise_se > 0 must NOT promote on confirm
            cand = write_json(p / "cand.json", {"candidate_id": "c1", "metrics": {"mse": 0.350}})
            inc = write_json(p / "inc.json", {"candidate_id": "b0", "metrics": {"mse": 0.400}})
            rules = write_json(
                p / "rules.json",
                {
                    "primary_metric": "mse",
                    "direction": "minimize",
                    "decision_stage": "confirm",
                    "aa_noise_se": 0.005,
                },
            )
            rc, res = run_guard("gate-decision", "--candidate", str(cand), "--incumbent", str(inc), "--baseline", str(inc), "--rules", str(rules))
            self.assertEqual(rc, 1)
            self.assertEqual(res["decision"], "ROLLBACK")
            self.assertIn("missing_paired_items_for_confirm", res["tripwires"])

            # 2b: Single cluster (n_clusters = 1) must fail closed
            cand_1c = write_json(
                p / "cand_1c.json",
                {"candidate_id": "c1c", "metrics": {"mse": 0.35}, "items": [{"id": "1", "cluster_id": "only", "value": 0.35}]},
            )
            inc_1c = write_json(
                p / "inc_1c.json",
                {"candidate_id": "b1c", "metrics": {"mse": 0.40}, "items": [{"id": "1", "cluster_id": "only", "value": 0.40}]},
            )
            rc2, res2 = run_guard("gate-decision", "--candidate", str(cand_1c), "--incumbent", str(inc_1c), "--baseline", str(inc_1c), "--rules", str(rules))
            self.assertEqual(rc2, 1)
            self.assertEqual(res2["decision"], "ROLLBACK")
            self.assertTrue(any("insufficient_clusters_for_confirm" in t for t in res2["tripwires"]))

            # 2c: Constant shift across only 3 clusters (SE_boot == 0, sign test 0.5^3 = 0.125 > 0.025) must NOT promote
            cand_3c = write_json(
                p / "cand_3c.json",
                {
                    "candidate_id": "c3c",
                    "metrics": {"mse": 0.35},
                    "items": [{"id": str(i), "cluster_id": f"cl_{i}", "value": 0.35} for i in range(3)],
                },
            )
            inc_3c = write_json(
                p / "inc_3c.json",
                {
                    "candidate_id": "b3c",
                    "metrics": {"mse": 0.40},
                    "items": [{"id": str(i), "cluster_id": f"cl_{i}", "value": 0.40} for i in range(3)],
                },
            )
            rc3, res3 = run_guard("gate-decision", "--candidate", str(cand_3c), "--incumbent", str(inc_3c), "--baseline", str(inc_3c), "--rules", str(rules))
            self.assertEqual(rc3, 1)
            self.assertEqual(res3["decision"], "ROLLBACK")
            self.assertIn("degenerate_zero_variance", res3["decision_reason"])

            # 2d: Constant shift across 8 independent clusters (0.5^8 = 0.0039 <= 0.025) legitimately promotes
            cand_8c = write_json(
                p / "cand_8c.json",
                {
                    "candidate_id": "c8c",
                    "metrics": {"mse": 0.35},
                    "items": [{"id": str(i), "cluster_id": f"cl_{i}", "value": 0.35} for i in range(8)],
                },
            )
            inc_8c = write_json(
                p / "inc_8c.json",
                {
                    "candidate_id": "b8c",
                    "metrics": {"mse": 0.40},
                    "items": [{"id": str(i), "cluster_id": f"cl_{i}", "value": 0.40} for i in range(8)],
                },
            )
            rc4, res4 = run_guard("gate-decision", "--candidate", str(cand_8c), "--incumbent", str(inc_8c), "--baseline", str(inc_8c), "--rules", str(rules))
            self.assertEqual(rc4, 0)
            self.assertEqual(res4["decision"], "PROMOTE")

    def test_03_rollback_streak_state_machine_replay(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            ledger = p / "ledger.jsonl"
            items_base = [{"id": str(i), "cluster_id": f"c_{i}", "value": 0.50 + 0.002 * i} for i in range(10)]
            items_bad = [{"id": str(i), "cluster_id": f"c_{i}", "value": 0.55 + 0.002 * i} for i in range(10)]
            items_good = [{"id": str(i), "cluster_id": f"c_{i}", "value": 0.42 + 0.001 * i} for i in range(10)]

            base_f = write_json(p / "base.json", {"candidate_id": "base", "metrics": {"mse": 0.509}, "items": items_base})
            rules_search = write_json(
                p / "rules_search.json",
                {"primary_metric": "mse", "direction": "minimize", "decision_stage": "search", "max_consecutive_rollbacks": 4},
            )
            rules_confirm = write_json(
                p / "rules_confirm.json",
                {"primary_metric": "mse", "direction": "minimize", "decision_stage": "confirm", "max_consecutive_rollbacks": 4},
            )

            # 3 consecutive rollbacks first
            for t in range(1, 4):
                bad_f = write_json(p / f"bad_{t}.json", {"candidate_id": f"bad_{t}", "metrics": {"mse": 0.559}, "items": items_bad})
                rc, res = run_guard(
                    "gate-decision", "--candidate", str(bad_f), "--incumbent", str(base_f),
                    "--baseline", str(base_f), "--rules", str(rules_search), "--ledger", str(ledger), "--trial-id", f"t{t}"
                )
                self.assertEqual(rc, 1)
                self.assertEqual(res["consecutive_rollbacks"], t)
                self.assertFalse(res["halt_recommended"])

            # Trial 4 wins PROVISIONAL_KEEP on search -> MUST NOT halt!
            good_search = write_json(
                p / "good_s.json", {"candidate_id": "cand_win", "protocol_id": "search", "metrics": {"mse": 0.4245}, "items": items_good}
            )
            rc_pk, res_pk = run_guard(
                "gate-decision", "--candidate", str(good_search), "--incumbent", str(base_f),
                "--baseline", str(base_f), "--rules", str(rules_search), "--ledger", str(ledger), "--trial-id", "t4"
            )
            self.assertEqual(rc_pk, 0)
            self.assertEqual(res_pk["decision"], "PROVISIONAL_KEEP")
            self.assertEqual(res_pk["consecutive_rollbacks"], 0)
            self.assertEqual(res_pk["unresolved_rollbacks_since_promote"], 3)
            self.assertFalse(res_pk["halt_recommended"])

            # Suppose cand_win FAILS confirm -> streak must become 3 + 1 = 4 -> halt_recommended == True!
            bad_confirm = write_json(
                p / "bad_c.json", {"candidate_id": "cand_win", "protocol_id": "confirm", "metrics": {"mse": 0.559}, "items": items_bad}
            )
            rc_cf, res_cf = run_guard(
                "gate-decision", "--candidate", str(bad_confirm), "--incumbent", str(base_f),
                "--baseline", str(base_f), "--rules", str(rules_confirm), "--ledger", str(ledger), "--trial-id", "t4_confirm"
            )
            self.assertEqual(rc_cf, 1)
            self.assertEqual(res_cf["decision"], "ROLLBACK")
            self.assertEqual(res_cf["consecutive_rollbacks"], 4)
            self.assertTrue(res_cf["halt_recommended"])

            # Now test a true PROMOTE resets everything to 0
            good_confirm = write_json(
                p / "good_c.json", {"candidate_id": "cand_real_win", "protocol_id": "confirm", "metrics": {"mse": 0.4245}, "items": items_good}
            )
            rc_pr, res_pr = run_guard(
                "gate-decision", "--candidate", str(good_confirm), "--incumbent", str(base_f),
                "--baseline", str(base_f), "--rules", str(rules_confirm), "--ledger", str(ledger), "--trial-id", "t5_confirm"
            )
            self.assertEqual(rc_pr, 0)
            self.assertEqual(res_pr["decision"], "PROMOTE")
            self.assertEqual(res_pr["consecutive_rollbacks"], 0)
            self.assertEqual(res_pr["unresolved_rollbacks_since_promote"], 0)
            self.assertFalse(res_pr["halt_recommended"])

    def test_04_metadata_cluster_splitting_attack_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            inc_items = [{"id": f"q_{i}", "cluster_id": f"seed_{i%2}", "value": 0.50} for i in range(20)]
            # Attacker splits 2 clusters into 20 clusters to manufacture tiny SE
            cand_items = [{"id": f"q_{i}", "cluster_id": f"fake_cluster_{i}", "value": 0.52} for i in range(20)]
            inc = write_json(p / "inc.json", {"candidate_id": "inc", "metrics": {"acc": 0.50}, "items": inc_items})
            cand = write_json(p / "cand.json", {"candidate_id": "atk", "metrics": {"acc": 0.52}, "items": cand_items})
            rules = write_json(p / "rules.json", {"primary_metric": "acc", "direction": "maximize", "decision_stage": "confirm"})
            rc, res = run_guard("gate-decision", "--candidate", str(cand), "--incumbent", str(inc), "--baseline", str(inc), "--rules", str(rules))
            self.assertEqual(rc, 1)
            self.assertEqual(res["decision"], "ROLLBACK")
            self.assertTrue(any("metadata tamper/mismatch" in t for t in res["tripwires"]))

    def test_05_equal_stratum_weighting_blocks_horizon_window_distortion(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            # H96 has 40 items (cand improves MSE by -0.01); H720 has 4 items (cand worsens MSE by +0.05)
            # Equal-item mean: (-40*0.01 + 4*0.05)/44 = -0.20/44 = -0.00454 (falsely looks better!)
            # Equal-stratum mean: (-0.01 + 0.05)/2 = +0.020 (correctly sees +0.02 MSE degradation!)
            inc_items = []
            cand_items = []
            for i in range(40):
                inc_items.append({"id": f"h96_{i}", "cluster_id": f"b_{i%8}", "stratum_id": "H96", "value": 0.30})
                cand_items.append({"id": f"h96_{i}", "cluster_id": f"b_{i%8}", "stratum_id": "H96", "value": 0.29 + (0.001 if i % 2 else -0.001)})
            for i in range(8):
                inc_items.append({"id": f"h720_{i}", "cluster_id": f"b_{i%8}", "stratum_id": "H720", "value": 0.40})
                cand_items.append({"id": f"h720_{i}", "cluster_id": f"b_{i%8}", "stratum_id": "H720", "value": 0.45 + (0.002 if i % 2 else -0.002)})

            inc = write_json(p / "inc.json", {"candidate_id": "inc", "metrics": {"mse": 0.35}, "items": inc_items})
            cand = write_json(p / "cand.json", {"candidate_id": "cand", "metrics": {"mse": 0.37}, "items": cand_items})
            rules = write_json(
                p / "rules.json",
                {
                    "primary_metric": "mse",
                    "direction": "minimize",
                    "aggregation_weighting": "equal_stratum",
                    "expected_strata": ["H96", "H720"],
                    "decision_stage": "search",
                },
            )
            rc, res = run_guard("gate-decision", "--candidate", str(cand), "--incumbent", str(inc), "--baseline", str(inc), "--rules", str(rules))
            self.assertEqual(rc, 1)
            self.assertEqual(res["decision"], "ROLLBACK")
            self.assertAlmostEqual(res["signed_improvement"], -0.02, places=4)

    def test_06_crossed_seed_cluster_and_macro_f1_from_counts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            # 5 seeds x 12 blocks; blocks are nearly uniform, but 1 seed is noisy while 4 improve strongly
            inc_items = []
            cand_items = []
            seed_shifts = [12, 11, 13, 10, 8]
            for s_idx, shift in enumerate(seed_shifts):
                for b_idx in range(12):
                    iid = f"s{s_idx}_b{b_idx}"
                    inc_items.append(
                        {"id": iid, "seed_id": f"s{s_idx}", "cluster_id": f"b{b_idx}", "stratum_id": "taskA", "class_id": "1", "tp": 20, "fp": 10, "fn": 10}
                    )
                    cand_items.append(
                        {
                            "id": iid,
                            "seed_id": f"s{s_idx}",
                            "cluster_id": f"b{b_idx}",
                            "stratum_id": "taskA",
                            "class_id": "1",
                            "tp": 20 + shift + (b_idx % 3),
                            "fp": 6,
                            "fn": 6,
                        }
                    )

            inc = write_json(p / "inc.json", {"candidate_id": "inc", "metrics": {"ood_f1": 0.6667}, "items": inc_items})
            cand = write_json(p / "cand.json", {"candidate_id": "cand", "metrics": {"ood_f1": 0.84}, "items": cand_items})
            rules = write_json(
                p / "rules.json",
                {
                    "primary_metric": "ood_f1",
                    "primary_statistic": "macro_f1_from_counts",
                    "aggregation_weighting": "equal_stratum",
                    "bootstrap_mode": "auto",
                    "direction": "maximize",
                    "decision_stage": "confirm",
                    "min_improvement": 0.05,
                },
            )
            rc, res = run_guard("gate-decision", "--candidate", str(cand), "--incumbent", str(inc), "--baseline", str(inc), "--rules", str(rules))
            self.assertEqual(rc, 0)
            self.assertEqual(res["decision"], "PROMOTE")
            self.assertEqual(res["bootstrap_mode"], "crossed_seed_cluster")
            self.assertGreater(res["effective_se"], 0.001)
            self.assertGreater(res["lcb_one_sided_vs_incumbent"], 0.10)

    def test_07_adaptive_confirm_multiplicity_and_proposer_redaction(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            ledger = p / "ledger.jsonl"
            proposer_out = p / "proposer_view.json"
            items_b = [{"id": str(i), "cluster_id": f"c_{i}", "value": 0.50 + 0.001 * i} for i in range(10)]
            items_c = [{"id": str(i), "cluster_id": f"c_{i}", "value": 0.505 + 0.001 * i} for i in range(10)]
            base = write_json(p / "base.json", {"candidate_id": "base", "metrics": {"score": 0.5045}, "items": items_b})
            rules = write_json(
                p / "rules.json",
                {
                    "primary_metric": "score",
                    "direction": "maximize",
                    "decision_stage": "confirm",
                    "alpha_one_sided": 0.025,
                    "max_confirm_looks": 2,
                    "min_improvement": 0.02,
                },
            )
            for idx in range(1, 4):
                cf = write_json(p / f"c{idx}.json", {"candidate_id": f"cand_{idx}", "metrics": {"score": 0.5095}, "items": items_c})
                rc, res = run_guard(
                    "gate-decision",
                    "--candidate", str(cf),
                    "--incumbent", str(base),
                    "--baseline", str(base),
                    "--rules", str(rules),
                    "--ledger", str(ledger),
                    "--proposer-out", str(proposer_out),
                    "--trial-id", f"c_{idx}",
                )
                self.assertEqual(rc, 1)
                self.assertAlmostEqual(res["effective_alpha_one_sided"], 0.0125, places=6)
                if idx == 3:
                    self.assertTrue(any("confirm_look_budget_exhausted" in t for t in res["tripwires"]))

            redacted = json.loads(proposer_out.read_text(encoding="utf-8"))
            self.assertNotIn("candidate_primary", redacted)
            self.assertNotIn("lcb_one_sided_vs_incumbent", redacted)
            self.assertEqual(redacted["decision"], "ROLLBACK")

    def test_08_atomic_gate_holdout_seal_and_mandatory_comparators(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            imm = p / "immutable"
            mut = p / "mutable"
            imm.mkdir()
            mut.mkdir()

            cand_code = mut / "candidate.py"
            cand_code.write_text("WEIGHT = 0.75\n", encoding="utf-8")

            rules = write_json(
                imm / "rules.json",
                {
                    "primary_metric": "mse",
                    "direction": "minimize",
                    "decision_stage": "confirm",
                    "required_comparators": ["xpatch", "ridge"],
                    "min_improvement": 0.004,
                },
            )
            manifest = p / "IMMUTABLE_MANIFEST.json"
            _, f_res = run_guard("freeze-manifest", "--manifest", str(manifest), "--root", str(p), "--paths", "immutable", check=True)
            pinned_sha = f_res["manifest_root_sha256"]

            ledger = p / "ledger.jsonl"
            seal = p / "HOLDOUT_SEAL.json"

            items_xpatch = [{"id": str(i), "cluster_id": f"c_{i}", "value": 0.450 + 0.002 * i} for i in range(12)]
            items_ridge = [{"id": str(i), "cluster_id": f"c_{i}", "value": 0.412 + (0.012 if i % 2 else -0.012)} for i in range(12)]
            items_cand = [{"id": str(i), "cluster_id": f"c_{i}", "value": 0.410 + 0.001 * i} for i in range(12)]

            xp_f = write_json(p / "xp.json", {"candidate_id": "xpatch", "metrics": {"mse": 0.461}, "items": items_xpatch})
            rg_f = write_json(p / "rg.json", {"candidate_id": "ridge", "metrics": {"mse": 0.412}, "items": items_ridge})
            cd_f = write_json(p / "cd.json", {"candidate_id": "t1_trial_08", "metrics": {"mse": 0.4155}, "items": items_cand})

            # Passes vs xpatch (PROMOTE)
            rc1, r1 = run_guard(
                "gate-decision", "--candidate", str(cd_f), "--incumbent", str(xp_f), "--baseline", str(xp_f),
                "--rules", str(rules), "--ledger", str(ledger), "--comparator-id", "xpatch",
                "--candidate-artifacts", str(cand_code), "--trial-id", "confirm_xp",
            )
            self.assertEqual(rc1, 0)
            self.assertEqual(r1["decision"], "PROMOTE")

            # Fails vs ridge (ROLLBACK)
            rc2, r2 = run_guard(
                "gate-decision", "--candidate", str(cd_f), "--incumbent", str(rg_f), "--baseline", str(xp_f),
                "--rules", str(rules), "--ledger", str(ledger), "--comparator-id", "ridge",
                "--candidate-artifacts", str(cand_code), "--trial-id", "confirm_rg",
            )
            self.assertEqual(rc2, 1)

            # Attempting Holdout MUST BE BLOCKED because mandatory comparator `ridge` is not PROMOTE!
            rc_h1, res_h1 = run_guard(
                "gate-holdout", "--seal-file", str(seal), "--manifest", str(manifest),
                "--expected-root-sha256", pinned_sha, "--ledger", str(ledger), "--rules", str(rules),
                "--candidate-id", "t1_trial_08", "--candidate-artifacts", str(cand_code),
                "--candidate-holdout", str(cd_f), "--comparator-holdout", f"xpatch={xp_f}", "--comparator-holdout", f"ridge={rg_f}",
            )
            self.assertEqual(rc_h1, 1)
            self.assertEqual(res_h1["status"], "BLOCKED_NOT_CONFIRMED")
            self.assertFalse(seal.exists())

            # Now test a candidate `t1_trial_09` that legitimately PROMOTEs against BOTH xpatch and ridge
            items_cand9 = [{"id": str(i), "cluster_id": f"c_{i}", "value": 0.380 + 0.001 * i} for i in range(12)]
            cd9_f = write_json(p / "cd9.json", {"candidate_id": "t1_trial_09", "metrics": {"mse": 0.3855}, "items": items_cand9})
            for comp_name, comp_file in (("xpatch", xp_f), ("ridge", rg_f)):
                run_guard(
                    "gate-decision", "--candidate", str(cd9_f), "--incumbent", str(comp_file), "--baseline", str(xp_f),
                    "--rules", str(rules), "--ledger", str(ledger), "--comparator-id", comp_name,
                    "--candidate-artifacts", str(cand_code), "--trial-id", f"c9_{comp_name}", check=True,
                )

            # Tamper candidate code by 1 byte -> Holdout MUST block on artifact SHA mismatch!
            cand_code.write_text("WEIGHT = 0.76\n", encoding="utf-8")
            rc_tamper, res_tamper = run_guard(
                "gate-holdout", "--seal-file", str(seal), "--manifest", str(manifest),
                "--expected-root-sha256", pinned_sha, "--ledger", str(ledger), "--rules", str(rules),
                "--candidate-id", "t1_trial_09", "--candidate-artifacts", str(cand_code),
                "--candidate-holdout", str(cd9_f), "--comparator-holdout", f"xpatch={xp_f}", "--comparator-holdout", f"ridge={rg_f}",
            )
            self.assertEqual(rc_tamper, 1)
            self.assertEqual(res_tamper["status"], "BLOCKED_NOT_CONFIRMED")
            self.assertTrue(len(res_tamper["artifact_sha_mismatches"]) > 0)

            # Restore exact promoted code -> Holdout succeeds ONCE and seals forever
            cand_code.write_text("WEIGHT = 0.75\n", encoding="utf-8")
            rc_ok, res_ok = run_guard(
                "gate-holdout", "--seal-file", str(seal), "--manifest", str(manifest),
                "--expected-root-sha256", pinned_sha, "--ledger", str(ledger), "--rules", str(rules),
                "--candidate-id", "t1_trial_09", "--candidate-artifacts", str(cand_code),
                "--candidate-holdout", str(cd9_f), "--comparator-holdout", f"xpatch={xp_f}", "--comparator-holdout", f"ridge={rg_f}",
            )
            self.assertEqual(rc_ok, 0)
            self.assertEqual(res_ok["status"], "FINALIZED")
            self.assertTrue(seal.exists())

            # Second Holdout attempt (even for same or another candidate) MUST return exit 2 HOLDOUT_ALREADY_CONSUMED
            rc_dup, res_dup = run_guard(
                "gate-holdout", "--seal-file", str(seal), "--manifest", str(manifest),
                "--expected-root-sha256", pinned_sha, "--ledger", str(ledger), "--rules", str(rules),
                "--candidate-id", "t1_trial_09", "--candidate-artifacts", str(cand_code),
                "--candidate-holdout", str(cd9_f), "--comparator-holdout", f"xpatch={xp_f}", "--comparator-holdout", f"ridge={rg_f}",
            )
            self.assertEqual(rc_dup, 2)
            self.assertEqual(res_dup["status"], "HOLDOUT_ALREADY_CONSUMED")

            # Verify ledger chain remains valid after HOLDOUT_SEALED append
            rc_vl, res_vl = run_guard("verify-ledger", "--ledger", str(ledger))
            self.assertEqual(rc_vl, 0)
            self.assertEqual(res_vl["status"], "VERIFIED")

    def test_09_audit_leaks_and_run_isolated_blocks_planted_cheats(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            secret_dir = p / "protected_holdout_labels"
            secret_dir.mkdir()
            (secret_dir / "gold_labels.json").write_text('{"q1": 1.0}', encoding="utf-8")

            # 9a: Planted leak candidate (sys.modules patch + seq_len override + holdout path literal)
            cheater = p / "cheater.py"
            cheater.write_text(
                "import sys\n"
                "seq_len = 336\n"
                "sys.modules['evaluator'] = None\n"
                "GOLD = 'protected_holdout_labels/gold_labels.json'\n",
                encoding="utf-8",
            )
            rc_lk, res_lk = run_guard("audit-leaks", "--paths", str(cheater), "--locked-constant", "seq_len=96")
            self.assertEqual(rc_lk, 1)
            self.assertEqual(res_lk["status"], "LEAK_DETECTED")
            codes = {f["code"] for f in res_lk["findings"]}
            self.assertIn("SYS_MODULES_MONKEYPATCH", codes)
            self.assertIn("LOCKED_PROTOCOL_OVERRIDE", codes)
            self.assertIn("HOLD_OR_TEST_PATH_LITERAL", codes)

            # 9b: Clean `predict` worker succeeds in scrubbed isolated subprocess
            clean_pred = p / "clean_pred.py"
            clean_pred.write_text(
                "import argparse, json\n"
                "ap = argparse.ArgumentParser()\n"
                "ap.add_argument('--input-json', required=True)\n"
                "ap.add_argument('--output-json', required=True)\n"
                "a = ap.parse_args()\n"
                "inp = json.loads(open(a.input_json, encoding='utf-8').read())\n"
                "preds = {k: v * 2.0 for k, v in inp['features'].items()}\n"
                "open(a.output_json, 'w', encoding='utf-8').write(json.dumps({'predictions': preds}))\n",
                encoding="utf-8",
            )
            inp_f = write_json(p / "inp.json", {"features": {"x1": 1.5, "x2": 3.0}})
            exp_ids = write_json(p / "exp_ids.json", ["x1", "x2"])
            out_f = p / "out_pred.json"
            rc_iso, res_iso = run_guard(
                "run-isolated", "--mode", "predict", "--candidate-script", str(clean_pred),
                "--input-json", str(inp_f), "--output-json", str(out_f),
                "--expected-ids-json", str(exp_ids), "--forbid-paths", str(secret_dir),
            )
            self.assertEqual(rc_iso, 0)
            self.assertEqual(res_iso["status"], "OK")
            self.assertEqual(json.loads(out_f.read_text(encoding="utf-8"))["predictions"], {"x1": 3.0, "x2": 6.0})

            # 9c: Causal online bin-packing worker (Best-Fit) vs illegal action worker
            bf_worker = p / "best_fit_online.py"
            bf_worker.write_text(
                "import sys, json\n"
                "for line in sys.stdin:\n"
                "    req = json.loads(line)\n"
                "    item = req['item']\n"
                "    bins = req['open_bins']\n"
                "    best_i = len(bins)\n"
                "    best_rem = 1e18\n"
                "    for i, rem in enumerate(bins):\n"
                "        if rem + 1e-9 >= item and (rem - item) < best_rem:\n"
                "            best_rem = rem - item\n"
                "            best_i = i\n"
                "    sys.stdout.write(json.dumps({'bin_index': best_i}) + '\\n')\n"
                "    sys.stdout.flush()\n",
                encoding="utf-8",
            )
            streams_f = write_json(
                p / "streams.json",
                {
                    "streams": [
                        {"id": "s1", "cluster_id": "fam1", "capacity": 100.0, "items": [60.0, 40.0, 55.0, 45.0]},
                        {"id": "s2", "cluster_id": "fam2", "capacity": 100.0, "items": [70.0, 70.0, 30.0, 30.0]},
                    ]
                },
            )
            out_online = p / "out_online.json"
            rc_on, res_on = run_guard(
                "run-isolated", "--mode", "causal_online", "--candidate-script", str(bf_worker),
                "--input-json", str(streams_f), "--output-json", str(out_online),
            )
            self.assertEqual(rc_on, 0)
            self.assertAlmostEqual(res_on["excess_bin_ratio"], 0.0, places=6)

            bad_online = p / "bad_online.py"
            bad_online.write_text(
                "import sys, json\n"
                "for line in sys.stdin:\n"
                "    sys.stdout.write(json.dumps({'bin_index': 999}) + '\\n')\n"
                "    sys.stdout.flush()\n",
                encoding="utf-8",
            )
            rc_bad, res_bad = run_guard(
                "run-isolated", "--mode", "causal_online", "--candidate-script", str(bad_online),
                "--input-json", str(streams_f), "--output-json", str(p / "bad_out.json"),
            )
            self.assertEqual(rc_bad, 1)
            self.assertEqual(res_bad["status"], "ILLEGAL_CAUSAL_ACTION")

    def test_10_manifest_tamper_and_ledger_chain_integrity(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            eval_dir = p / "eval"
            eval_dir.mkdir()
            f1 = eval_dir / "scorer.py"
            f1.write_text("SCORE = 1\n", encoding="utf-8")
            manifest = p / "IMMUTABLE_MANIFEST.json"
            _, fr = run_guard("freeze-manifest", "--manifest", str(manifest), "--root", str(p), "--paths", "eval", check=True)
            pinned = fr["manifest_root_sha256"]

            rc_v0, _ = run_guard("verify-manifest", "--manifest", str(manifest), "--expected-root-sha256", pinned)
            self.assertEqual(rc_v0, 0)

            # Inject shadow file inside protected root -> TAMPERED
            (eval_dir / "conftest.py").write_text("import os\n", encoding="utf-8")
            rc_v1, res_v1 = run_guard("verify-manifest", "--manifest", str(manifest), "--expected-root-sha256", pinned)
            self.assertEqual(rc_v1, 1)
            self.assertEqual(res_v1["status"], "TAMPERED")
            self.assertIn("eval/conftest.py", res_v1["added_inside_protected_roots"])

    def test_11_per_stratum_conformal_and_privacy_guardrails(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            # 5 strata have 0.95 coverage, 1 stratum collapses to 0.75 (global mean 0.917 >= 0.90!)
            inc_items = []
            cand_items = []
            for st_idx in range(6):
                cov = 0.95 if st_idx < 5 else 0.75
                for j in range(10):
                    iid = f"d{st_idx}_{j}"
                    inc_items.append(
                        {"id": iid, "cluster_id": f"b_{j}", "stratum_id": f"ds_{st_idx}", "value": 2.80, "aux": {"coverage": 0.91, "eps_spent": 3.0}}
                    )
                    cand_items.append(
                        {"id": iid, "cluster_id": f"b_{j}", "stratum_id": f"ds_{st_idx}", "value": 2.40, "aux": {"coverage": cov, "eps_spent": 3.0}}
                    )

            inc = write_json(p / "inc.json", {"candidate_id": "inc", "metrics": {"band_len": 2.80}, "items": inc_items})
            cand = write_json(p / "cand.json", {"candidate_id": "cand", "metrics": {"band_len": 2.40}, "items": cand_items})
            rules = write_json(
                p / "rules.json",
                {
                    "primary_metric": "band_len",
                    "direction": "minimize",
                    "aggregation_weighting": "equal_stratum",
                    "decision_stage": "confirm",
                    "per_stratum_hard_floors": {"coverage": 0.88},
                    "per_stratum_hard_ceilings": {"eps_spent": 3.0},
                },
            )
            rc, res = run_guard("gate-decision", "--candidate", str(cand), "--incumbent", str(inc), "--baseline", str(inc), "--rules", str(rules))
            self.assertEqual(rc, 1)
            self.assertEqual(res["decision"], "ROLLBACK")
            self.assertTrue(any("per_stratum_floor_failed:ds_5.coverage" in t for t in res["tripwires"]))

    def test_12_unified_diff_and_dp_clean_weight_leak_detection(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            patch = p / "paper96_sim.diff"
            patch.write_text(
                "--- a/main-sklearn.py\n"
                "+++ b/main-sklearn.py\n"
                "@@ -55,3 +55,5 @@\n"
                "+            self._base_coef = self.coef_ - noise\n"
                "+    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)\n",
                encoding="utf-8",
            )
            rc, res = run_guard("audit-leaks", "--paths", str(patch))
            self.assertEqual(rc, 1)
            self.assertEqual(res["status"], "LEAK_DETECTED")
            codes = {f["code"] for f in res["findings"]}
            self.assertIn("DP_NOISE_SUBTRACTION_OR_CLEAN_CACHE", codes)
            self.assertIn("DIFF_SPLIT_PROTOCOL_MUTATION", codes)

    def test_13_python_dp_clean_weight_cache_static_detection(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            f_dp = Path(td) / "dp_classifier.py"
            f_dp.write_text(
                "import numpy as np\n"
                "class BadDPSVM:\n"
                "    def fit(self, W, sigma):\n"
                "        noise = np.random.normal(0.0, sigma, W.shape)\n"
                "        self.coef_ = W + noise\n"
                "        self._base_coef = self.coef_ - noise\n",
                encoding="utf-8",
            )
            rc, res = run_guard("audit-leaks", "--paths", str(f_dp))
            self.assertEqual(rc, 1)
            self.assertEqual(res["status"], "LEAK_DETECTED")
            codes = {f["code"] for f in res["findings"]}
            self.assertIn("DP_NOISE_SUBTRACTION_OR_CLEAN_CACHE", codes)

    def test_14_stratum_bootstrap_multiplicity_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            # When cluster_id == stratum_id across 8 benchmark datasets and all 8 improve by +0.08..+0.15,
            # multiplicity-weighted stratum bootstrap must confirm cleanly with positive LCB.
            inc_items = []
            cand_items = []
            for d_idx in range(8):
                ds = f"ds_{d_idx}"
                base_v = 0.20 + 0.08 * d_idx
                gain_v = 0.09 + 0.01 * (d_idx % 4)
                for s_idx in range(4):
                    iid = f"{ds}_s{s_idx}"
                    inc_items.append({"id": iid, "seed_id": f"s{s_idx}", "cluster_id": ds, "stratum_id": ds, "value": base_v})
                    cand_items.append({"id": iid, "seed_id": f"s{s_idx}", "cluster_id": ds, "stratum_id": ds, "value": base_v + gain_v})
            inc = write_json(p / "inc.json", {"candidate_id": "inc", "metrics": {"acc": 0.48}, "items": inc_items})
            cand = write_json(p / "cand.json", {"candidate_id": "cand", "metrics": {"acc": 0.585}, "items": cand_items})
            rules = write_json(
                p / "rules.json",
                {
                    "primary_metric": "acc",
                    "aggregation_weighting": "equal_stratum",
                    "expected_strata": [f"ds_{i}" for i in range(8)],
                    "direction": "maximize",
                    "decision_stage": "confirm",
                    "min_improvement": 0.05,
                },
            )
            rc, res = run_guard("gate-decision", "--candidate", str(cand), "--incumbent", str(inc), "--baseline", str(inc), "--rules", str(rules))
            self.assertEqual(rc, 0)
            self.assertEqual(res["decision"], "PROMOTE")
            self.assertGreater(res["lcb_one_sided_vs_incumbent"], 0.08)

    def test_15_small_cluster_exact_signflip_blocks_underpowered_4_clusters(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            # With G=4 clusters, 2^-4 = 0.0625 > alpha=0.025: bootstrap percentile might look positive,
            # but exact Rademacher sign-flip p-value (0.0625) correctly blocks under-powered promotion!
            inc_items = [{"id": f"i{k}", "cluster_id": f"c{k}", "value": 0.40} for k in range(4)]
            cand_items = [{"id": f"i{k}", "cluster_id": f"c{k}", "value": 0.48 + 0.005 * k} for k in range(4)]
            inc = write_json(p / "inc.json", {"candidate_id": "inc", "metrics": {"acc": 0.40}, "items": inc_items})
            cand = write_json(p / "cand.json", {"candidate_id": "c4", "metrics": {"acc": 0.4875}, "items": cand_items})
            rules = write_json(
                p / "rules.json",
                {"primary_metric": "acc", "direction": "maximize", "decision_stage": "confirm", "alpha_one_sided": 0.025},
            )
            rc, res = run_guard("gate-decision", "--candidate", str(cand), "--incumbent", str(inc), "--baseline", str(inc), "--rules", str(rules))
            self.assertEqual(rc, 1)
            self.assertEqual(res["decision"], "ROLLBACK")
            self.assertIn("small_cluster_signflip_failed", res["decision_reason"])

    def test_16_confirm_rules_hash_pin_and_terminal_k_halt_and_redaction(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            ledger = p / "ledger.jsonl"
            prop_out = p / "prop.json"
            inc_items = [{"id": f"i{k}", "cluster_id": f"c{k}", "stratum_id": f"s{k%2}", "value": 0.50, "aux": {"cov": 0.92}} for k in range(8)]
            bad_items = [{"id": f"i{k}", "cluster_id": f"c{k}", "stratum_id": f"s{k%2}", "value": 0.55, "aux": {"cov": 0.70}} for k in range(8)]
            inc = write_json(p / "inc.json", {"candidate_id": "inc", "metrics": {"acc": 0.50}, "items": inc_items})
            bad = write_json(p / "bad.json", {"candidate_id": "c1", "metrics": {"acc": 0.55}, "items": bad_items})
            rules = write_json(
                p / "rules.json",
                {
                    "primary_metric": "acc",
                    "direction": "maximize",
                    "decision_stage": "confirm",
                    "max_confirm_looks": 1,
                    "per_stratum_hard_floors": {"cov": 0.88},
                },
            )
            rc1, _ = run_guard(
                "gate-decision", "--candidate", str(bad), "--incumbent", str(inc), "--baseline", str(inc),
                "--rules", str(rules), "--ledger", str(ledger), "--proposer-out", str(prop_out), "--trial-id", "t1",
            )
            self.assertEqual(rc1, 1)
            # Proposer view must NOT leak stratum name `s0.cov` on confirm!
            red = json.loads(prop_out.read_text(encoding="utf-8"))
            self.assertEqual(red["tripwires"], ["confirm_gate_or_guardrail_failed"])

            # Mutating confirm rules mid-campaign must fire confirm_rules_mutated_mid_campaign
            rules_mut = write_json(
                p / "rules_mut.json",
                {"primary_metric": "acc", "direction": "maximize", "decision_stage": "confirm", "max_confirm_looks": 5},
            )
            bad2 = write_json(p / "bad2.json", {"candidate_id": "c2", "metrics": {"acc": 0.55}, "items": bad_items})
            _, res2 = run_guard(
                "gate-decision", "--candidate", str(bad2), "--incumbent", str(inc), "--baseline", str(inc),
                "--rules", str(rules_mut), "--ledger", str(ledger), "--trial-id", "t2",
            )
            self.assertIn("confirm_rules_mutated_mid_campaign", res2["tripwires"])

            # Even a subsequent Search PROVISIONAL_KEEP must recommend terminal halt once K=1 confirm look is spent!
            rules_s = write_json(
                p / "rules_s.json",
                {"primary_metric": "acc", "direction": "maximize", "decision_stage": "search", "max_confirm_looks": 1},
            )
            good_s = write_json(p / "gs.json", {"candidate_id": "c3", "metrics": {"acc": 0.55}, "items": bad_items})
            _, res_s = run_guard(
                "gate-decision", "--candidate", str(good_s), "--incumbent", str(inc), "--baseline", str(inc),
                "--rules", str(rules_s), "--ledger", str(ledger), "--trial-id", "t3",
            )
            self.assertEqual(res_s["decision"], "PROVISIONAL_KEEP")
            self.assertTrue(res_s["halt_recommended"])

    def test_17_right_skewed_baseline_blowup_preserves_positive_bootstrap_lcb(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            # Replays Phase-C B5 (TreeHFD) empirical lesson: candidate (0.08) beats unstable baseline
            # on EVERY cluster ([0.55..0.65] on 7 clusters, [80.0] blowup on 1 cluster).
            # Nonparametric bootstrap LCB (> +0.45) and exact sign-flip (2^-8 = 0.0039 <= 0.0125) both pass,
            # whereas symmetric `mean - z * stdev` was negative due to right-skewed positive gains!
            ref_vals = [0.55, 0.58, 0.60, 0.62, 0.57, 0.61, 0.59, 80.0]
            inc_items = [{"id": f"i{k}", "cluster_id": f"c{k}", "value": ref_vals[k]} for k in range(8)]
            cand_items = [{"id": f"i{k}", "cluster_id": f"c{k}", "value": 0.08} for k in range(8)]
            inc = write_json(p / "inc.json", {"candidate_id": "inc", "metrics": {"mse": 10.515}, "items": inc_items})
            cand = write_json(p / "cand.json", {"candidate_id": "cand", "metrics": {"mse": 0.08}, "items": cand_items})
            rules = write_json(
                p / "rules.json",
                {
                    "primary_metric": "mse",
                    "direction": "minimize",
                    "decision_stage": "confirm",
                    "alpha_one_sided": 0.025,
                    "max_confirm_looks": 2,
                    "min_improvement": 0.01,
                },
            )
            rc, res = run_guard("gate-decision", "--candidate", str(cand), "--incumbent", str(inc), "--baseline", str(inc), "--rules", str(rules))
            self.assertEqual(rc, 0)
            self.assertEqual(res["decision"], "PROMOTE")
            self.assertGreater(res["lcb_one_sided_vs_incumbent"], 0.40)

    def test_18_exact_g16_signflip_margin_alignment_and_joint_external_noise(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            # 16 clusters with strictly positive gains above margin=0.01 -> exact 2^-16 = 1/65536 sign-flip p-value
            inc_items = [{"id": f"i{k}", "cluster_id": f"c{k:02d}", "value": 0.40} for k in range(16)]
            cand_items = [{"id": f"i{k}", "cluster_id": f"c{k:02d}", "value": 0.43 + 0.001 * k} for k in range(16)]
            inc = write_json(p / "inc.json", {"candidate_id": "inc", "metrics": {"acc": 0.40}, "items": inc_items})
            cand = write_json(p / "cand.json", {"candidate_id": "cand", "metrics": {"acc": 0.4375}, "items": cand_items})
            rules_ok = write_json(
                p / "rules_ok.json",
                {"primary_metric": "acc", "direction": "maximize", "decision_stage": "confirm", "min_improvement": 0.01},
            )
            rc1, res1 = run_guard("gate-decision", "--candidate", str(cand), "--incumbent", str(inc), "--baseline", str(inc), "--rules", str(rules_ok))
            self.assertEqual(rc1, 0)
            self.assertEqual(res1["decision"], "PROMOTE")
            self.assertEqual(res1["signflip_method"], "exact")
            self.assertAlmostEqual(res1["signflip_p_one_sided_vs_incumbent"], 1.0 / 65536.0, places=9)

            # Margin-aligned sign-flip: if min_improvement=0.05 > observed gains (~0.0375), signflip_p is 1.0 and blocks!
            rules_margin = write_json(
                p / "rules_m.json",
                {"primary_metric": "acc", "direction": "maximize", "decision_stage": "confirm", "min_improvement": 0.05},
            )
            rc2, res2 = run_guard("gate-decision", "--candidate", str(cand), "--incumbent", str(inc), "--baseline", str(inc), "--rules", str(rules_margin))
            self.assertEqual(rc2, 1)
            self.assertEqual(res2["decision"], "ROLLBACK")
            self.assertEqual(res2["signflip_p_one_sided_vs_incumbent"], 1.0)

            # Joint external noise (`aa_noise_se=0.05`) convolved inside bootstrap lowers LCB below 0.01!
            rules_ext = write_json(
                p / "rules_ext.json",
                {"primary_metric": "acc", "direction": "maximize", "decision_stage": "confirm", "min_improvement": 0.01, "aa_noise_se": 0.05},
            )
            rc3, res3 = run_guard("gate-decision", "--candidate", str(cand), "--incumbent", str(inc), "--baseline", str(inc), "--rules", str(rules_ext))
            self.assertLess(res3["lcb_one_sided_vs_incumbent"], res1["lcb_one_sided_vs_incumbent"] - 0.05)

    def test_19_holdout_finalize_verifies_per_stratum_guardrails(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            art = p / "cand.py"
            art.write_text("x = 1\n", encoding="utf-8")
            ev = p / "eval"
            ev.mkdir()
            rules = write_json(
                ev / "rules.json",
                {
                    "primary_metric": "width",
                    "direction": "minimize",
                    "decision_stage": "confirm",
                    "per_stratum_hard_floors": {"cov": 0.90},
                    "required_comparators": ["base"],
                },
            )
            mf = p / "MANIFEST.json"
            _, mf_res = run_guard("freeze-manifest", "--manifest", str(mf), "--root", str(p), "--paths", "eval")
            rsha = mf_res["manifest_root_sha256"]

            # Confirm passes cov=0.92 >= 0.90 across 8 clusters -> PROMOTE
            c_conf = write_json(p / "c_conf.json", {"candidate_id": "c1", "metrics": {"width": 0.8}, "items": [{"id": f"i{k}", "cluster_id": f"g{k}", "stratum_id": "s1", "value": 0.8, "aux": {"cov": 0.92}} for k in range(8)]})
            b_conf = write_json(p / "b_conf.json", {"candidate_id": "base", "metrics": {"width": 1.0}, "items": [{"id": f"i{k}", "cluster_id": f"g{k}", "stratum_id": "s1", "value": 1.0, "aux": {"cov": 0.92}} for k in range(8)]})
            ledger = p / "ledger.jsonl"
            rc_d, _ = run_guard("gate-decision", "--candidate", str(c_conf), "--incumbent", str(b_conf), "--baseline", str(b_conf), "--rules", str(rules), "--ledger", str(ledger), "--trial-id", "t1", "--comparator-id", "base", "--candidate-artifacts", str(art))
            self.assertEqual(rc_d, 0)

            seal = p / "HOLDOUT_SEAL.json"
            rc_r, _ = run_guard("gate-holdout", "--phase", "reserve", "--seal-file", str(seal), "--manifest", str(mf), "--expected-root-sha256", rsha, "--ledger", str(ledger), "--rules", str(rules), "--candidate-id", "c1", "--candidate-artifacts", str(art))
            self.assertEqual(rc_r, 0)

            # On Holdout, candidate under-covers (cov=0.84 < 0.90) -> holdout_guardrails_passed must be False!
            c_hold = write_json(p / "c_hold.json", {"candidate_id": "c1", "metrics": {"width": 0.78}, "items": [{"id": f"h{k}", "cluster_id": f"g{k}", "stratum_id": "unseen_shift", "value": 0.78, "aux": {"cov": 0.84}} for k in range(8)]})
            b_hold = write_json(p / "b_hold.json", {"candidate_id": "base", "metrics": {"width": 1.0}, "items": [{"id": f"h{k}", "cluster_id": f"g{k}", "stratum_id": "unseen_shift", "value": 1.0, "aux": {"cov": 0.92}} for k in range(8)]})
            rc_f, fin = run_guard("gate-holdout", "--phase", "finalize", "--seal-file", str(seal), "--manifest", str(mf), "--expected-root-sha256", rsha, "--ledger", str(ledger), "--rules", str(rules), "--candidate-id", "c1", "--candidate-artifacts", str(art), "--candidate-holdout", str(c_hold), "--comparator-holdout", f"base={b_hold}")
            self.assertEqual(rc_f, 0)
            self.assertFalse(fin["holdout_guardrails_passed"])
            self.assertTrue(any("per_stratum_floor_failed:unseen_shift.cov" in t for t in fin["holdout_tripwires"]))

    def test_20_g24_margin_enforced_on_lcb_and_preconditions_mde_and_signflip_floor(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            # 20a: G=24 clusters (>=20, so small-cluster signflip is inactive).
            # Candidate mean gain is +0.052 >= min_improvement=0.05, and LCB > 0 (+0.028),
            # BUT LCB (+0.028) <= 0.05! v3.7 must return INCONCLUSIVE_RETEST (rc=3), NOT PROMOTE.
            inc_items = [{"id": f"i{k}", "cluster_id": f"c{k:02d}", "value": 0.50} for k in range(24)]
            noisy_cand_items = [
                {"id": f"i{k}", "cluster_id": f"c{k:02d}", "value": 0.552 + (0.065 if k % 2 == 0 else -0.065)}
                for k in range(24)
            ]
            strong_cand_items = [
                {"id": f"i{k}", "cluster_id": f"c{k:02d}", "value": 0.60 + (0.015 if k % 2 == 0 else -0.015)}
                for k in range(24)
            ]
            inc = write_json(p / "inc.json", {"candidate_id": "inc", "metrics": {"acc": 0.50}, "items": inc_items})
            c_weak = write_json(p / "cw.json", {"candidate_id": "cw", "metrics": {"acc": 0.552}, "items": noisy_cand_items})
            c_strong = write_json(p / "cs.json", {"candidate_id": "cs", "metrics": {"acc": 0.60}, "items": strong_cand_items})
            rules = write_json(
                p / "rules.json",
                {"primary_metric": "acc", "direction": "maximize", "decision_stage": "confirm", "min_improvement": 0.05},
            )
            rc_w, res_w = run_guard("gate-decision", "--candidate", str(c_weak), "--incumbent", str(inc), "--baseline", str(inc), "--rules", str(rules))
            self.assertEqual(rc_w, 3)
            self.assertEqual(res_w["decision"], "INCONCLUSIVE_RETEST")
            self.assertGreater(res_w["lcb_one_sided_vs_incumbent"], 0.0)
            self.assertLessEqual(res_w["margin_excess_lcb_one_sided"], 0.0)

            rc_s, res_s = run_guard("gate-decision", "--candidate", str(c_strong), "--incumbent", str(inc), "--baseline", str(inc), "--rules", str(rules))
            self.assertEqual(rc_s, 0)
            self.assertEqual(res_s["decision"], "PROMOTE")
            self.assertGreater(res_s["margin_excess_lcb_one_sided"], 0.02)

            # 20b: check-preconditions includes min_improvement in MDE, blocks 2^-G > alpha/K, and warns on zero-floor margin
            pre = write_json(
                p / "pre.json",
                {
                    "archetype": "ml",
                    "verifier_type": "deterministic",
                    "has_disjoint_dev_and_holdout": True,
                    "evaluator_locked_and_isolated": True,
                    "hermetic_reset_available": True,
                    "n_confirm_clusters": 4,
                    "confirm_alpha": 0.025,
                    "max_confirm_looks": 2,
                    "paired_noise_se": 0.01,
                    "min_improvement": 0.0005,
                    "comparator_search_metric": 0.00052,
                    "direction": "minimize",
                    "metric_bound": 0.0,
                    "current_incumbent_id": "w4_champ",
                    "required_comparators": ["base"],
                },
            )
            rc_p, res_p = run_guard("check-preconditions", "--config", str(pre))
            self.assertEqual(rc_p, 1)
            self.assertEqual(res_p["verdict"], "NO_GO")
            self.assertTrue(any("B8_UNDERPOWERED_CLUSTER_SIGNFLIP_FLOOR" in b for b in res_p["blockers"]))
            self.assertTrue(any("S3_ADDITIVE_MARGIN_NEAR_BOUNDARY" in w for w in res_p["warnings"]))
            self.assertTrue(any("S5_MISSING_INCUMBENT_IN_COMPARATORS" in w for w in res_p["warnings"]))
            self.assertGreater(res_p["mde_80_power"], 0.0005 + 0.025)

    def test_21_crossed_3seed_lottery_blocked_while_consistent_3seed_promotes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            # S=3 training seeds x G=20 clusters:
            # Lottery candidate has seed shifts [+0.10, +0.12, -0.01] -> ALL 20 cluster averages are +0.070,
            # and 26/27 (96.3% > 95%) of naive 3-seed bootstrap draws are positive, YET true Satterthwaite
            # df=2 t-bound (t_0.95,2 = 2.92) correctly sees SE=0.0404 -> LCB = -0.048 < 0 and blocks!
            inc_items = []
            lottery_items = []
            solid_items = []
            lot_shifts = [0.10, 0.12, -0.01]
            sol_shifts = [0.10, 0.11, 0.105]
            for s_idx in range(3):
                for g_idx in range(20):
                    iid = f"s{s_idx}_g{g_idx:02d}"
                    jitter = 0.001 if (g_idx % 2 == 0) else -0.001
                    inc_items.append({"id": iid, "seed_id": f"s{s_idx}", "cluster_id": f"g{g_idx:02d}", "value": 0.50})
                    lottery_items.append(
                        {"id": iid, "seed_id": f"s{s_idx}", "cluster_id": f"g{g_idx:02d}", "value": 0.50 + lot_shifts[s_idx] + jitter}
                    )
                    solid_items.append(
                        {"id": iid, "seed_id": f"s{s_idx}", "cluster_id": f"g{g_idx:02d}", "value": 0.50 + sol_shifts[s_idx] + jitter}
                    )
            inc = write_json(p / "inc.json", {"candidate_id": "inc", "metrics": {"acc": 0.50}, "items": inc_items})
            c_lot = write_json(p / "lot.json", {"candidate_id": "lot", "metrics": {"acc": 0.57}, "items": lottery_items})
            c_sol = write_json(p / "sol.json", {"candidate_id": "sol", "metrics": {"acc": 0.605}, "items": solid_items})
            rules = write_json(
                p / "rules.json",
                {"primary_metric": "acc", "direction": "maximize", "decision_stage": "confirm", "alpha_one_sided": 0.05, "min_improvement": 0.02},
            )
            rc_lot, res_lot = run_guard("gate-decision", "--candidate", str(c_lot), "--incumbent", str(inc), "--baseline", str(inc), "--rules", str(rules))
            self.assertNotEqual(rc_lot, 0)
            self.assertNotEqual(res_lot["decision"], "PROMOTE")
            self.assertLess(res_lot["lcb_one_sided_vs_incumbent"], 0.0)

            rc_sol, res_sol = run_guard("gate-decision", "--candidate", str(c_sol), "--incumbent", str(inc), "--baseline", str(inc), "--rules", str(rules))
            self.assertEqual(rc_sol, 0)
            self.assertEqual(res_sol["decision"], "PROMOTE")
            self.assertGreater(res_sol["margin_excess_lcb_one_sided"], 0.05)

    def test_22_auto_required_incumbent_and_relative_headroom_and_log_ratio_margins(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            # 22a: Near-zero D9 scenario (Dykstra-50 = 5.22e-4 vs exact KKT = 1e-10 across 12 clusters)
            dyk_items = [{"id": f"i{k}", "cluster_id": f"c{k}", "value": 4.0e-4 + 2.0e-5 * k} for k in range(12)]
            kkt_items = [{"id": f"i{k}", "cluster_id": f"c{k}", "value": 1.0e-10} for k in range(12)]
            dyk = write_json(p / "dyk.json", {"candidate_id": "dyk", "metrics": {"rmse": 5.1e-4}, "items": dyk_items})
            kkt = write_json(p / "kkt.json", {"candidate_id": "kkt", "metrics": {"rmse": 1.0e-10}, "items": kkt_items})

            rules_hr = write_json(
                p / "rules_hr.json",
                {
                    "primary_metric": "rmse",
                    "direction": "minimize",
                    "decision_stage": "confirm",
                    "margin_mode": "relative_headroom",
                    "metric_bound": 0.0,
                    "min_improvement": 0.25,
                },
            )
            rc_hr, res_hr = run_guard("gate-decision", "--candidate", str(kkt), "--incumbent", str(dyk), "--baseline", str(dyk), "--rules", str(rules_hr))
            self.assertEqual(rc_hr, 0)
            self.assertEqual(res_hr["decision"], "PROMOTE")

            rules_lr = write_json(
                p / "rules_lr.json",
                {
                    "primary_metric": "rmse",
                    "direction": "minimize",
                    "decision_stage": "confirm",
                    "margin_mode": "log_ratio",
                    "min_improvement": 10.0,
                },
            )
            rc_lr, res_lr = run_guard("gate-decision", "--candidate", str(kkt), "--incumbent", str(dyk), "--baseline", str(dyk), "--rules", str(rules_lr))
            self.assertEqual(rc_lr, 0)
            self.assertEqual(res_lr["decision"], "PROMOTE")

            # 22b: Auto-required current_incumbent_id blocks Holdout unsealing when incumbent wasn't beaten (D4 fix)
            ev = p / "eval"
            ev.mkdir()
            art = p / "solver.py"
            art.write_text("SOLVER = 1\n", encoding="utf-8")
            rules_inc = write_json(
                ev / "rules_inc.json",
                {
                    "primary_metric": "rmse",
                    "direction": "minimize",
                    "decision_stage": "confirm",
                    "margin_mode": "relative_headroom",
                    "metric_bound": 0.0,
                    "min_improvement": 0.10,
                    "required_comparators": ["dyk"],
                    "current_incumbent_id": "w4_champ",
                },
            )
            mf = p / "MANIFEST.json"
            _, mfr = run_guard("freeze-manifest", "--manifest", str(mf), "--root", str(p), "--paths", "eval", check=True)
            ledger = p / "ledger.jsonl"
            run_guard(
                "gate-decision", "--candidate", str(kkt), "--incumbent", str(dyk), "--baseline", str(dyk),
                "--rules", str(rules_inc), "--ledger", str(ledger), "--comparator-id", "dyk", "--candidate-artifacts", str(art), check=True,
            )
            seal = p / "SEAL.json"
            rc_blk, res_blk = run_guard(
                "gate-holdout", "--phase", "reserve", "--seal-file", str(seal), "--manifest", str(mf),
                "--expected-root-sha256", mfr["manifest_root_sha256"], "--ledger", str(ledger), "--rules", str(rules_inc),
                "--candidate-id", "kkt", "--candidate-artifacts", str(art),
            )
            self.assertEqual(rc_blk, 1)
            self.assertEqual(res_blk["status"], "BLOCKED_NOT_CONFIRMED")
            self.assertIn("w4_champ:MISSING", res_blk["missing_or_failed_comparators"])

    def test_23_m0_anchored_noninferiority_efficiency_compound_gate_blocks_biocreep(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            m0_items = [{"id": f"q{k}", "cluster_id": f"b{k}", "value": 0.800, "aux": {"lat_ms": 100.0, "flip_rate": 0.0}} for k in range(12)]
            c1_items = [{"id": f"q{k}", "cluster_id": f"b{k}", "value": 0.797 + (0.001 if k % 2 else -0.001), "aux": {"lat_ms": 65.0 + (k % 3), "flip_rate": 0.02}} for k in range(12)]
            # Biocreep candidate: drops only 0.007 vs c1, BUT drops 0.012 vs M0 (> max_quality_drop_vs_baseline=0.008)!
            creep_items = [{"id": f"q{k}", "cluster_id": f"b{k}", "value": 0.788, "aux": {"lat_ms": 45.0, "flip_rate": 0.03}} for k in range(12)]
            slow_items = [{"id": f"q{k}", "cluster_id": f"b{k}", "value": 0.802, "aux": {"lat_ms": 94.0, "flip_rate": 0.01}} for k in range(12)]

            m0 = write_json(p / "m0.json", {"candidate_id": "m0", "metrics": {"acc": 0.800, "flip_rate": 0.0}, "items": m0_items})
            c1 = write_json(p / "c1.json", {"candidate_id": "c1", "metrics": {"acc": 0.797, "flip_rate": 0.02}, "items": c1_items})
            creep = write_json(p / "creep.json", {"candidate_id": "creep", "metrics": {"acc": 0.788, "flip_rate": 0.03}, "items": creep_items})
            slow = write_json(p / "slow.json", {"candidate_id": "slow", "metrics": {"acc": 0.802, "flip_rate": 0.01}, "items": slow_items})

            rules = write_json(
                p / "rules_ni.json",
                {
                    "primary_metric": "acc",
                    "direction": "maximize",
                    "decision_stage": "confirm",
                    "decision_mode": "noninferiority_efficiency",
                    "max_quality_drop_vs_baseline": 0.008,
                    "efficiency_metric": "lat_ms",
                    "efficiency_direction": "minimize",
                    "min_speedup_ratio": 1.25,
                    "hard_ceilings": {"flip_rate": 0.05},
                },
            )
            rc1, r1 = run_guard("gate-decision", "--candidate", str(c1), "--incumbent", str(m0), "--baseline", str(m0), "--rules", str(rules))
            self.assertEqual(rc1, 0)
            self.assertEqual(r1["decision"], "PROMOTE")

            rc_cr, r_cr = run_guard("gate-decision", "--candidate", str(creep), "--incumbent", str(c1), "--baseline", str(m0), "--rules", str(rules))
            self.assertEqual(rc_cr, 1)
            self.assertEqual(r_cr["decision"], "ROLLBACK")
            self.assertIn("m0_quality_noninferiority_failed", r_cr["decision_reason"])

            rc_sl, r_sl = run_guard("gate-decision", "--candidate", str(slow), "--incumbent", str(m0), "--baseline", str(m0), "--rules", str(rules))
            self.assertEqual(rc_sl, 1)
            self.assertEqual(r_sl["decision"], "ROLLBACK")
            self.assertIn("efficiency_gain_insufficient", r_sl["decision_reason"])

    def test_24_select_search_winner_is_order_invariant(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            base_it = [{"id": f"i{k}", "cluster_id": f"c{k}", "value": 0.066} for k in range(10)]
            c1_it = [{"id": f"i{k}", "cluster_id": f"c{k}", "value": 0.061} for k in range(10)]
            c2_it = [{"id": f"i{k}", "cluster_id": f"c{k}", "value": 0.058} for k in range(10)]
            bad_it = [{"id": f"i{k}", "cluster_id": f"c{k}", "value": 0.052} for k in range(10)]

            base = write_json(p / "base.json", {"candidate_id": "base", "metrics": {"sw1": 0.066, "manifold_err": 1e-12}, "items": base_it})
            ca = write_json(p / "ca.json", {"candidate_id": "opus_r1", "metrics": {"sw1": 0.061, "manifold_err": 1e-12, "eval_wall_clock_sec": 3.0}, "items": c1_it})
            cb = write_json(p / "cb.json", {"candidate_id": "astra_r1", "metrics": {"sw1": 0.058, "manifold_err": 1e-12, "eval_wall_clock_sec": 2.0}, "items": c2_it})
            cbad = write_json(p / "cbad.json", {"candidate_id": "opus_r2_drift", "metrics": {"sw1": 0.052, "manifold_err": 0.056, "eval_wall_clock_sec": 2.5}, "items": bad_it})
            rules = write_json(
                p / "rules_s.json",
                {
                    "primary_metric": "sw1",
                    "direction": "minimize",
                    "decision_stage": "search",
                    "min_improvement": 0.004,
                    "hard_ceilings": {"manifold_err": 1e-4},
                },
            )
            rc1, r1 = run_guard(
                "select-search-winner", "--candidates", str(ca), str(cb), str(cbad),
                "--incumbent", str(base), "--baseline", str(base), "--rules", str(rules),
            )
            rc2, r2 = run_guard(
                "select-search-winner", "--candidates", str(cbad), str(cb), str(ca),
                "--incumbent", str(base), "--baseline", str(base), "--rules", str(rules),
            )
            self.assertEqual(rc1, 0)
            self.assertEqual(rc2, 0)
            self.assertEqual(r1["selected_candidate_id"], "astra_r1")
            self.assertEqual(r2["selected_candidate_id"], "astra_r1")

    def test_25_audit_leaks_v2_allowlist_lean_binary_testskip_and_audit_splits(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            # 25a: Allowlist diff escape + test skip marker
            diff_f = p / "escape.diff"
            diff_f.write_text(
                "--- a/Cargo.toml\n"
                "+++ b/Cargo.toml\n"
                "@@ -1,2 +1,3 @@\n"
                "+opt-level = 3\n"
                "--- a/tests/test_core.rs\n"
                "+++ b/tests/test_core.rs\n"
                "@@ -10,2 +10,3 @@\n"
                "+#[ignore]\n",
                encoding="utf-8",
            )
            rc_d, res_d = run_guard("audit-leaks", "--paths", str(diff_f), "--allowlist", "src/*.rs")
            self.assertEqual(rc_d, 1)
            codes_d = {f["code"] for f in res_d["findings"]}
            self.assertIn("UNAUTHORIZED_FILE_OUTSIDE_ALLOWLIST", codes_d)
            self.assertIn("TEST_SKIP_MARKER_ADDED", codes_d)

            # 25b: Lean 4 comment-aware proof bypass + Python GC heap inspection + .pth hook
            lean_clean = p / "Clean.lean"
            lean_clean.write_text("/- nested /- sorry -/ comment -/\n-- sorry in line comment\ntheorem ok : 1 = 1 := rfl\n", encoding="utf-8")
            rc_lc, _ = run_guard("audit-leaks", "--paths", str(lean_clean))
            self.assertEqual(rc_lc, 0)

            lean_bad = p / "Cheat.lean"
            lean_bad.write_text("theorem cheat : False := by sorry\n", encoding="utf-8")
            py_gc = p / "heap_scrape.py"
            py_gc.write_text("import gc\nobjs = gc.get_objects()\n", encoding="utf-8")
            pth_hook = p / "evil_hook.pth"
            pth_hook.write_text("import os\n", encoding="utf-8")
            rc_lb, res_lb = run_guard("audit-leaks", "--paths", str(lean_bad), str(py_gc), str(pth_hook))
            self.assertEqual(rc_lb, 1)
            codes_lb = {f["code"] for f in res_lb["findings"]}
            self.assertIn("LEAN_SORRY_OR_ADMIT", codes_lb)
            self.assertIn("GC_HEAP_INSPECTION_EXPLOIT", codes_lb)
            self.assertIn("UNTRUSTED_BINARY_OR_STARTUP_HOOK", codes_lb)

            # 25c: audit-splits detects exact feature-row duplicates, entity overlap, and temporal lookahead
            tr_f = write_json(
                p / "train.json",
                [
                    {"x1": 1.0, "x2": 2.0, "user_id": "u1", "ts": 100, "y": 0},
                    {"x1": 3.0, "x2": 4.0, "user_id": "u2", "ts": 200, "y": 1},
                ],
            )
            cf_bad = write_json(
                p / "confirm_bad.json",
                [
                    {"x1": 1.0, "x2": 2.0, "user_id": "u1", "ts": 150, "y": 1},
                ],
            )
            rc_sb, res_sb = run_guard(
                "audit-splits", "--train", str(tr_f), "--confirm", str(cf_bad),
                "--group-key", "user_id", "--timestamp-key", "ts",
            )
            self.assertEqual(rc_sb, 1)
            scodes = {f["code"] for f in res_sb["findings"]}
            self.assertIn("EXACT_FEATURE_ROW_LEAKAGE", scodes)
            self.assertIn("GROUP_ENTITY_LEAKAGE", scodes)
            self.assertIn("TEMPORAL_LOOKAHEAD_LEAKAGE", scodes)

            cf_ok = write_json(
                p / "confirm_ok.json",
                [{"x1": 9.0, "x2": 8.0, "user_id": "u9", "ts": 250, "y": 1}],
            )
            rc_so, _ = run_guard(
                "audit-splits", "--train", str(tr_f), "--confirm", str(cf_ok),
                "--group-key", "user_id", "--timestamp-key", "ts", "--embargo-gap", "20",
            )
            self.assertEqual(rc_so, 0)

    def test_26_run_isolated_build_test_bench_strips_git_pins_artifact_and_checks_test_count(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            repo = p / "repo"
            (repo / ".git" / "logs").mkdir(parents=True)
            (repo / ".git" / "logs" / "HEAD").write_text("SECRET_FUTURE_COMMIT\n", encoding="utf-8")
            (repo / "pkg").mkdir(parents=True)
            (repo / "pkg" / "__init__.py").write_text("", encoding="utf-8")
            (repo / "pkg" / "fastmath.py").write_text("def double(x):\n    return x * 2\n", encoding="utf-8")
            (repo / "build_step.py").write_text(
                "from pathlib import Path\n"
                "assert not Path('.git').exists(), '.git was NOT stripped!'\n"
                "Path('dist').mkdir(exist_ok=True)\n"
                "Path('dist/compiled.artifact').write_text('BIN_V1', encoding='utf-8')\n",
                encoding="utf-8",
            )
            trusted_tests = p / "trusted_tests"
            trusted_tests.mkdir()
            (trusted_tests / "__init__.py").write_text("", encoding="utf-8")
            (trusted_tests / "test_fastmath.py").write_text(
                "import unittest\n"
                "from pkg.fastmath import double\n"
                "class T(unittest.TestCase):\n"
                "    def test_1(self): self.assertEqual(double(2), 4)\n"
                "    def test_2(self): self.assertEqual(double(-3), -6)\n"
                "    def test_3(self): self.assertEqual(double(0), 0)\n"
                "if __name__ == '__main__': unittest.main()\n",
                encoding="utf-8",
            )
            (repo / "bench_runner.py").write_text(
                "import argparse, json, sys\n"
                "from pathlib import Path\n"
                "sys.path.insert(0, str(Path(__file__).resolve().parent))\n"
                "from pkg.fastmath import double\n"
                "ap = argparse.ArgumentParser()\n"
                "ap.add_argument('--output-json', required=True)\n"
                "a = ap.parse_args()\n"
                "assert double(5) == 10\n"
                "items = [{'id': f'b{k}', 'cluster_id': f'b{k}', 'value': 1.25} for k in range(8)]\n"
                "open(a.output_json, 'w', encoding='utf-8').write(json.dumps({'candidate_id': 'c1', 'metrics': {'lat_ms': 1.25}, 'items': items}))\n",
                encoding="utf-8",
            )
            out_ok = p / "btb_ok.json"
            rc_ok, res_ok = run_guard(
                "run-isolated", "--mode", "build_test_bench",
                "--workspace-root", str(repo),
                "--build-cmd", "{python} build_step.py",
                "--artifact-glob", "dist/*.artifact",
                "--trusted-tests-dir", str(trusted_tests),
                "--trusted-tests-relpath", "tests",
                "--test-cmd", "{python} -m unittest discover -s tests -t .",
                "--min-tests-executed", "3",
                "--bench-cmd", "{python} bench_runner.py --output-json {output_json}",
                "--output-json", str(out_ok),
            )
            self.assertEqual(rc_ok, 0)
            self.assertEqual(res_ok["status"], "OK")
            self.assertEqual(res_ok["tests_executed"], 3)
            self.assertIsNotNone(res_ok["built_artifact_sha256"])

            # Requiring 5 tests when only 3 ran must fail closed with TEST_COUNT_REGRESSION
            rc_reg, res_reg = run_guard(
                "run-isolated", "--mode", "build_test_bench",
                "--workspace-root", str(repo),
                "--trusted-tests-dir", str(trusted_tests),
                "--trusted-tests-relpath", "tests",
                "--test-cmd", "{python} -m unittest discover -s tests -t .",
                "--min-tests-executed", "5",
                "--bench-cmd", "{python} bench_runner.py --output-json {output_json}",
                "--output-json", str(p / "btb_reg.json"),
            )
            self.assertEqual(rc_reg, 1)
            self.assertEqual(res_reg["status"], "TEST_COUNT_REGRESSION")

    def test_27_run_isolated_train_lock_score_and_generic_causal_online(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            imm = p / "immutable_eval"
            imm.mkdir()
            scorer = imm / "score_locked.py"
            scorer.write_text(
                "import argparse, json\n"
                "from pathlib import Path\n"
                "ap = argparse.ArgumentParser()\n"
                "ap.add_argument('--ckpt', required=True)\n"
                "ap.add_argument('--output-json', required=True)\n"
                "a = ap.parse_args()\n"
                "w = json.loads((Path(a.ckpt) / 'weights.safetensors.json').read_text(encoding='utf-8'))['w']\n"
                "items = [{'id': f'c{k}', 'cluster_id': f'c{k}', 'value': w * 10.0} for k in range(8)]\n"
                "Path(a.output_json).write_text(json.dumps({'candidate_id': 'm1', 'metrics': {'val_score': w * 10.0}, 'items': items}), encoding='utf-8')\n",
                encoding="utf-8",
            )

            # 27a: Reject unsafe .pt pickle checkpoint
            ws_pkl = p / "ws_pkl"
            ws_pkl.mkdir()
            (ws_pkl / "train.py").write_text(
                "from pathlib import Path\n"
                "Path('ckpt').mkdir(exist_ok=True)\n"
                "Path('ckpt/model.pt').write_bytes(b'PICKLE')\n",
                encoding="utf-8",
            )
            rc_pkl, res_pkl = run_guard(
                "run-isolated", "--mode", "train_lock_score",
                "--workspace-root", str(ws_pkl),
                "--forbid-paths", str(imm),
                "--train-cmd", "{python} train.py",
                "--checkpoint-relpath", "ckpt",
                "--score-cmd", f"{{python}} {scorer.as_posix()} --ckpt {{checkpoint_path}} --output-json {{output_json}}",
                "--output-json", str(p / "out_pkl.json"),
            )
            self.assertEqual(rc_pkl, 1)
            self.assertEqual(res_pkl["status"], "UNSAFE_PICKLE_CHECKPOINT_FORBIDDEN")

            # 27b: Reject checkpoint selected by peeking on Confirm split
            ws_peek = p / "ws_peek"
            ws_peek.mkdir()
            (ws_peek / "train.py").write_text(
                "import json\nfrom pathlib import Path\n"
                "Path('ckpt').mkdir(exist_ok=True)\n"
                "Path('ckpt/weights.safetensors.json').write_text(json.dumps({'w': 0.9}), encoding='utf-8')\n"
                "Path('ckpt/checkpoint_meta.json').write_text(json.dumps({'selected_on_split': 'confirm'}), encoding='utf-8')\n",
                encoding="utf-8",
            )
            rc_pk, res_pk = run_guard(
                "run-isolated", "--mode", "train_lock_score",
                "--workspace-root", str(ws_peek),
                "--forbid-paths", str(imm),
                "--train-cmd", "{python} train.py",
                "--checkpoint-relpath", "ckpt",
                "--score-cmd", f"{{python}} {scorer.as_posix()} --ckpt {{checkpoint_path}} --output-json {{output_json}}",
                "--output-json", str(p / "out_pk.json"),
            )
            self.assertEqual(rc_pk, 1)
            self.assertEqual(res_pk["status"], "CHECKPOINT_PEEKING_FORBIDDEN")

            # 27c: Clean inner-val checkpoint passes train -> hash lock -> delete train code -> score
            ws_ok = p / "ws_ok"
            ws_ok.mkdir()
            (ws_ok / "train.py").write_text(
                "import json\nfrom pathlib import Path\n"
                "assert not Path('immutable_eval').exists()\n"
                "Path('ckpt').mkdir(exist_ok=True)\n"
                "Path('ckpt/weights.safetensors.json').write_text(json.dumps({'w': 0.85}), encoding='utf-8')\n"
                "Path('ckpt/checkpoint_meta.json').write_text(json.dumps({'selected_on_split': 'train_inner_val'}), encoding='utf-8')\n",
                encoding="utf-8",
            )
            out_tls = p / "out_tls.json"
            rc_ok, res_ok = run_guard(
                "run-isolated", "--mode", "train_lock_score",
                "--workspace-root", str(ws_ok),
                "--forbid-paths", str(imm),
                "--train-cmd", "{python} train.py",
                "--checkpoint-relpath", "ckpt",
                "--score-cmd", f"{{python}} {scorer.as_posix()} --ckpt {{checkpoint_path}} --output-json {{output_json}}",
                "--output-json", str(out_tls),
            )
            self.assertEqual(rc_ok, 0)
            self.assertEqual(res_ok["status"], "OK")
            tls_data = json.loads(out_tls.read_text(encoding="utf-8"))
            self.assertAlmostEqual(tls_data["metrics"]["val_score"], 8.5, places=6)

            # 27d: Generic causal_online step protocol hides action_values from candidate stdin
            pol_worker = p / "pol.py"
            pol_worker.write_text(
                "import sys, json\n"
                "for line in sys.stdin:\n"
                "    req = json.loads(line)\n"
                "    assert 'action_values' not in req\n"
                "    act = 'fast' if req['obs']['load'] > 0.5 else 'accurate'\n"
                "    sys.stdout.write(json.dumps({'action': act}) + '\\n')\n"
                "    sys.stdout.flush()\n",
                encoding="utf-8",
            )
            gen_streams = write_json(
                p / "gen_streams.json",
                {
                    "protocol": "generic_step",
                    "primary_metric": "reward",
                    "streams": [
                        {
                            "id": "ep1",
                            "cluster_id": "u1",
                            "steps": [
                                {"obs": {"load": 0.8}, "valid_actions": ["fast", "accurate"], "action_values": {"fast": 1.0, "accurate": 0.2}},
                                {"obs": {"load": 0.2}, "valid_actions": ["fast", "accurate"], "action_values": {"fast": 0.3, "accurate": 0.9}},
                            ],
                        }
                    ],
                },
            )
            rc_go, res_go = run_guard(
                "run-isolated", "--mode", "causal_online", "--candidate-script", str(pol_worker),
                "--input-json", str(gen_streams), "--output-json", str(p / "out_go.json"),
            )
            self.assertEqual(rc_go, 0)
            self.assertAlmostEqual(res_go["reward"], 0.95, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
