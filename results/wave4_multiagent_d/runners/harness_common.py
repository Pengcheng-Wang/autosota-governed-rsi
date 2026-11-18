#!/usr/bin/env python3
"""Shared Governed Loop Guard v3.6 orchestration helpers for Wave-4 Multi-Agent RSI."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

W4_ROOT = Path("/scratch/autosota_governed_v4")
GUARD = W4_ROOT / "guard" / "governed_loop_guard.py"


def write_json(path: Path, obj: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run_guard(*args: str, allow_fail: bool = False) -> tuple[int, dict[str, Any]]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    p = subprocess.run(
        [sys.executable, str(GUARD), *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    out = (p.stdout or "").strip()
    data: dict[str, Any] = {}
    if out:
        try:
            data = json.loads(out)
        except Exception:
            data = {"raw_stdout": out, "raw_stderr": (p.stderr or "")[:500]}
    if p.returncode != 0 and not allow_fail:
        raise RuntimeError(f"Guard failed ({p.returncode}) on {args[:4]}: {out[:600]} {(p.stderr or '')[:400]}")
    return p.returncode, data


def run_isolated_predict(
    candidate_py: Path,
    input_json: Path,
    output_json: Path,
    expected_ids_json: Path,
    forbid_dir: Path,
    timeout_sec: int = 300,
) -> dict[str, Any]:
    run_guard("audit-leaks", "--paths", str(candidate_py))
    _, iso = run_guard(
        "run-isolated",
        "--mode", "predict",
        "--candidate-script", str(candidate_py),
        "--input-json", str(input_json),
        "--output-json", str(output_json),
        "--expected-ids-json", str(expected_ids_json),
        "--forbid-paths", str(forbid_dir),
        "--timeout-sec", str(timeout_sec),
    )
    return iso


def execute_two_round_tournament(
    bench_id: str,
    bdir: Path,
    score_fn,
    baseline_id: str,
    comparators: list[str],
    autosota_audit_id: str | None,
    round1_candidates: dict[str, str],  # {cand_id: proposer_agent_tag}
    round2_candidates: dict[str, str],  # {cand_id: proposer_agent_tag}
    rules_search: dict[str, Any],
    rules_confirm: dict[str, Any],
    phase: str,
) -> dict[str, Any]:
    """
    Executes either:
      - phase == 'round1': Freezes manifest, runs baselines + AutoSOTA audit + all Round-1 proposer candidates
        on SEARCH ONLY (writing redacted proposer_verdict.json, keeping Confirm & Holdout 100% untouched!).
      - phase == 'round2_final': Verifies manifest, runs Round-2 iterative candidates on SEARCH, selects overall
        Search winner, runs Confirmation Gate across all required_comparators, and if PROMOTE holds across all,
        executes atomic single-touch Protected Holdout (`gate-holdout reserve -> finalize`).
    """
    imm = bdir / "immutable_eval"
    runs = bdir / "runs"
    cdir = bdir / "candidates"
    mf_path = bdir / "IMMUTABLE_MANIFEST.json"
    pin_path = W4_ROOT / "seals" / f"PIN_{bench_id}.txt"
    ledger_path = bdir / "trial_ledger.jsonl"
    rs_path = imm / "gate_rules_search.json"
    rc_path = imm / "gate_rules_confirm.json"
    r1_state_path = runs / "round1_state.json"

    if phase == "round1":
        write_json(rs_path, rules_search)
        write_json(rc_path, rules_confirm)
        _, mf_info = run_guard("freeze-manifest", "--manifest", str(mf_path), "--root", str(bdir), "--paths", "immutable_eval")
        root_sha = mf_info["manifest_root_sha256"]
        pin_path.parent.mkdir(parents=True, exist_ok=True)
        pin_path.write_text(root_sha, encoding="utf-8")
        if ledger_path.exists():
            ledger_path.unlink()

        base_res = score_fn(baseline_id, "search")
        base_p = write_json(runs / f"search_{baseline_id}.json", base_res)

        autosota_audit = None
        if autosota_audit_id:
            auto_res = score_fn(autosota_audit_id, "search")
            auto_p = write_json(runs / f"search_{autosota_audit_id}.json", auto_res)
            _, auto_dec = run_guard(
                "gate-decision",
                "--candidate", str(auto_p),
                "--incumbent", str(base_p),
                "--baseline", str(base_p),
                "--rules", str(rs_path),
                "--trial-id", f"{bench_id}_audit_{autosota_audit_id}",
                allow_fail=True,
            )
            autosota_audit = {
                "autosota_id": autosota_audit_id,
                "baseline_metrics": base_res["metrics"],
                "autosota_metrics": auto_res["metrics"],
                "decision": auto_dec["decision"],
                "tripwires": auto_dec["tripwires"],
            }

        inc_p = base_p
        inc_id = baseline_id
        r1_board = {}
        for idx, (cid, agent_tag) in enumerate(round1_candidates.items(), start=1):
            c_py = cdir / f"{cid}.py"
            res = score_fn(cid, "search")
            cp = write_json(runs / f"search_{cid}.json", res)
            redacted_p = runs / f"proposer_r1_{cid}.json"
            _, dec = run_guard(
                "gate-decision",
                "--candidate", str(cp),
                "--incumbent", str(inc_p),
                "--baseline", str(base_p),
                "--rules", str(rs_path),
                "--ledger", str(ledger_path),
                "--trial-id", f"{bench_id}_r1_{idx:02d}_{cid}",
                "--candidate-artifacts", str(c_py),
                "--proposer-out", str(redacted_p),
                "--hypothesis", f"Round-1 proposal from {agent_tag}",
                allow_fail=True,
            )
            r1_board[cid] = {
                "proposer_agent": agent_tag,
                "metrics": res["metrics"],
                "decision_vs_running_incumbent": dec["decision"],
                "tripwires": dec["tripwires"],
                "redacted_feedback": read_json(redacted_p),
            }
            if dec["decision"] == "PROVISIONAL_KEEP":
                inc_p = cp
                inc_id = cid

        state = {
            "bench_id": bench_id,
            "manifest_root_sha256": root_sha,
            "baseline_search": base_res["metrics"],
            "autosota_audit": autosota_audit,
            "round1_leaderboard": r1_board,
            "round1_incumbent_id": inc_id,
        }
        write_json(r1_state_path, state)
        write_json(W4_ROOT / "results" / f"r1_{bench_id}.json", state)
        return state

    elif phase == "round2_final":
        root_sha = pin_path.read_text(encoding="utf-8").strip()
        run_guard("verify-manifest", "--manifest", str(mf_path), "--expected-root-sha256", root_sha)
        r1_state = read_json(r1_state_path)
        base_p = runs / f"search_{baseline_id}.json"
        inc_id = r1_state["round1_incumbent_id"]
        inc_p = runs / f"search_{inc_id}.json"

        r2_board = {}
        for idx, (cid, agent_tag) in enumerate(round2_candidates.items(), start=1):
            c_py = cdir / f"{cid}.py"
            res = score_fn(cid, "search")
            cp = write_json(runs / f"search_{cid}.json", res)
            redacted_p = runs / f"proposer_r2_{cid}.json"
            _, dec = run_guard(
                "gate-decision",
                "--candidate", str(cp),
                "--incumbent", str(inc_p),
                "--baseline", str(base_p),
                "--rules", str(rs_path),
                "--ledger", str(ledger_path),
                "--trial-id", f"{bench_id}_r2_{idx:02d}_{cid}",
                "--candidate-artifacts", str(c_py),
                "--proposer-out", str(redacted_p),
                "--hypothesis", f"Round-2 iterative proposal from {agent_tag}",
                allow_fail=True,
            )
            r2_board[cid] = {
                "proposer_agent": agent_tag,
                "metrics": res["metrics"],
                "decision_vs_running_incumbent": dec["decision"],
                "tripwires": dec["tripwires"],
            }
            if dec["decision"] == "PROVISIONAL_KEEP":
                inc_p = cp
                inc_id = cid

        # If no candidate beat baseline on Search, still test best valid candidate or halt cleanly
        if inc_id == baseline_id:
            # Pick best tripwire-clean candidate across R1+R2 for diagnostic Confirm or halt if none beat baseline
            summary = {
                "benchmark": bench_id,
                "manifest_root_sha256": root_sha,
                "baseline_search": r1_state["baseline_search"],
                "autosota_audit": r1_state["autosota_audit"],
                "round1_leaderboard": r1_state["round1_leaderboard"],
                "round2_leaderboard": r2_board,
                "selected_finalist": None,
                "status": "NO_SEARCH_GAIN_HALTED_BEFORE_CONFIRM",
                "confirm_verdicts": {},
                "holdout_reserve": {"status": "NOT_ATTEMPTED_NO_SEARCH_WIN"},
                "holdout_finalized": None,
                "ledger_verification": run_guard("verify-ledger", "--ledger", str(ledger_path))[1],
            }
            write_json(W4_ROOT / "results" / f"summary_{bench_id}.json", summary)
            return summary

        best_cid = inc_id
        best_py = cdir / f"{best_cid}.py"

        # Evaluate mandatory comparators + finalist on Confirm
        comp_conf_paths = {}
        for comp in comparators:
            comp_conf_paths[comp] = write_json(runs / f"confirm_{comp}.json", score_fn(comp, "confirm"))
        if baseline_id not in comp_conf_paths:
            comp_conf_paths[baseline_id] = write_json(runs / f"confirm_{baseline_id}.json", score_fn(baseline_id, "confirm"))
        cand_conf_p = write_json(runs / f"confirm_{best_cid}.json", score_fn(best_cid, "confirm"))

        confirm_verdicts = {}
        for comp in comparators:
            _, dec = run_guard(
                "gate-decision",
                "--candidate", str(cand_conf_p),
                "--incumbent", str(comp_conf_paths[comp]),
                "--baseline", str(comp_conf_paths[baseline_id]),
                "--rules", str(rc_path),
                "--ledger", str(ledger_path),
                "--trial-id", f"{bench_id}_confirm_{best_cid}_vs_{comp}",
                "--comparator-id", comp,
                "--candidate-artifacts", str(best_py),
                allow_fail=True,
            )
            confirm_verdicts[comp] = {
                "cand": dec["candidate_primary"],
                "ref": dec["incumbent_primary"],
                "gain": dec["signed_improvement"],
                "lcb": dec["lcb_one_sided_vs_incumbent"],
                "signflip_p": dec.get("signflip_p_one_sided_vs_incumbent"),
                "signflip_method": dec.get("signflip_method"),
                "decision": dec["decision"],
                "tripwires": dec["tripwires"],
            }

        seal_path = W4_ROOT / "seals" / f"HOLDOUT_SEAL_{bench_id}.json"
        rc_h, holdout_res = run_guard(
            "gate-holdout",
            "--phase", "reserve",
            "--seal-file", str(seal_path),
            "--manifest", str(mf_path),
            "--expected-root-sha256", root_sha,
            "--ledger", str(ledger_path),
            "--rules", str(rc_path),
            "--candidate-id", best_cid,
            "--candidate-artifacts", str(best_py),
            allow_fail=True,
        )
        holdout_fin = None
        if rc_h == 0 and holdout_res.get("status") == "RESERVED":
            cand_h = write_json(runs / f"holdout_{best_cid}.json", score_fn(best_cid, "holdout"))
            comp_h_args = []
            for comp in comparators:
                cp_h = write_json(runs / f"holdout_{comp}.json", score_fn(comp, "holdout"))
                comp_h_args.extend(["--comparator-holdout", f"{comp}={cp_h}"])
            _, holdout_fin = run_guard(
                "gate-holdout",
                "--phase", "finalize",
                "--seal-file", str(seal_path),
                "--manifest", str(mf_path),
                "--expected-root-sha256", root_sha,
                "--ledger", str(ledger_path),
                "--rules", str(rc_path),
                "--candidate-id", best_cid,
                "--candidate-artifacts", str(best_py),
                "--candidate-holdout", str(cand_h),
                *comp_h_args,
            )

        _, ledger_ver = run_guard("verify-ledger", "--ledger", str(ledger_path))
        all_proposers = {**round1_candidates, **round2_candidates}
        summary = {
            "benchmark": bench_id,
            "manifest_root_sha256": root_sha,
            "baseline_search": r1_state["baseline_search"],
            "autosota_audit": r1_state["autosota_audit"],
            "round1_leaderboard": r1_state["round1_leaderboard"],
            "round2_leaderboard": r2_board,
            "selected_finalist": best_cid,
            "selected_finalist_proposer": all_proposers.get(best_cid, "unknown"),
            "confirm_verdicts": confirm_verdicts,
            "holdout_reserve": holdout_res,
            "holdout_finalized": holdout_fin,
            "ledger_verification": ledger_ver,
        }
        write_json(W4_ROOT / "results" / f"summary_{bench_id}.json", summary)
        return summary
    else:
        raise ValueError(phase)
