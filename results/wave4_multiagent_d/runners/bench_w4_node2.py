#!/usr/bin/env python3
"""
Wave-4 Multi-Agent Governed RSI Harness — Node 2 (8 x A100-80GB):
  - D7: Paper #73 RLD (Graph Max-Cut Combinatorial Optimization on BA / ER / Regular / Weighted Graphs)
  - D8: Paper #74 LSVI (Fast Full-Covariance Gaussian Variational Inference on Real Tabular Logistic Posteriors)
  - D9: Paper #2 PINet (Hard-Constrained Equality+Box QP Projection Layers under Strict Feasibility <= 1e-4)
All baselines and multi-agent candidates run out-of-process via governed_loop_guard.py run-isolated (python -I -B).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from harness_common import (
    W4_ROOT,
    execute_two_round_tournament,
    read_json,
    run_isolated_predict,
    write_json,
)


# ============================================================================
# D7: Paper #73 RLD — Combinatorial Graph Max-Cut
# ============================================================================
def gen_graph(fam: str, n: int, rng: np.random.RandomState) -> np.ndarray:
    A = np.zeros((n, n), dtype=np.float64)
    if fam == "ba":
        m_attach = 4
        for i in range(1, m_attach + 1):
            A[0, i] = A[i, 0] = 1.0
        deg = np.sum(A, axis=1)
        for v in range(m_attach + 1, n):
            probs = (deg[:v] + 0.5) / np.sum(deg[:v] + 0.5)
            targets = rng.choice(v, size=m_attach, replace=False, p=probs)
            for u in targets:
                A[v, u] = A[u, v] = 1.0
            deg = np.sum(A, axis=1)
    elif fam == "er":
        p = 8.0 / n
        upper = (rng.rand(n, n) < p).astype(np.float64)
        upper = np.triu(upper, k=1)
        A = upper + upper.T
    elif fam == "regular_ish":
        # Union of 3 random Hamiltonian cycles -> 6-regular sparse graph
        for _ in range(3):
            perm = rng.permutation(n)
            for idx in range(n):
                u, v = perm[idx], perm[(idx + 1) % n]
                if u != v:
                    A[u, v] = A[v, u] = 1.0
    elif fam == "weighted_frustrated":
        p = 10.0 / n
        upper = (rng.rand(n, n) < p).astype(np.float64) * rng.uniform(0.3, 2.0, size=(n, n))
        upper = np.triu(upper, k=1)
        A = upper + upper.T
    else:
        raise ValueError(fam)
    return A


def prepare_and_score_d7(phase: str, r1_map: dict[str, str], r2_map: dict[str, str]) -> dict:
    bdir = W4_ROOT / "tracks" / "bench_d7_rld_maxcut"
    imm = bdir / "immutable_eval"
    runs = bdir / "runs"

    if phase == "round1":
        imm.mkdir(parents=True, exist_ok=True)
        runs.mkdir(parents=True, exist_ok=True)
        for st_idx, st in enumerate(["search", "confirm", "holdout"]):
            items_in = []
            items_gt = []
            for fam_idx, fam in enumerate(["ba", "er", "regular_ish", "weighted_frustrated"]):
                for k in range(4):
                    rng = np.random.RandomState(7300 + st_idx * 100 + fam_idx * 10 + k)
                    n_nodes = 140 if k % 2 == 0 else 180
                    A = gen_graph(fam, n_nodes, rng)
                    cid_base = f"{st}_{fam}_{k:02d}"
                    seeds = [0, 1] if st == "search" else [10, 11, 12]
                    for sd in seeds:
                        iid = f"{cid_base}_s{sd}"
                        items_in.append({
                            "id": iid,
                            "seed": sd,
                            "n": n_nodes,
                            "adj": np.round(A, 5).tolist(),
                        })
                        items_gt.append({
                            "id": iid,
                            "cluster_id": cid_base,
                            "stratum_id": fam,
                            "seed_id": f"s{sd}",
                            "adj": np.round(A, 5).tolist(),
                        })
            write_json(runs / f"input_{st}.json", {"items": items_in})
            write_json(imm / f"gt_{st}.json", items_gt)
            write_json(imm / f"ids_{st}.json", [x["id"] for x in items_gt])

    def score_d7(cid: str, stage: str) -> dict:
        c_py = bdir / "candidates" / f"{cid}.py"
        in_p = runs / f"input_{stage}.json"
        out_p = runs / f"preds_{stage}_{cid}.json"
        ids_p = imm / f"ids_{stage}.json"
        iso = run_isolated_predict(c_py, in_p, out_p, ids_p, imm, timeout_sec=240)
        preds_map = {r["id"]: np.asarray(r["prediction"], dtype=np.float64) for r in read_json(out_p)["predictions"]}
        gt_list = read_json(imm / f"gt_{stage}.json")

        items = []
        for g in gt_list:
            iid = g["id"]
            A = np.asarray(g["adj"], dtype=np.float64)
            s_raw = preds_map[iid]
            # Enforce exact {-1, +1} spin projection in verifier
            s = np.where(s_raw >= 0.0, 1.0, -1.0)
            if len(s) != len(A):
                raise RuntimeError(f"Spin length mismatch on {iid}")
            cut_val = float(0.25 * (np.sum(A) - float(s @ A @ s)))
            items.append({
                "id": iid,
                "cluster_id": g["cluster_id"],
                "stratum_id": g["stratum_id"],
                "seed_id": g["seed_id"],
                "value": cut_val,
            })

        by_st = {}
        for it in items:
            by_st.setdefault(it["stratum_id"], []).append(it["value"])
        eq_cut = float(np.mean([np.mean(v) for v in by_st.values()]))
        return {
            "candidate_id": cid,
            "metrics": {
                "maxcut_value": eq_cut,
                "eval_wall_clock_sec": float(iso["guard_wall_clock_sec"]),
            },
            "items": items,
        }

    rules_search = {
        "primary_metric": "maxcut_value",
        "direction": "maximize",
        "aggregation_weighting": "equal_stratum",
        "decision_stage": "search",
        "min_improvement": 0.25,
        "max_ratio_vs_baseline": {"eval_wall_clock_sec": 1.6},
    }
    rules_confirm = {
        "primary_metric": "maxcut_value",
        "direction": "maximize",
        "aggregation_weighting": "equal_stratum",
        "expected_strata": ["ba", "er", "regular_ish", "weighted_frustrated"],
        "decision_stage": "confirm",
        "alpha_one_sided": 0.025,
        "max_confirm_looks": 2,
        "min_improvement": 0.25,
        "min_confirm_clusters": 12,
        "max_ratio_vs_baseline": {"eval_wall_clock_sec": 1.6},
        "required_comparators": [
            "d7_paper_rlsa_langevin",
            "d7_autosota_rlsa_leap",
            "d7_comp_spectral_greedy",
        ],
    }
    return execute_two_round_tournament(
        bench_id="d7",
        bdir=bdir,
        score_fn=score_d7,
        baseline_id="d7_paper_rlsa_langevin",
        comparators=["d7_paper_rlsa_langevin", "d7_autosota_rlsa_leap", "d7_comp_spectral_greedy"],
        autosota_audit_id="d7_autosota_rlsa_leap",
        round1_candidates=r1_map,
        round2_candidates=r2_map,
        rules_search=rules_search,
        rules_confirm=rules_confirm,
        phase=phase,
    )


# ============================================================================
# D8: Paper #74 LSVI — Fast Full-Covariance Gaussian Variational Inference
# ============================================================================
def prepare_and_score_d8(phase: str, r1_map: dict[str, str], r2_map: dict[str, str]) -> dict:
    bdir = W4_ROOT / "tracks" / "bench_d8_lsvi_vi"
    imm = bdir / "immutable_eval"
    runs = bdir / "runs"

    if phase == "round1":
        imm.mkdir(parents=True, exist_ok=True)
        runs.mkdir(parents=True, exist_ok=True)
        acs_df = pd.read_csv(W4_ROOT / "data" / "acs" / "real.csv")
        X_acs_all = acs_df.drop(columns=["PINCP"]).values.astype(np.float64)
        y_acs_all = acs_df["PINCP"].values.astype(np.float64)

        comp_tr = pd.read_csv(W4_ROOT / "data" / "cmnn" / "train_compas.csv", header=None).values.astype(np.float64)
        comp_te = pd.read_csv(W4_ROOT / "data" / "cmnn" / "test_compas.csv", header=None).values.astype(np.float64)
        comp_all = np.vstack([comp_tr, comp_te])
        X_comp_all, y_comp_all = comp_all[:, :-1], comp_all[:, -1]

        rng = np.random.RandomState(742026)
        p_acs = rng.permutation(len(X_acs_all))
        p_comp = rng.permutation(len(X_comp_all))

        for st_idx, st in enumerate(["search", "confirm", "holdout"]):
            items_in = []
            items_gt = []
            # 8 disjoint sub-posteriors from ACS + 8 disjoint sub-posteriors from COMPAS = 16 clusters!
            for fam, X_src, y_src, perm_arr in [
                ("acs_logit", X_acs_all, y_acs_all, p_acs),
                ("compas_logit", X_comp_all, y_comp_all, p_comp),
            ]:
                for b in range(8):
                    idx_slot = st_idx * 8 + b
                    n_sub = 90 if b % 2 == 0 else 150  # finite-sample skewed posteriors where Laplace != VI!
                    sel = perm_arr[idx_slot * 160: idx_slot * 160 + n_sub]
                    Xb = X_src[sel].copy()
                    xm = np.mean(Xb, axis=0)
                    xs = np.where(np.std(Xb, axis=0) > 1e-6, np.std(Xb, axis=0), 1.0)
                    Xb = np.hstack([(Xb - xm) / xs, np.ones((len(Xb), 1))])
                    yb = y_src[sel].copy()
                    d = Xb.shape[1]
                    prior_prec = 0.25 if b % 2 == 0 else 1.0  # diffuse prior -> stronger posterior skewness!

                    # Antithetic 4096+4096 = 8192 common random normal draws for exact low-variance ELBO verification
                    rng_mc = np.random.RandomState(88000 + st_idx * 100 + b + (0 if fam == "acs_logit" else 50))
                    eps_half = rng_mc.normal(0.0, 1.0, size=(4096, d))
                    eps_mc = np.vstack([eps_half, -eps_half])

                    iid = f"{st}_{fam}_b{b:02d}"
                    items_in.append({
                        "id": iid,
                        "d": d,
                        "prior_prec": prior_prec,
                        "X": np.round(Xb, 6).tolist(),
                        "y": yb.tolist(),
                    })
                    items_gt.append({
                        "id": iid,
                        "stratum_id": fam,
                        "d": d,
                        "prior_prec": prior_prec,
                        "X": Xb.tolist(),
                        "y": yb.tolist(),
                        "eps_seed": int(88000 + st_idx * 100 + b + (0 if fam == "acs_logit" else 50)),
                    })

            write_json(runs / f"input_{st}.json", {"items": items_in})
            write_json(imm / f"gt_{st}.json", items_gt)
            write_json(imm / f"ids_{st}.json", [x["id"] for x in items_gt])

    def score_d8(cid: str, stage: str) -> dict:
        c_py = bdir / "candidates" / f"{cid}.py"
        in_p = runs / f"input_{stage}.json"
        out_p = runs / f"preds_{stage}_{cid}.json"
        ids_p = imm / f"ids_{stage}.json"
        iso = run_isolated_predict(c_py, in_p, out_p, ids_p, imm, timeout_sec=240)
        preds_map = {r["id"]: r["prediction"] for r in read_json(out_p)["predictions"]}
        gt_list = read_json(imm / f"gt_{stage}.json")

        items = []
        for g in gt_list:
            iid = g["id"]
            d = int(g["d"])
            prior_prec = float(g["prior_prec"])
            Xb = np.asarray(g["X"], dtype=np.float64)
            yb = np.asarray(g["y"], dtype=np.float64)
            rng_mc = np.random.RandomState(int(g["eps_seed"]))
            eh = rng_mc.normal(0.0, 1.0, size=(4096, d))
            eps_mc = np.vstack([eh, -eh])  # [8192, d]

            p_obj = preds_map[iid]
            mu = np.asarray(p_obj["mu"], dtype=np.float64)
            L = np.tril(np.asarray(p_obj["L"], dtype=np.float64))
            diag_L = np.diag(L)
            if np.any(diag_L <= 1e-10) or not np.all(np.isfinite(mu)) or not np.all(np.isfinite(L)):
                raise RuntimeError(f"Non-PD Cholesky L on {iid} for {cid}")

            thetas = mu[None, :] + eps_mc @ L.T  # [8192, d]
            logits = np.clip(thetas @ Xb.T, -30.0, 30.0)  # [8192, N]
            log_lik = np.sum(yb[None, :] * logits - np.log1p(np.exp(logits)), axis=1)
            log_prior = -0.5 * prior_prec * np.sum(thetas ** 2, axis=1) + 0.5 * d * math.log(prior_prec / (2.0 * math.pi))
            entropy_q = 0.5 * d * (1.0 + math.log(2.0 * math.pi)) + float(np.sum(np.log(diag_L)))
            elbo = float(np.mean(log_lik + log_prior) + entropy_q)

            items.append({
                "id": iid,
                "cluster_id": iid,
                "stratum_id": g["stratum_id"],
                "value": elbo,
            })

        by_st = {}
        for it in items:
            by_st.setdefault(it["stratum_id"], []).append(it["value"])
        eq_elbo = float(np.mean([np.mean(v) for v in by_st.values()]))
        return {
            "candidate_id": cid,
            "metrics": {
                "elbo": eq_elbo,
                "eval_wall_clock_sec": float(iso["guard_wall_clock_sec"]),
            },
            "items": items,
        }

    rules_search = {
        "primary_metric": "elbo",
        "direction": "maximize",
        "aggregation_weighting": "equal_stratum",
        "decision_stage": "search",
        "min_improvement": 0.015,
        "max_ratio_vs_baseline": {"eval_wall_clock_sec": 2.2},
    }
    rules_confirm = {
        "primary_metric": "elbo",
        "direction": "maximize",
        "aggregation_weighting": "equal_stratum",
        "expected_strata": ["acs_logit", "compas_logit"],
        "decision_stage": "confirm",
        "alpha_one_sided": 0.025,
        "max_confirm_looks": 2,
        "min_improvement": 0.015,
        "min_confirm_clusters": 12,
        "max_ratio_vs_baseline": {"eval_wall_clock_sec": 2.2},
        "required_comparators": [
            "d8_paper_lsvi",
            "d8_comp_laplace_newton",
            "d8_comp_bbvi_reparam_1x",
        ],
    }
    return execute_two_round_tournament(
        bench_id="d8",
        bdir=bdir,
        score_fn=score_d8,
        baseline_id="d8_paper_lsvi",
        comparators=["d8_paper_lsvi", "d8_comp_laplace_newton", "d8_comp_bbvi_reparam_1x"],
        autosota_audit_id="d8_autosota_bbvi_8seed_long",
        round1_candidates=r1_map,
        round2_candidates=r2_map,
        rules_search=rules_search,
        rules_confirm=rules_confirm,
        phase=phase,
    )


# ============================================================================
# D9: Paper #2 PINet — Hard-Constrained Equality + Box QP Projection
# ============================================================================
def exact_qp_project(A: np.ndarray, b: np.ndarray, lo: np.ndarray, hi: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """High-precision 800-iteration over-relaxed Dykstra/ADMM reference solver for argmin_{Az=b, lo<=z<=hi} 0.5||z-y||^2."""
    # Y: [B, d]
    m, d = A.shape
    P_aff_mat = np.eye(d) - A.T @ np.linalg.solve(A @ A.T + 1e-12 * np.eye(m), A)
    p_aff_shift = A.T @ np.linalg.solve(A @ A.T + 1e-12 * np.eye(m), b)  # [d]

    def proj_aff(Z: np.ndarray) -> np.ndarray:
        return Z @ P_aff_mat.T + p_aff_shift[None, :]

    # Exact Boyle-Dykstra alternating projection onto Affine cap Box!
    X = Y.copy()
    P_inc = np.zeros_like(Y)
    Q_inc = np.zeros_like(Y)
    for _ in range(800):
        Y_mid = proj_aff(X + P_inc)
        P_inc = X + P_inc - Y_mid
        X_new = np.clip(Y_mid + Q_inc, lo[None, :], hi[None, :])
        Q_inc = Y_mid + Q_inc - X_new
        if np.max(np.abs(X_new - X)) < 1e-12:
            X = X_new
            break
        X = X_new
    return X


def prepare_and_score_d9(phase: str, r1_map: dict[str, str], r2_map: dict[str, str]) -> dict:
    bdir = W4_ROOT / "tracks" / "bench_d9_pinet_qp"
    imm = bdir / "immutable_eval"
    runs = bdir / "runs"

    if phase == "round1":
        imm.mkdir(parents=True, exist_ok=True)
        runs.mkdir(parents=True, exist_ok=True)
        regimes = [
            ("well_cond", 48, 14, 1.0, 1.2),
            ("ill_cond", 48, 16, 12.0, 1.0),
            ("tight_box", 64, 20, 4.0, 0.45),
            ("high_coupled", 64, 28, 6.0, 0.8),
        ]
        for st_idx, st in enumerate(["search", "confirm", "holdout"]):
            items_in = []
            items_gt = []
            for reg_idx, (rname, d, m, cond_exp, box_half) in enumerate(regimes):
                for k in range(4):
                    rng = np.random.RandomState(2000 + st_idx * 100 + reg_idx * 10 + k)
                    U_mat, _ = np.linalg.qr(rng.normal(0, 1, (m, m)))
                    V_mat, _ = np.linalg.qr(rng.normal(0, 1, (d, m)))
                    svals = np.geomspace(1.0, cond_exp, m)
                    A = (U_mat * svals[None, :]) @ V_mat.T  # [m, d]
                    lo = -np.full(d, box_half, dtype=np.float64)
                    hi = np.full(d, box_half, dtype=np.float64)
                    # Guarantee strict interior Slater point z_feas in (lo, hi) so Az = b cap [lo, hi] is non-empty!
                    z_feas = rng.uniform(0.55 * lo, 0.55 * hi)
                    b_vec = A @ z_feas
                    # Unprojected neural activations Y [48, d] strongly violating both box & affine constraints
                    Y_batch = rng.normal(0.0, 1.6 * box_half, size=(48, d))
                    Z_star = exact_qp_project(A, b_vec, lo, hi, Y_batch)

                    iid = f"{st}_{rname}_{k:02d}"
                    items_in.append({
                        "id": iid,
                        "d": d,
                        "m": m,
                        "A": np.round(A, 8).tolist(),
                        "b": np.round(b_vec, 8).tolist(),
                        "lo": lo.tolist(),
                        "hi": hi.tolist(),
                        "Y": np.round(Y_batch, 8).tolist(),
                    })
                    items_gt.append({
                        "id": iid,
                        "stratum_id": rname,
                        "A": np.round(A, 8).tolist(),
                        "b": np.round(b_vec, 8).tolist(),
                        "lo": lo.tolist(),
                        "hi": hi.tolist(),
                        "Z_star": Z_star.tolist(),
                    })

            write_json(runs / f"input_{st}.json", {"items": items_in})
            write_json(imm / f"gt_{st}.json", items_gt)
            write_json(imm / f"ids_{st}.json", [x["id"] for x in items_gt])

    def score_d9(cid: str, stage: str) -> dict:
        c_py = bdir / "candidates" / f"{cid}.py"
        in_p = runs / f"input_{stage}.json"
        out_p = runs / f"preds_{stage}_{cid}.json"
        ids_p = imm / f"ids_{stage}.json"
        iso = run_isolated_predict(c_py, in_p, out_p, ids_p, imm, timeout_sec=240)
        preds_map = {r["id"]: np.asarray(r["prediction"], dtype=np.float64) for r in read_json(out_p)["predictions"]}
        gt_list = read_json(imm / f"gt_{stage}.json")

        items = []
        max_feas_all = 0.0
        for g in gt_list:
            iid = g["id"]
            A = np.asarray(g["A"], dtype=np.float64)
            b_vec = np.asarray(g["b"], dtype=np.float64)
            lo = np.asarray(g["lo"], dtype=np.float64)
            hi = np.asarray(g["hi"], dtype=np.float64)
            Z_star = np.asarray(g["Z_star"], dtype=np.float64)
            Z_hat = preds_map[iid]

            eq_err = float(np.max(np.abs(Z_hat @ A.T - b_vec[None, :])))
            box_err = float(max(np.max(np.maximum(0.0, lo[None, :] - Z_hat)), np.max(np.maximum(0.0, Z_hat - hi[None, :]))))
            feas_err = max(eq_err, box_err)
            max_feas_all = max(max_feas_all, feas_err)
            rmse_opt = float(np.sqrt(np.mean((Z_hat - Z_star) ** 2)))

            items.append({
                "id": iid,
                "cluster_id": iid,
                "stratum_id": g["stratum_id"],
                "value": rmse_opt,
                "aux": {"feasibility_err": feas_err},
            })

        by_st = {}
        for it in items:
            by_st.setdefault(it["stratum_id"], []).append(it["value"])
        eq_rmse = float(np.mean([np.mean(v) for v in by_st.values()]))
        return {
            "candidate_id": cid,
            "metrics": {
                "opt_rmse": eq_rmse,
                "max_feasibility_err": max_feas_all,
                "eval_wall_clock_sec": float(iso["guard_wall_clock_sec"]),
            },
            "items": items,
        }

    rules_search = {
        "primary_metric": "opt_rmse",
        "direction": "minimize",
        "aggregation_weighting": "equal_stratum",
        "decision_stage": "search",
        "min_improvement": 0.0005,
        "hard_ceilings": {"max_feasibility_err": 1e-4},
        "per_stratum_hard_ceilings": {"feasibility_err": 1e-4},
        "max_ratio_vs_baseline": {"eval_wall_clock_sec": 1.25},
    }
    rules_confirm = {
        "primary_metric": "opt_rmse",
        "direction": "minimize",
        "aggregation_weighting": "equal_stratum",
        "expected_strata": ["high_coupled", "ill_cond", "tight_box", "well_cond"],
        "decision_stage": "confirm",
        "alpha_one_sided": 0.025,
        "max_confirm_looks": 2,
        "min_improvement": 0.0005,
        "min_confirm_clusters": 12,
        "hard_ceilings": {"max_feasibility_err": 1e-4},
        "per_stratum_hard_ceilings": {"feasibility_err": 1e-4},
        "max_ratio_vs_baseline": {"eval_wall_clock_sec": 1.25},
        "required_comparators": [
            "d9_paper_pinet_dr50",
            "d9_comp_overrelaxed_dr50",
            "d9_comp_dykstra50",
        ],
    }
    return execute_two_round_tournament(
        bench_id="d9",
        bdir=bdir,
        score_fn=score_d9,
        baseline_id="d9_paper_pinet_dr50",
        comparators=["d9_paper_pinet_dr50", "d9_comp_overrelaxed_dr50", "d9_comp_dykstra50"],
        autosota_audit_id="d9_autosota_pinet_dr10_infeasible",
        round1_candidates=r1_map,
        round2_candidates=r2_map,
        rules_search=rules_search,
        rules_confirm=rules_confirm,
        phase=phase,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["round1", "round2_final"], required=True)
    ap.add_argument("--spec-json", required=True)
    args = ap.parse_args()
    spec = read_json(Path(args.spec_json))
    for bid, fn in [("d7", prepare_and_score_d7), ("d8", prepare_and_score_d8), ("d9", prepare_and_score_d9)]:
        if bid in spec:
            print(f"=== [Node-2 {args.phase}] Running {bid.upper()} ===", flush=True)
            res = fn(args.phase, spec[bid]["r1"], spec[bid].get("r2", {}))
            print(f"=== [Node-2 {args.phase}] Done {bid.upper()}: {res.get('round1_incumbent_id') or res.get('selected_finalist')} ===", flush=True)


if __name__ == "__main__":
    main()
