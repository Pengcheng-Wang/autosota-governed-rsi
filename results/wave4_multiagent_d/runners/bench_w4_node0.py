#!/usr/bin/env python3
"""
Wave-4 Multi-Agent Governed RSI Harness — Node 0 (8 x A100-80GB):
  - D1: Paper #48 shapiq (Order-2 k-SII Shapley Interaction Estimation under Hard-Capped budget=200)
  - D2: Paper #80 LatentScoreReweight (Real ACS_income_AZ Worst-Group Classification)
  - D3: Paper #77 CMNN (Certified 0.0%-Violation Partial-Order Monotonic Learning on auto/heart/compas/blog)
All baselines and multi-agent candidates run out-of-process via governed_loop_guard.py run-isolated (python -I -B).
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from harness_common import (
    W4_ROOT,
    execute_two_round_tournament,
    read_json,
    run_guard,
    run_isolated_predict,
    write_json,
)


# ============================================================================
# D1: Paper #48 shapiq — Order-2 k-SII under Hard-Capped budget=200
# ============================================================================
def exact_ksii_order2(v_table: np.ndarray, n: int = 10) -> np.ndarray:
    """
    Exact order-2 k-SII vector of length n + n*(n-1)//2 (= 55 for n=10) from full 2^n table.
    Entries 0..n-1: k-SII main effects phi^(2)_i = I_i - 0.5 * sum_{j != i} I_{ij}.
    Entries n..54: pairwise Shapley interaction indices I_{ij} (0 <= i < j < n).
    """
    N_masks = 1 << n
    masks = np.arange(N_masks, dtype=np.int64)
    bits = ((masks[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int64)
    sizes = np.sum(bits, axis=1)
    fact = np.array([math.factorial(k) for k in range(n + 1)], dtype=np.float64)

    # Exact Shapley values I_i
    shap_1 = np.zeros(n, dtype=np.float64)
    for i in range(n):
        without_i = bits[:, i] == 0
        m_sub = masks[without_i]
        s_sub = sizes[without_i]
        w1 = fact[s_sub] * fact[n - s_sub - 1] / fact[n]
        diffs = v_table[m_sub | (1 << i)] - v_table[m_sub]
        shap_1[i] = float(np.sum(w1 * diffs))

    # Exact pairwise SII I_{ij}
    pairs = []
    pair_mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            both0 = (bits[:, i] == 0) & (bits[:, j] == 0)
            m00 = masks[both0]
            s00 = sizes[both0]
            w2 = fact[s00] * fact[n - s00 - 2] / fact[n - 1]
            delta_ij = (
                v_table[m00 | (1 << i) | (1 << j)]
                - v_table[m00 | (1 << i)]
                - v_table[m00 | (1 << j)]
                + v_table[m00]
            )
            val_ij = float(np.sum(w2 * delta_ij))
            pair_mat[i, j] = pair_mat[j, i] = val_ij
            pairs.append(val_ij)

    ksii_main = shap_1 - 0.5 * np.sum(pair_mat, axis=1)
    return np.concatenate([ksii_main, np.array(pairs, dtype=np.float64)])


def build_d1_games() -> dict:
    """Build 48 real + synthetic n=10 cooperative games (16 per split across 4 families)."""
    data_dir = W4_ROOT / "data" / "shapiq"
    n = 10
    N_masks = 1 << n
    bits = ((np.arange(N_masks)[:, None] >> np.arange(n)[None, :]) & 1).astype(np.float64)

    # Fit fast real-data coalition models on bundled California Housing, Adult Census, and Bike CSVs
    cali_df = pd.read_csv(data_dir / "california_housing.csv")
    Xc_raw = StandardScaler().fit_transform(cali_df.iloc[:, :8].values.astype(np.float64))
    # Augment 2 real geographic cross-features to reach n=10 players
    Xc = np.hstack([Xc_raw, (Xc_raw[:, 6:7] * Xc_raw[:, 7:8]), (Xc_raw[:, 0:1] * Xc_raw[:, 5:6])])
    yc = cali_df.iloc[:, 8].values.astype(np.float64)
    cali_model = HistGradientBoostingRegressor(max_depth=5, max_iter=120, random_state=42).fit(Xc[:12000], yc[:12000])
    Xc_bg = np.mean(Xc[:2000], axis=0)
    Xc_pool = Xc[12000:15000]

    bike_df = pd.read_csv(data_dir / "bike.csv")
    bike_cols = ["season", "year", "month", "hour", "holiday", "weekday", "workingday", "temp", "humidity", "windspeed"]
    Xb_parts = []
    for bc in bike_cols:
        if pd.api.types.is_numeric_dtype(bike_df[bc]):
            Xb_parts.append(bike_df[[bc]].values.astype(np.float64))
        else:
            Xb_parts.append(OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit_transform(bike_df[[bc]].astype(str)))
    Xb = StandardScaler().fit_transform(np.hstack(Xb_parts))
    yb = np.log1p(np.maximum(0.0, bike_df["count"].values.astype(np.float64)))
    bike_model = HistGradientBoostingRegressor(max_depth=5, max_iter=120, random_state=42).fit(Xb[:12000], yb[:12000])
    Xb_bg = np.mean(Xb[:2000], axis=0)
    Xb_pool = Xb[12000:15000]

    adult_df = pd.read_csv(data_dir / "adult_census.csv")
    ya = (adult_df["class"].astype(str).str.contains(">50K")).astype(np.float64).values
    cat_cols = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex"]
    num_cols = ["age", "capital-gain", "hours-per-week"]
    Xa_cat = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit_transform(adult_df[cat_cols].astype(str))
    Xa_num = StandardScaler().fit_transform(adult_df[num_cols].values.astype(np.float64))
    Xa = np.hstack([Xa_num, Xa_cat])
    adult_model = HistGradientBoostingClassifier(max_depth=5, max_iter=120, random_state=42).fit(Xa[:25000], ya[:25000])
    Xa_bg = np.median(Xa[:3000], axis=0)
    Xa_pool = Xa[25000:29000]

    splits_out = {}
    for stage, base_seed in [("search", 100), ("confirm", 200), ("holdout", 300)]:
        games = []
        for fam_idx, fam in enumerate(["soum", "california", "bike", "adult"]):
            for k in range(4):
                rng = np.random.RandomState(base_seed + fam_idx * 10 + k)
                gid = f"{stage}_{fam}_{k:02d}"
                if fam == "soum":
                    # Sparse Synergy-of-Unanimity-Models game (Paper #48 exact benchmark family)
                    n_terms = 35
                    v_tab = np.zeros(N_masks, dtype=np.float64)
                    for _ in range(n_terms):
                        sz = int(rng.choice([1, 2, 3, 4], p=[0.20, 0.45, 0.25, 0.10]))
                        players = rng.choice(n, size=sz, replace=False)
                        req_mask = int(sum(1 << int(p) for p in players))
                        coef = float(rng.normal(0.0, 1.5 / sz))
                        active = ((np.arange(N_masks) & req_mask) == req_mask).astype(np.float64)
                        v_tab += coef * active
                elif fam == "california":
                    x_inst = Xc_pool[int(rng.randint(len(Xc_pool)))]
                    X_coal = bits * x_inst[None, :] + (1.0 - bits) * Xc_bg[None, :]
                    v_tab = cali_model.predict(X_coal).astype(np.float64)
                elif fam == "bike":
                    x_inst = Xb_pool[int(rng.randint(len(Xb_pool)))]
                    X_coal = bits * x_inst[None, :] + (1.0 - bits) * Xb_bg[None, :]
                    v_tab = bike_model.predict(X_coal).astype(np.float64)
                elif fam == "adult":
                    x_inst = Xa_pool[int(rng.randint(len(Xa_pool)))]
                    X_coal = bits * x_inst[None, :] + (1.0 - bits) * Xa_bg[None, :]
                    probs = np.clip(adult_model.predict_proba(X_coal)[:, 1], 1e-5, 1.0 - 1e-5)
                    v_tab = np.log(probs / (1.0 - probs)).astype(np.float64)
                v_tab = v_tab - v_tab[0]
                gt_ksii = exact_ksii_order2(v_tab, n=n)
                games.append({
                    "id": gid,
                    "stratum_id": fam,
                    "n": n,
                    "v_table": v_tab.tolist(),
                    "gt_ksii": gt_ksii.tolist(),
                })
        splits_out[stage] = games
    return splits_out


def prepare_and_score_d1(phase: str, r1_map: dict[str, str], r2_map: dict[str, str]) -> dict:
    bdir = W4_ROOT / "tracks" / "bench_d1_shapiq"
    imm = bdir / "immutable_eval"
    runs = bdir / "runs"
    if phase == "round1":
        imm.mkdir(parents=True, exist_ok=True)
        runs.mkdir(parents=True, exist_ok=True)
        games_by_stage = build_d1_games()
        for st, glist in games_by_stage.items():
            write_json(imm / f"games_oracle_{st}.json", glist)
            write_json(imm / f"ids_{st}.json", [g["id"] for g in glist])

    def score_d1(cid: str, stage: str) -> dict:
        c_py = bdir / "candidates" / f"{cid}.py"
        oracle_games = read_json(imm / f"games_oracle_{stage}.json")
        ids_p = imm / f"ids_{stage}.json"
        t0 = time.perf_counter()

        # Pass 1: candidate proposes batch-1 coalition masks (NO access to v_table!)
        req1_p = runs / f"d1_req1_{stage}_{cid}.json"
        out1_p = runs / f"d1_out1_{stage}_{cid}.json"
        write_json(req1_p, {
            "step": "propose_b1",
            "items": [{"id": g["id"], "n": g["n"], "max_budget": 200} for g in oracle_games],
        })
        iso1 = run_isolated_predict(c_py, req1_p, out1_p, ids_p, imm, timeout_sec=180)
        resp1 = {r["id"]: r["prediction"] for r in read_json(out1_p)["predictions"]}

        # Host oracle evaluates batch-1 queries (mechanically tracking total distinct queries!)
        obs_by_id = {}
        max_queries_used = 0
        need_b2 = False
        for g in oracle_games:
            gid = g["id"]
            v_tab = g["v_table"]
            q_raw = resp1[gid]
            q_masks = [int(m) & 1023 for m in q_raw["queries_b1"]]
            # Always ensure empty (0) and grand (1023) are counted inside budget
            seen = {}
            for m in [0, 1023, *q_masks]:
                if m not in seen:
                    seen[m] = float(v_tab[m])
            obs_by_id[gid] = seen
            if q_raw.get("want_b2", False):
                need_b2 = True

        if need_b2:
            req2_p = runs / f"d1_req2_{stage}_{cid}.json"
            out2_p = runs / f"d1_out2_{stage}_{cid}.json"
            write_json(req2_p, {
                "step": "propose_b2",
                "items": [
                    {
                        "id": g["id"],
                        "n": g["n"],
                        "observed": [[int(m), val] for m, val in obs_by_id[g["id"]].items()],
                        "remaining_budget": max(0, 200 - len(obs_by_id[g["id"]])),
                    }
                    for g in oracle_games
                ],
            })
            iso2 = run_isolated_predict(c_py, req2_p, out2_p, ids_p, imm, timeout_sec=180)
            resp2 = {r["id"]: r["prediction"] for r in read_json(out2_p)["predictions"]}
            for g in oracle_games:
                gid = g["id"]
                v_tab = g["v_table"]
                seen = obs_by_id[gid]
                for m in resp2[gid].get("queries_b2", []):
                    mi = int(m) & 1023
                    if mi not in seen:
                        seen[mi] = float(v_tab[mi])
        else:
            iso2 = {"guard_wall_clock_sec": 0.0}

        # Pass 3: candidate estimates the 55 order-2 k-SII values from observed (mask, v(mask)) ONLY!
        # Note: if an ungoverned method like AutoSOTA N_ENSEMBLE=10 requested >200 queries, we record
        # the exact query count so hard_ceilings: {"max_coalition_evals": 200.0} catches it!
        req3_items = []
        for g in oracle_games:
            gid = g["id"]
            seen = obs_by_id[gid]
            max_queries_used = max(max_queries_used, len(seen))
            # If candidate stayed <=200, pass all seen; if >200 (audit case), pass seen so audit measures query breach
            req3_items.append({
                "id": gid,
                "n": g["n"],
                "observed": [[int(m), val] for m, val in seen.items()],
            })
        req3_p = runs / f"d1_req3_{stage}_{cid}.json"
        out3_p = runs / f"d1_out3_{stage}_{cid}.json"
        write_json(req3_p, {"step": "estimate", "items": req3_items})
        iso3 = run_isolated_predict(c_py, req3_p, out3_p, ids_p, imm, timeout_sec=180)
        ests = {r["id"]: np.asarray(r["prediction"], dtype=np.float64) for r in read_json(out3_p)["predictions"]}

        wall = float(iso1["guard_wall_clock_sec"]) + float(iso2["guard_wall_clock_sec"]) + float(iso3["guard_wall_clock_sec"])
        items = []
        mse_all = []
        for g in oracle_games:
            gid = g["id"]
            gt = np.asarray(g["gt_ksii"], dtype=np.float64)
            est = ests[gid]
            if est.shape != gt.shape or not np.all(np.isfinite(est)):
                raise RuntimeError(f"Invalid k-SII estimate shape/NaN for {cid} on {gid}")
            top10_gt = set(np.argsort(-np.abs(gt))[:10].tolist())
            top10_est = set(np.argsort(-np.abs(est))[:10].tolist())
            p10 = len(top10_gt & top10_est) / 10.0
            mse = float(np.mean((est - gt) ** 2))
            mse_all.append(mse)
            q_cnt = float(len(obs_by_id[gid]))
            items.append({
                "id": gid,
                "cluster_id": gid,
                "stratum_id": g["stratum_id"],
                "value": p10,
                "aux": {"coalition_evals": q_cnt, "ksii_mse": mse},
            })

        return {
            "candidate_id": cid,
            "metrics": {
                "precision_at_10": float(np.mean([it["value"] for it in items])),
                "ksii_mse": float(np.mean(mse_all)),
                "max_coalition_evals": float(max_queries_used),
                "eval_wall_clock_sec": wall,
            },
            "items": items,
        }

    rules_search = {
        "primary_metric": "precision_at_10",
        "direction": "maximize",
        "aggregation_weighting": "equal_stratum",
        "decision_stage": "search",
        "min_improvement": 0.01,
        "hard_ceilings": {"max_coalition_evals": 200.0},
        "per_stratum_hard_ceilings": {"coalition_evals": 200.0},
        "max_ratio_vs_baseline": {"eval_wall_clock_sec": 2.5},
    }
    rules_confirm = {
        "primary_metric": "precision_at_10",
        "direction": "maximize",
        "aggregation_weighting": "equal_stratum",
        "expected_strata": ["adult", "bike", "california", "soum"],
        "decision_stage": "confirm",
        "alpha_one_sided": 0.025,
        "max_confirm_looks": 2,
        "min_improvement": 0.01,
        "min_confirm_clusters": 12,
        "hard_ceilings": {"max_coalition_evals": 200.0},
        "per_stratum_hard_ceilings": {"coalition_evals": 200.0},
        "max_ratio_vs_baseline": {"eval_wall_clock_sec": 2.5},
        "required_comparators": [
            "d1_base_mc_antithetic_200",
            "d1_paper_svarmiq_200",
            "d1_autosota_stratsize_1x_200",
        ],
    }
    return execute_two_round_tournament(
        bench_id="d1",
        bdir=bdir,
        score_fn=score_d1,
        baseline_id="d1_base_mc_antithetic_200",
        comparators=["d1_base_mc_antithetic_200", "d1_paper_svarmiq_200", "d1_autosota_stratsize_1x_200"],
        autosota_audit_id="d1_autosota_ensemble10_cheat",
        round1_candidates=r1_map,
        round2_candidates=r2_map,
        rules_search=rules_search,
        rules_confirm=rules_confirm,
        phase=phase,
    )


# ============================================================================
# D2: Paper #80 LatentScoreReweight — Real ACS_income_AZ Worst-Group Accuracy
# ============================================================================
def prepare_and_score_d2(phase: str, r1_map: dict[str, str], r2_map: dict[str, str]) -> dict:
    bdir = W4_ROOT / "tracks" / "bench_d2_acs_reweight"
    imm = bdir / "immutable_eval"
    runs = bdir / "runs"

    if phase == "round1":
        imm.mkdir(parents=True, exist_ok=True)
        runs.mkdir(parents=True, exist_ok=True)
        acs_dir = W4_ROOT / "data" / "acs"
        real_df = pd.read_csv(acs_dir / "real.csv")
        test_df = pd.read_csv(acs_dir / "test.csv")
        w_df = pd.read_csv(acs_dir / "weight.csv")
        weights_all = w_df["weight"].values.astype(np.float64)

        feat_cols = [c for c in real_df.columns if c != "PINCP"]
        sex_idx = feat_cols.index("SEX")
        race_idx = feat_cols.index("race")

        rng = np.random.RandomState(20261118)
        perm = rng.permutation(len(real_df))
        tr_idx = perm[:14000]
        rest_real = real_df.iloc[perm[14000:]].copy()
        pool_eval = pd.concat([rest_real, test_df], ignore_index=True)
        perm_ev = rng.permutation(len(pool_eval))
        pool_eval = pool_eval.iloc[perm_ev].reset_index(drop=True)

        X_tr = real_df.iloc[tr_idx][feat_cols].values.astype(np.float64)
        y_tr = real_df.iloc[tr_idx]["PINCP"].values.astype(np.int64)
        w_tr = weights_all[tr_idx]
        # 4 demographic groups: (SEX==2) * 2 + (race > 0)
        g_tr = ((X_tr[:, sex_idx] > 1.5).astype(np.int64) * 2 + (X_tr[:, race_idx] > 0.5).astype(np.int64))

        write_json(runs / "train_bundle.json", {
            "feat_cols": feat_cols,
            "sex_idx": sex_idx,
            "race_idx": race_idx,
            "X_train": np.round(X_tr, 5).tolist(),
            "y_train": y_tr.tolist(),
            "g_train": g_tr.tolist(),
            "w_score": np.round(w_tr, 6).tolist(),
        })

        # 3 disjoint splits of 16 blocks each (380 rows/block -> ~18,240 eval rows!)
        slices = {"search": (0, 6080), "confirm": (6080, 12160), "holdout": (12160, 18240)}
        for st, (lo, hi) in slices.items():
            sub = pool_eval.iloc[lo:hi].reset_index(drop=True)
            X_ev = sub[feat_cols].values.astype(np.float64)
            y_ev = sub["PINCP"].values.astype(np.int64)
            g_ev = ((X_ev[:, sex_idx] > 1.5).astype(np.int64) * 2 + (X_ev[:, race_idx] > 0.5).astype(np.int64))
            # Class-conditional x demographic groups = 8 subpopulation groups (g * 2 + y) as in GroupDRO/Paper #80!
            g8_ev = g_ev * 2 + y_ev
            blocks_feat = []
            blocks_gt = []
            for b in range(16):
                b_lo, b_hi = b * 380, (b + 1) * 380
                bid = f"acs_{st}_b{b:02d}"
                blocks_feat.append({"id": bid, "X_eval": np.round(X_ev[b_lo:b_hi], 5).tolist()})
                blocks_gt.append({"id": bid, "y_eval": y_ev[b_lo:b_hi].tolist(), "g8_eval": g8_ev[b_lo:b_hi].tolist()})
            write_json(runs / f"eval_features_{st}.json", {
                "train_bundle_path": str(runs / "train_bundle.json"),
                "items": blocks_feat,
            })
            write_json(imm / f"gt_{st}.json", blocks_gt)
            write_json(imm / f"ids_{st}.json", [x["id"] for x in blocks_gt])

    def score_d2(cid: str, stage: str) -> dict:
        c_py = bdir / "candidates" / f"{cid}.py"
        in_p = runs / f"eval_features_{stage}.json"
        out_p = runs / f"preds_{stage}_{cid}.json"
        ids_p = imm / f"ids_{stage}.json"
        iso = run_isolated_predict(c_py, in_p, out_p, ids_p, imm, timeout_sec=240)
        preds_map = {r["id"]: np.asarray(r["prediction"], dtype=np.int64) for r in read_json(out_p)["predictions"]}
        gt_blocks = read_json(imm / f"gt_{stage}.json")

        items = []
        all_correct = []
        for blk in gt_blocks:
            bid = blk["id"]
            yb = np.asarray(blk["y_eval"], dtype=np.int64)
            gb = np.asarray(blk["g8_eval"], dtype=np.int64)
            pb = preds_map[bid]
            corr = (pb == yb).astype(np.float64)
            all_correct.append(float(np.mean(corr)))
            grp_accs = []
            for g_val in range(8):
                m = gb == g_val
                if np.sum(m) >= 3:
                    grp_accs.append(float(np.mean(corr[m])))
            wg_acc = min(grp_accs)
            items.append({
                "id": bid,
                "cluster_id": bid,
                "stratum_id": "acs_az",
                "value": wg_acc,
                "aux": {"mean_acc": float(np.mean(corr))},
            })

        return {
            "candidate_id": cid,
            "metrics": {
                "worst_group_acc": float(np.mean([it["value"] for it in items])),
                "mean_acc": float(np.mean(all_correct)),
                "eval_wall_clock_sec": float(iso["guard_wall_clock_sec"]),
            },
            "items": items,
        }

    rules_search = {
        "primary_metric": "worst_group_acc",
        "direction": "maximize",
        "decision_stage": "search",
        "min_improvement": 0.005,
        "hard_floors": {"mean_acc": 0.74},
        "max_ratio_vs_baseline": {"eval_wall_clock_sec": 2.5},
    }
    rules_confirm = {
        "primary_metric": "worst_group_acc",
        "direction": "maximize",
        "decision_stage": "confirm",
        "alpha_one_sided": 0.025,
        "max_confirm_looks": 2,
        "min_improvement": 0.005,
        "min_confirm_clusters": 12,
        "hard_floors": {"mean_acc": 0.74},
        "per_stratum_hard_floors": {"mean_acc": 0.74},
        "max_ratio_vs_baseline": {"eval_wall_clock_sec": 2.5},
        "required_comparators": [
            "d2_base_erm_mlp",
            "d2_paper_latent_score_reweight",
            "d2_comp_group_dro",
        ],
    }
    return execute_two_round_tournament(
        bench_id="d2",
        bdir=bdir,
        score_fn=score_d2,
        baseline_id="d2_base_erm_mlp",
        comparators=["d2_base_erm_mlp", "d2_paper_latent_score_reweight", "d2_comp_group_dro"],
        autosota_audit_id="d2_paper_latent_score_reweight",
        round1_candidates=r1_map,
        round2_candidates=r2_map,
        rules_search=rules_search,
        rules_confirm=rules_confirm,
        phase=phase,
    )


# ============================================================================
# D3: Paper #77 CMNN — Certified 0.0%-Violation Monotonic Tabular Learning
# ============================================================================
def prepare_and_score_d3(phase: str, r1_map: dict[str, str], r2_map: dict[str, str]) -> dict:
    bdir = W4_ROOT / "tracks" / "bench_d3_cmnn"
    imm = bdir / "immutable_eval"
    runs = bdir / "runs"

    if phase == "round1":
        imm.mkdir(parents=True, exist_ok=True)
        runs.mkdir(parents=True, exist_ok=True)
        cdir_data = W4_ROOT / "data" / "cmnn"

        def load_ds(name: str):
            if name == "blog":
                arr = pd.read_csv(cdir_data / "test_blog.csv", header=None).values.astype(np.float64)
                mask = np.array([1 if i in range(51, 55) or i in range(56, 60) else 0 for i in range(280)], dtype=np.int64)
                n_blocks = 4
            elif name == "compas":
                a1 = pd.read_csv(cdir_data / "train_compas.csv", header=None).values.astype(np.float64)
                a2 = pd.read_csv(cdir_data / "test_compas.csv", header=None).values.astype(np.float64)
                arr = np.vstack([a1, a2])
                mask = np.array([1, 1, 1, 1] + [0] * 9, dtype=np.int64)
                n_blocks = 4
            elif name == "auto":
                a1 = pd.read_csv(cdir_data / "train_auto.csv", header=None).values.astype(np.float64)
                a2 = pd.read_csv(cdir_data / "test_auto.csv", header=None).values.astype(np.float64)
                arr = np.vstack([a1, a2])
                mask = np.array([0, -1, -1, -1, 0, 0, 0], dtype=np.int64)
                n_blocks = 2
            elif name == "heart":
                a1 = pd.read_csv(cdir_data / "train_heart.csv", header=None).values.astype(np.float64)
                a2 = pd.read_csv(cdir_data / "test_heart.csv", header=None).values.astype(np.float64)
                arr = np.vstack([a1, a2])
                mask = np.array([0] * 3 + [1] * 2 + [0] * 8, dtype=np.int64)
                n_blocks = 2
            else:
                raise ValueError(name)
            return arr[:, :-1], arr[:, -1], mask, n_blocks

        rng = np.random.RandomState(772026)
        stage_inputs = {"search": [], "confirm": [], "holdout": []}
        stage_gt = {"search": [], "confirm": [], "holdout": []}
        train_shared = {}

        for ds_name in ["auto", "heart", "compas", "blog"]:
            X_all, y_all, mask_mono, n_blk = load_ds(ds_name)
            perm = rng.permutation(len(X_all))
            X_all, y_all = X_all[perm], y_all[perm]
            n_tot = len(X_all)
            n_tr = int(0.46 * n_tot)
            n_s = int(0.18 * n_tot)
            n_c = int(0.18 * n_tot)

            X_tr_raw, y_tr_raw = X_all[:n_tr], y_all[:n_tr]
            xm = np.mean(X_tr_raw, axis=0)
            xs = np.where(np.std(X_tr_raw, axis=0) > 1e-6, np.std(X_tr_raw, axis=0), 1.0)
            ym = float(np.mean(y_tr_raw))
            ys = float(np.std(y_tr_raw)) if np.std(y_tr_raw) > 1e-6 else 1.0

            X_tr = (X_tr_raw - xm) / xs
            y_tr = (y_tr_raw - ym) / ys

            mono_idx = np.where(mask_mono != 0)[0]
            free_idx = np.where(mask_mono == 0)[0]
            mono_signs = mask_mono[mono_idx].astype(np.float64)

            X_tr_free = X_tr[:, free_idx]
            X_tr_mono = X_tr[:, mono_idx] * mono_signs[None, :]
            train_shared[ds_name] = {
                "X_train_free": np.round(X_tr_free, 6).tolist(),
                "X_train_mono": np.round(X_tr_mono, 6).tolist(),
                "y_train": np.round(y_tr, 6).tolist(),
            }

            split_slices = {
                "search": (n_tr, n_tr + n_s),
                "confirm": (n_tr + n_s, n_tr + n_s + n_c),
                "holdout": (n_tr + n_s + n_c, n_tot),
            }
            for st, (lo, hi) in split_slices.items():
                X_ev = (X_all[lo:hi] - xm) / xs
                y_ev = (y_all[lo:hi] - ym) / ys
                blk_indices = np.array_split(np.arange(len(X_ev)), n_blk)
                for b_i, bi in enumerate(blk_indices):
                    Xf_b = X_ev[bi][:, free_idx]
                    Xm_b = X_ev[bi][:, mono_idx] * mono_signs[None, :]
                    yb = y_ev[bi]

                    # Build 64 anonymous paired monotonicity probes (x_low <= x_high on X_mono!)
                    n_probes = min(64, len(Xf_b))
                    p_sel = rng.choice(len(Xf_b), size=n_probes, replace=(len(Xf_b) < n_probes))
                    delta_pos = np.abs(rng.normal(0.0, 0.75, size=(n_probes, len(mono_idx)))) + 0.05
                    Xf_p_lo, Xm_p_lo = Xf_b[p_sel].copy(), Xm_b[p_sel].copy()
                    Xf_p_hi, Xm_p_hi = Xf_b[p_sel].copy(), Xm_b[p_sel] + delta_pos

                    Xf_comb = np.vstack([Xf_b, Xf_p_lo, Xf_p_hi])
                    Xm_comb = np.vstack([Xm_b, Xm_p_lo, Xm_p_hi])
                    shuf = rng.permutation(len(Xf_comb))
                    inv_shuf = np.argsort(shuf)

                    for sd in [0, 1, 2]:
                        item_id = f"{ds_name}_b{b_i:02d}_s{sd}"
                        stage_inputs[st].append({
                            "id": item_id,
                            "dataset": ds_name,
                            "seed": sd,
                            "X_eval_free": np.round(Xf_comb[shuf], 6).tolist(),
                            "X_eval_mono": np.round(Xm_comb[shuf], 6).tolist(),
                        })
                        stage_gt[st].append({
                            "id": item_id,
                            "cluster_id": f"{ds_name}_b{b_i:02d}",
                            "stratum_id": ds_name,
                            "seed_id": f"s{sd}",
                            "n_eval": int(len(yb)),
                            "n_probes": int(n_probes),
                            "inv_shuf": inv_shuf.tolist(),
                            "y_eval": yb.tolist(),
                        })

        train_shared_p = write_json(runs / "train_datasets.json", train_shared)
        for st in ["search", "confirm", "holdout"]:
            write_json(runs / f"inputs_{st}.json", {"train_datasets_path": str(train_shared_p), "items": stage_inputs[st]})
            write_json(imm / f"gt_{st}.json", stage_gt[st])
            write_json(imm / f"ids_{st}.json", [x["id"] for x in stage_gt[st]])

    def score_d3(cid: str, stage: str) -> dict:
        c_py = bdir / "candidates" / f"{cid}.py"
        in_p = runs / f"inputs_{stage}.json"
        out_p = runs / f"preds_{stage}_{cid}.json"
        ids_p = imm / f"ids_{stage}.json"
        iso = run_isolated_predict(c_py, in_p, out_p, ids_p, imm, timeout_sec=360)
        preds_map = {r["id"]: np.asarray(r["prediction"], dtype=np.float64) for r in read_json(out_p)["predictions"]}
        gt_list = read_json(imm / f"gt_{stage}.json")

        items = []
        max_viol = 0.0
        for g in gt_list:
            iid = g["id"]
            unshuf = preds_map[iid][np.asarray(g["inv_shuf"], dtype=np.int64)]
            n_ev, n_pr = g["n_eval"], g["n_probes"]
            pred_ev = unshuf[:n_ev]
            pred_lo = unshuf[n_ev:n_ev + n_pr]
            pred_hi = unshuf[n_ev + n_pr:n_ev + 2 * n_pr]
            viol_rate = float(np.mean((pred_hi + 1e-5) < pred_lo))
            max_viol = max(max_viol, viol_rate)
            yb = np.asarray(g["y_eval"], dtype=np.float64)
            rmse = float(np.sqrt(np.mean((pred_ev - yb) ** 2)))
            items.append({
                "id": iid,
                "cluster_id": g["cluster_id"],
                "stratum_id": g["stratum_id"],
                "seed_id": g["seed_id"],
                "value": rmse,
                "aux": {"monotonicity_viol_rate": viol_rate},
            })

        # Equal-stratum RMSE across the 4 datasets
        by_ds = {}
        for it in items:
            by_ds.setdefault(it["stratum_id"], []).append(it["value"])
        eq_rmse = float(np.mean([np.mean(v) for v in by_ds.values()]))
        return {
            "candidate_id": cid,
            "metrics": {
                "std_rmse": eq_rmse,
                "max_monotonicity_viol_rate": max_viol,
                "eval_wall_clock_sec": float(iso["guard_wall_clock_sec"]),
            },
            "items": items,
        }

    rules_search = {
        "primary_metric": "std_rmse",
        "direction": "minimize",
        "aggregation_weighting": "equal_stratum",
        "decision_stage": "search",
        "min_improvement": 0.004,
        "hard_ceilings": {"max_monotonicity_viol_rate": 0.0},
        "per_stratum_hard_ceilings": {"monotonicity_viol_rate": 0.0},
        "max_ratio_vs_baseline": {"eval_wall_clock_sec": 2.2},
    }
    rules_confirm = {
        "primary_metric": "std_rmse",
        "direction": "minimize",
        "aggregation_weighting": "equal_stratum",
        "expected_strata": ["auto", "blog", "compas", "heart"],
        "decision_stage": "confirm",
        "alpha_one_sided": 0.025,
        "max_confirm_looks": 2,
        "min_improvement": 0.004,
        "min_confirm_clusters": 12,
        "hard_ceilings": {"max_monotonicity_viol_rate": 0.0},
        "per_stratum_hard_ceilings": {"monotonicity_viol_rate": 0.0},
        "max_ratio_vs_baseline": {"eval_wall_clock_sec": 2.2},
        "required_comparators": [
            "d3_paper_cmnn_100ep",
            "d3_autosota_cmnn_150ep",
            "d3_comp_monotone_ridge_spline",
        ],
    }
    return execute_two_round_tournament(
        bench_id="d3",
        bdir=bdir,
        score_fn=score_d3,
        baseline_id="d3_paper_cmnn_100ep",
        comparators=["d3_paper_cmnn_100ep", "d3_autosota_cmnn_150ep", "d3_comp_monotone_ridge_spline"],
        autosota_audit_id="d3_autosota_cmnn_150ep",
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
    for bid, fn in [("d1", prepare_and_score_d1), ("d2", prepare_and_score_d2), ("d3", prepare_and_score_d3)]:
        if bid in spec:
            print(f"=== [Node-0 {args.phase}] Running {bid.upper()} ===", flush=True)
            res = fn(args.phase, spec[bid]["r1"], spec[bid].get("r2", {}))
            print(f"=== [Node-0 {args.phase}] Done {bid.upper()}: {res.get('round1_incumbent_id') or res.get('selected_finalist')} ===", flush=True)


if __name__ == "__main__":
    main()
