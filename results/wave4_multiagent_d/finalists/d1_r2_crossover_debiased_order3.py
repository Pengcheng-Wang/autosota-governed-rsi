import argparse, json, math
import numpy as np

R2_MODE = 'crossover_debiased_order3'
_CACHED_DOPT = {}

def make_design_order2(masks, n, pairs_ij):
    bits = ((masks[:, None] >> np.arange(n)[None, :]) & 1).astype(np.float64)
    zp = np.column_stack([bits[:, i] * bits[:, j] for (i, j) in pairs_ij])
    return np.hstack([bits, zp])

def get_cached_dopt(n, target_len):
    key = (n, target_len)
    if key in _CACHED_DOPT:
        return list(_CACHED_DOPT[key])
    full = (1 << n) - 1
    pairs_ij = [(i, j) for i in range(n) for j in range(i + 1, n)]
    cand_masks = np.arange(1, full, dtype=np.int64)
    Z_all = make_design_order2(cand_masks, n, pairs_ij)
    sizes = np.sum(Z_all[:, :n], axis=1).astype(np.int64)
    sw = np.array([math.sqrt((n - 1) / (math.comb(n, int(s)) * s * (n - s))) for s in sizes])
    Zw = Z_all * sw[:, None]

    chosen_set = {0, full}
    chosen_list = [0, full]
    G_inv = np.eye(55) * 40.0
    for i in range(n):
        for m in (1 << i, full ^ (1 << i)):
            if m not in chosen_set:
                chosen_set.add(m); chosen_list.append(m)
                row = Zw[m - 1]
                G_inv -= np.outer(G_inv @ row, row @ G_inv) / (1.0 + float(row @ G_inv @ row))
    avail = np.ones(len(cand_masks), dtype=bool)
    for m in chosen_list:
        if 0 < m < full:
            avail[m - 1] = False
    while len(chosen_list) < target_len:
        lev = np.sum((Zw @ G_inv) * Zw, axis=1)
        lev[~avail] = -1.0
        m = int(cand_masks[int(np.argmax(lev))])
        for ma in (m, full ^ m):
            if ma not in chosen_set and len(chosen_list) < target_len:
                chosen_set.add(ma); chosen_list.append(ma)
                avail[ma - 1] = False
                row = Zw[ma - 1]
                G_inv -= np.outer(G_inv @ row, row @ G_inv) / (1.0 + float(row @ G_inv @ row))
    _CACHED_DOPT[key] = chosen_list[:target_len]
    return list(_CACHED_DOPT[key])

def fit_wls_exact_ksii(n, observed, debiased_order3=False):
    obs_dict = {int(m): float(v) for m, v in observed}
    full = (1 << n) - 1
    v0 = obs_dict.get(0, 0.0)
    vN = obs_dict.get(full, 0.0)
    masks = np.array([m for m in obs_dict.keys() if m not in (0, full)], dtype=np.int64)
    vals = np.array([obs_dict[int(m)] - v0 for m in masks], dtype=np.float64)
    pairs_ij = [(i, j) for i in range(n) for j in range(i + 1, n)]
    X2 = make_design_order2(masks, n, pairs_ij)
    bits = X2[:, :n]
    sizes = np.sum(bits, axis=1).astype(np.int64)
    w = np.array([120.0 if (s == 1 or s == n - 1) else float((n - 1) / (math.comb(n, int(s)) * s * (n - s))) for s in sizes])

    if not debiased_order3:
        X_aug = np.vstack([X2, np.ones((1, 55))])
        y_aug = np.append(vals, vN - v0)
        w_aug = np.append(w, 1e5)
        XtW = X_aug.T * w_aug[None, :]
        return np.linalg.solve(XtW @ X_aug + np.diag([1e-6] * n + [1.2e-3] * 45), XtW @ y_aug)

    # Crossover R2: Fit Order-2 + all 120 Order-3 Mobius terms via Ridge, then map EXACT Mobius->k-SII(order 2)!
    # Recall for v(S) = sum_{|T|<=3} m_T prod_{i in T} z_i:
    # Exact pairwise SII is I_{ij} = m_{ij} + 0.5 * sum_{k not in {i,j}} m_{ijk}!
    # Exact Shapley value is phi_i = m_i + 0.5 * sum_{j!=i} m_{ij} + (1/3) * sum_{j<k} m_{ijk},
    # so k-SII main effect is phi^(2)_i = phi_i - 0.5 * sum_{j!=i} I_{ij} = m_i + (1/12) * sum_{j<k, i in {i,j,k}} ... wait:
    # phi_i - 0.5 * sum_{j!=i} (m_{ij} + 0.5 * sum_{k} m_{ijk}) = m_i + (1/3 - 2 * 0.25) sum_{j<k != i} m_{ijk} = m_i - (1/6) sum_{j<k != i} m_{ijk}!
    triples = [(i, j, k) for i in range(n) for j in range(i + 1, n) for k in range(j + 1, n)]
    Z3 = np.column_stack([bits[:, i] * bits[:, j] * bits[:, k] for (i, j, k) in triples])  # [Q, 120]
    X3 = np.hstack([X2, Z3])  # [Q, 175]
    X_aug = np.vstack([X3, np.ones((1, X3.shape[1]))])
    y_aug = np.append(vals, vN - v0)
    w_aug = np.append(w, 1e5)
    XtW = X_aug.T * w_aug[None, :]
    reg = np.diag([1e-6] * n + [1.5e-3] * 45 + [2.5e-2] * len(triples))
    m_hat = np.linalg.solve(XtW @ X_aug + reg, XtW @ y_aug)

    m1 = m_hat[:n]
    m2 = m_hat[n:55]
    m3 = m_hat[55:]

    # Map (m1, m2, m3) -> exact k-SII order-2!
    pair_idx_map = {p: idx for idx, p in enumerate(pairs_ij)}
    I2 = m2.copy()
    main_corr = np.zeros(n, dtype=np.float64)
    for t_idx, (i, j, k) in enumerate(triples):
        v3 = m3[t_idx]
        I2[pair_idx_map[(i, j)]] += 0.5 * v3
        I2[pair_idx_map[(i, k)]] += 0.5 * v3
        I2[pair_idx_map[(j, k)]] += 0.5 * v3
        main_corr[i] -= (1.0 / 6.0) * v3
        main_corr[j] -= (1.0 / 6.0) * v3
        main_corr[k] -= (1.0 / 6.0) * v3
    ksii_main = m1 + main_corr
    return np.concatenate([ksii_main, I2])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    req = json.loads(open(args.input_json, encoding="utf-8").read())
    step = req["step"]
    preds = []
    for it in req["items"]:
        gid = it["id"]
        n = int(it["n"])
        full = (1 << n) - 1
        if step == "propose_b1":
            if R2_MODE == "critic_cached_dopt_active":
                preds.append({"id": gid, "prediction": {"queries_b1": get_cached_dopt(n, 120), "want_b2": True}})
            else:
                # Cached 200-query D-optimal design (0.01s across all 16 games!)
                preds.append({"id": gid, "prediction": {"queries_b1": get_cached_dopt(n, 200), "want_b2": False}})
        elif step == "propose_b2":
            rem = int(it["remaining_budget"])
            seen = {int(m) for m, _ in it["observed"]}
            beta1 = fit_wls_exact_ksii(n, it["observed"], debiased_order3=False)
            pairs_ij = [(i, j) for i in range(n) for j in range(i + 1, n)]
            top_p = [pairs_ij[k] for k in np.argsort(-np.abs(beta1[n:]))[:20]]
            qs2 = []
            # Query exact pair coalitions and leave-two-out complements first!
            for (i, j) in top_p:
                for m in ((1 << i) | (1 << j), full ^ ((1 << i) | (1 << j))):
                    if m not in seen and len(qs2) < rem:
                        seen.add(m); qs2.append(m)
            # Fill remaining slots with cached D-optimal sequence
            for m in get_cached_dopt(n, 200):
                if m not in seen and len(qs2) < rem:
                    seen.add(m); qs2.append(m)
            preds.append({"id": gid, "prediction": {"queries_b2": qs2[:rem]}})
        elif step == "estimate":
            est = fit_wls_exact_ksii(n, it["observed"], debiased_order3=(R2_MODE == "crossover_debiased_order3"))
            preds.append({"id": gid, "prediction": est.tolist()})
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f)

if __name__ == "__main__":
    main()
