import argparse, json
import numpy as np

MODE = 'explore_multi_imp_stab'

def standardize_obs(X_obs, M):
    cnt = np.maximum(1.0, np.sum(M, axis=0, keepdims=True))
    mean = np.sum(X_obs * M, axis=0, keepdims=True) / cnt
    Xc = (X_obs - mean) * M
    var = np.sum((Xc ** 2) * M, axis=0, keepdims=True) / np.maximum(1.0, cnt - 1.0)
    std = np.where(var > 1e-8, np.sqrt(var), 1.0)
    return Xc / std

def pcorr_from_prec(Theta):
    iu = np.triu_indices(Theta.shape[0], k=1)
    d = np.sqrt(np.maximum(1e-12, np.diag(Theta)))
    return np.abs(Theta[iu]) / (d[iu[0]] * d[iu[1]])

def admm_glasso(C_psd, lam=0.12, rho=1.5, n_iter=70):
    """Exact Boyd et al. Eigendecomposition ADMM for Graphical Lasso: min_{Theta>0} Tr(C Theta) - logdet(Theta) + lam ||Theta_-diag||_1."""
    p = C_psd.shape[0]
    Z = np.eye(p, dtype=np.float64)
    U = np.zeros((p, p), dtype=np.float64)
    thresh = lam / rho
    for _ in range(n_iter):
        mat = rho * (Z - U) - C_psd
        vals, Q = np.linalg.eigh(0.5 * (mat + mat.T))
        theta_eig = (vals + np.sqrt(vals ** 2 + 4.0 * rho)) / (2.0 * rho)
        Theta = (Q * theta_eig[None, :]) @ Q.T
        V = Theta + U
        Z_new = np.sign(V) * np.maximum(0.0, np.abs(V) - thresh)
        np.fill_diagonal(Z_new, np.diag(V))
        U = U + Theta - Z_new
        Z = Z_new
    return Z

def estimate_d5_bc(X_obs, M, seed):
    X = standardize_obs(X_obs, M)
    T, p = X.shape
    iu = np.triu_indices(p, k=1)

    if MODE == "spectral_ipw_admm_glasso":
        # Estimate time-varying observation propensity pi_t from row-level observed volatility |mean_obs(X_t)| + column rate!
        obs_cnt_t = np.maximum(1.0, np.sum(M, axis=1))
        row_vol = np.sum(np.abs(X) * M, axis=1) / obs_cnt_t  # proxy for |market factor|!
        # Logistic regression / empirical calibration of P(M_{ti}=1 | row_vol_t)
        col_rate = np.clip(np.mean(M, axis=0), 0.15, 0.99)  # [p]
        # Center row_vol and fit slope via correlation with row obs rate
        r_rate = np.clip(np.mean(M, axis=1), 0.15, 0.99)
        v_c = row_vol - np.mean(row_vol)
        slope = float(np.sum(v_c * (r_rate - np.mean(r_rate))) / (np.sum(v_c ** 2) + 1e-6))
        pi_ti = np.clip(col_rate[None, :] + slope * v_c[:, None], 0.15, 0.99)
        # IPW Horvitz-Thompson covariance C_{ij} = (1/T) sum_t M_{ti} M_{tj} X_{ti} X_{tj} / (pi_{ti} * pi_{tj})
        W_ipw = M / pi_ti
        # Normalize weights per pair so effective sample weights sum to T!
        denom_ij = np.maximum(1.0, W_ipw.T @ W_ipw)
        XW = X * W_ipw
        C_hat = (X * M).T @ (X * M) / np.maximum(1.0, M.T @ M)
        C_ipw = (XW.T @ XW) / denom_ij * float(np.mean(W_ipw ** 2))
        C_mix = 0.5 * C_hat + 0.5 * C_ipw
        vals, vecs = np.linalg.eigh(0.5 * (C_mix + C_mix.T))
        C_psd = (vecs * np.maximum(0.06, vals)[None, :]) @ vecs.T
        C_psd = C_psd / np.sqrt(np.outer(np.diag(C_psd), np.diag(C_psd)) + 1e-12)
        Th1 = admm_glasso(C_psd, lam=0.10, rho=1.5, n_iter=75)
        Th2 = admm_glasso(C_psd, lam=0.16, rho=1.5, n_iter=75)
        return (0.5 * pcorr_from_prec(Th1) + 0.5 * pcorr_from_prec(Th2)).tolist()

    # Explorer: Multiple Conditional Imputation (K=6 draws from regularized covariance) + Stability-Selected ADMM Glasso!
    rng = np.random.RandomState(500 + seed)
    n_ij = np.maximum(1.0, M.T @ M)
    S0 = (X.T @ X) / n_ij
    vals, vecs = np.linalg.eigh(0.5 * (S0 + S0.T))
    Sigma = 0.85 * ((vecs * np.maximum(0.08, vals)[None, :]) @ vecs.T) + 0.15 * np.eye(p)

    # 3 quick EM steps to refine conditional generator
    for _ in range(3):
        M1 = X.copy()
        Sc = np.zeros((p, p))
        for t in range(T):
            miss = np.where(M[t] < 0.5)[0]
            if 0 < len(miss) < p:
                obs = np.where(M[t] > 0.5)[0]
                Soo_i = np.linalg.inv(Sigma[np.ix_(obs, obs)] + 1e-4 * np.eye(len(obs)))
                W_mo = Sigma[np.ix_(miss, obs)] @ Soo_i
                M1[t, miss] = W_mo @ X[t, obs]
                Sc[np.ix_(miss, miss)] += Sigma[np.ix_(miss, miss)] - W_mo @ Sigma[np.ix_(obs, miss)]
        Sigma = 0.90 * ((M1.T @ M1 + Sc) / T) + 0.10 * np.eye(p)

    # 5 conditional multiple imputations + stability probability * partial correlation magnitude!
    sel_freq = np.zeros(len(iu[0]))
    mag_acc = np.zeros(len(iu[0]))
    for k_imp in range(5):
        Xi = X.copy()
        for t in range(T):
            miss = np.where(M[t] < 0.5)[0]
            if 0 < len(miss) < p:
                obs = np.where(M[t] > 0.5)[0]
                Soo_i = np.linalg.inv(Sigma[np.ix_(obs, obs)] + 1e-4 * np.eye(len(obs)))
                mu_m = Sigma[np.ix_(miss, obs)] @ (Soo_i @ X[t, obs])
                cov_m = Sigma[np.ix_(miss, miss)] - Sigma[np.ix_(miss, obs)] @ Soo_i @ Sigma[np.ix_(obs, miss)]
                L_m = np.linalg.cholesky(0.5 * (cov_m + cov_m.T) + 1e-5 * np.eye(len(miss)))
                Xi[t, miss] = mu_m + L_m @ rng.normal(0, 1, size=len(miss))
        # Subsample 85% of rows for stability selection
        sub = rng.choice(T, size=int(0.85 * T), replace=False)
        Ci = np.corrcoef(Xi[sub].T)
        Th = admm_glasso(Ci, lam=0.13, rho=1.5, n_iter=55)
        pc = pcorr_from_prec(Th)
        sel_freq += (pc > 1e-3).astype(np.float64) / 5.0
        mag_acc += pc / 5.0
    return ((0.35 + 0.65 * sel_freq) * mag_acc).tolist()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    req = json.loads(open(args.input_json, encoding="utf-8").read())
    preds = [{"id": it["id"], "prediction": estimate_d5_bc(np.asarray(it["X_obs"], dtype=np.float64), np.asarray(it["mask"], dtype=np.float64), idx)} for idx, it in enumerate(req["items"])]
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f)

if __name__ == "__main__":
    main()
