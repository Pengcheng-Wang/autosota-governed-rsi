import argparse, json, math
import numpy as np
import torch

torch.set_num_threads(1)
MODE = 'explore_natgrad_tr'

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]

def halton_seq(n_pts, dim):
    U = np.empty((n_pts, dim), dtype=np.float64)
    for j in range(dim):
        b = PRIMES[j % len(PRIMES)]
        for i in range(n_pts):
            f, r, k = 1.0, 0.0, i + 1
            while k > 0:
                f /= b
                r += f * (k % b)
                k //= b
            U[i, j] = r
    return np.clip(U, 1e-6, 1.0 - 1e-6)

def inv_norm_cdf(u):
    # High-accuracy Acklam rational inverse normal CDF via torch.erfinv
    ut = torch.tensor(u, dtype=torch.float64)
    return (math.sqrt(2.0) * torch.erfinv(2.0 * ut - 1.0)).numpy()

def newton_init(X, y, prior_prec):
    N, d = X.shape
    w = np.zeros(d, dtype=np.float64)
    eye = prior_prec * np.eye(d)
    for _ in range(12):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -25.0, 25.0)))
        H = (X.T * np.maximum(1e-6, p * (1.0 - p))[None, :]) @ X + eye
        step = np.linalg.solve(H, X.T @ (p - y) + prior_prec * w)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    p = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -25.0, 25.0)))
    Sigma = np.linalg.inv((X.T * np.maximum(1e-6, p * (1.0 - p))[None, :]) @ X + eye)
    return w, Sigma

def fit_d8_bc(X, y, prior_prec):
    N, d = X.shape
    mu0, Sig0 = newton_init(X, y, prior_prec)
    L0 = np.linalg.cholesky(0.5 * (Sig0 + Sig0.T) + 1e-9 * np.eye(d))

    # Build antithetic Halton Quasi-Monte Carlo normal draws (256 + 256 = 512 low-discrepancy points!)
    eh = inv_norm_cdf(halton_seq(256, d))
    eps_qmc = np.vstack([eh, -eh])  # [512, d]

    if MODE == "spectral_qmc_lbfgs":
        Xt = torch.tensor(X, dtype=torch.float64)
        yt = torch.tensor(y, dtype=torch.float64)
        epst = torch.tensor(eps_qmc, dtype=torch.float64)
        tril_idx = torch.tril_indices(d, d, offset=-1)
        params = torch.cat([
            torch.tensor(mu0, dtype=torch.float64),
            torch.tensor(np.log(np.diag(L0)), dtype=torch.float64),
            torch.tensor(L0[tril_idx[0].numpy(), tril_idx[1].numpy()], dtype=torch.float64),
        ]).clone().detach().requires_grad_(True)

        opt = torch.optim.LBFGS([params], lr=1.0, max_iter=35, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            mu = params[:d]
            log_d = params[d:2 * d]
            L_mat = torch.zeros((d, d), dtype=torch.float64)
            L_mat[range(d), range(d)] = torch.exp(log_d)
            L_mat[tril_idx[0], tril_idx[1]] = params[2 * d:]
            th = mu.unsqueeze(0) + epst @ L_mat.T
            logits = torch.clamp(th @ Xt.T, -30.0, 30.0)
            ll = torch.sum(yt.unsqueeze(0) * logits - torch.nn.functional.softplus(logits), dim=1)
            lp = -0.5 * prior_prec * torch.sum(th ** 2, dim=1)
            loss = -(torch.mean(ll + lp) + torch.sum(log_d))
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            p_np = params.numpy()
            mu_f = p_np[:d]
            L_f = np.zeros((d, d), dtype=np.float64)
            np.fill_diagonal(L_f, np.exp(p_np[d:2 * d]))
            L_f[tril_idx[0].numpy(), tril_idx[1].numpy()] = p_np[2 * d:]
        return {"mu": mu_f.tolist(), "L": L_f.tolist()}

    # Explorer: Whitened QMC Natural-Gradient Khan-Lin VON/VOGN with Armijo Trust-Region Line Search!
    mu = mu0.copy()
    Prec = np.linalg.inv(Sig0)
    Sigma = Sig0.copy()

    def eval_qmc_elbo(m_c, S_c):
        L_c = np.linalg.cholesky(0.5 * (S_c + S_c.T) + 1e-10 * np.eye(d))
        th = m_c[None, :] + eps_qmc @ L_c.T
        lg = np.clip(th @ X.T, -30.0, 30.0)
        ll = np.sum(y[None, :] * lg - np.log1p(np.exp(lg)), axis=1)
        lp = -0.5 * prior_prec * np.sum(th ** 2, axis=1)
        return float(np.mean(ll + lp) + np.sum(np.log(np.diag(L_c)))), L_c

    cur_elbo, cur_L = eval_qmc_elbo(mu, Sigma)
    for _ in range(15):
        th = mu[None, :] + eps_qmc @ cur_L.T
        lg = np.clip(th @ X.T, -30.0, 30.0)
        sig = 1.0 / (1.0 + np.exp(-lg))
        g_bar = np.mean((sig - y[None, :]) @ X + prior_prec * th, axis=0)
        H_bar = (X.T * np.mean(np.maximum(1e-7, sig * (1.0 - sig)), axis=0)[None, :]) @ X + prior_prec * np.eye(d)
        for beta in [1.0, 0.5, 0.25, 0.1]:
            Prec_try = (1.0 - beta) * Prec + beta * H_bar
            Sig_try = np.linalg.inv(Prec_try)
            mu_try = mu - beta * (Sig_try @ g_bar)
            el_try, L_try = eval_qmc_elbo(mu_try, Sig_try)
            if el_try >= cur_elbo - 1e-10:
                mu, Prec, Sigma, cur_elbo, cur_L = mu_try, Prec_try, Sig_try, el_try, L_try
                break
    return {"mu": mu.tolist(), "L": cur_L.tolist()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    req = json.loads(open(args.input_json, encoding="utf-8").read())
    preds = [{"id": it["id"], "prediction": fit_d8_bc(np.asarray(it["X"], dtype=np.float64), np.asarray(it["y"], dtype=np.float64), float(it["prior_prec"]))} for it in req["items"]]
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f)

if __name__ == "__main__":
    main()
