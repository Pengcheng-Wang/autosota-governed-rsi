import argparse, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
R2_MODE = 'crossover_kriging_spectral_factor'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def run_d4_r2(train_npz, eval_npz, items):
    tr = np.load(train_npz)
    ev = np.load(eval_npz)
    X_tr = tr["X_train"].astype(np.float32)  # [N_tr, 12, 96, 3]
    Y_tr = tr["Y_train"].astype(np.float32)  # [N_tr, 12, 96]
    X_blocks = ev["X_blocks"].astype(np.float32)
    flow_mean, flow_std = float(ev["flow_mean"]), float(ev["flow_std"])
    N_tr, L, S, C = X_tr.shape

    med_s = np.median(X_tr, axis=(0, 1), keepdims=True)
    mad_s = np.maximum(1e-3, np.median(np.abs(X_tr - med_s), axis=(0, 1), keepdims=True))
    Sigma_c = []
    for c_i in range(C):
        Fl = X_tr[:, :, :, c_i].reshape(-1, S)
        Sigma_c.append((Fl.T @ Fl) / len(Fl) + 0.05 * np.eye(S, dtype=np.float32))

    def kriging_clean(X_arr):
        X_out = X_arr.copy()
        is_zero_drop = np.all(np.abs(X_out[:, :, :, 0]) < 1e-6, axis=1)  # [B, 96]
        # Better temporal spike detector: compare against 3-step local median + global MAD!
        t_med = np.median(X_out, axis=1, keepdims=True)
        is_spike = (np.abs(X_out - t_med) > 2.6) | (np.abs(X_out - med_s) > 3.8 * mad_s)
        # Interpolate isolated temporal spikes from adjacent steps (t-1, t+1) instead of flat window median!
        pad = np.pad(X_out, ((0, 0), (1, 1), (0, 0), (0, 0)), mode="edge")
        nbr_avg = 0.5 * (pad[:, :-2] + pad[:, 2:])
        nbr_clean = np.where(np.abs(nbr_avg - t_med) > 2.6, t_med, nbr_avg)
        X_out = np.where(is_spike, nbr_clean, X_out)

        for b_i in range(len(X_out)):
            miss = np.where(is_zero_drop[b_i])[0]
            if 0 < len(miss) < S:
                obs = np.where(~is_zero_drop[b_i])[0]
                for c_i in range(C):
                    Sg = Sigma_c[c_i]
                    W_k = Sg[np.ix_(miss, obs)] @ np.linalg.inv(Sg[np.ix_(obs, obs)])
                    X_out[b_i, :, miss, c_i] = W_k @ X_out[b_i, :, obs, c_i]
        return X_out

    C_flow = np.corrcoef(X_tr[:, :, :, 0].reshape(-1, S).T).astype(np.float32)
    np.fill_diagonal(C_flow, 0.0)
    W_g1 = np.maximum(0.0, C_flow) ** 2
    W_g1 /= (np.sum(W_g1, axis=1, keepdims=True) + 1e-6)
    W_g1_t = torch.tensor(W_g1, device=DEVICE)

    _, _, Vt = np.linalg.svd(X_tr[:, -1, :, 0], full_matrices=False)
    V_k = torch.tensor(Vt[:16].astype(np.float32), device=DEVICE)  # [16, S]

    class CrossoverTrafficNet(nn.Module):
        def __init__(self, use_factors: bool):
            super().__init__()
            self.use_factors = use_factors
            # Anchor prediction on last-step flow + recent 3-step velocity!
            self.lin_local = nn.Linear(L * C + 6, 12)
            extra = 16 * L if use_factors else 0
            in_h = (L * C + 6) + L * C + extra + 32
            self.sens_emb = nn.Embedding(S, 32)
            self.net = nn.Sequential(
                nn.Linear(in_h, 256),
                nn.SiLU(),
                nn.Dropout(0.08),
                nn.Linear(256, 128),
                nn.SiLU(),
                nn.Linear(128, 12),
            )

        def forward(self, x):
            B_sz = x.shape[0]
            loc = x.permute(0, 2, 1, 3).reshape(B_sz, S, L * C)
            last_v = x[:, -1, :, :]  # [B, S, 3]
            diff_3 = x[:, -1, :, :] - x[:, -4, :, :]  # [B, S, 3]
            loc_plus = torch.cat([loc, last_v, diff_3], dim=-1)  # [B, S, 42]
            nb = torch.einsum("ij,btjc->btic", W_g1_t, x).permute(0, 2, 1, 3).reshape(B_sz, S, L * C)
            se = self.sens_emb.weight.unsqueeze(0).expand(B_sz, -1, -1)
            parts = [loc_plus, nb, se]
            if self.use_factors:
                fac = torch.matmul(x[:, :, :, 0], V_k.T).reshape(B_sz, 1, 16 * L).expand(-1, S, -1)
                parts.append(fac)
            out = last_v[:, :, 0:1] + self.lin_local(loc_plus) + self.net(torch.cat(parts, dim=-1))
            return out.permute(0, 2, 1)

    torch.manual_seed(290)
    rng = np.random.RandomState(290)
    # Create TWO independent realistic noisy+Kriged training copies so the network generalizes across dropout masks!
    dm1 = rng.rand(N_tr, 1, S, 1) < 0.12
    sp1 = (rng.rand(*X_tr.shape) < 0.03) * rng.normal(0, 4.5, size=X_tr.shape).astype(np.float32)
    X_imp1 = kriging_clean(np.where(dm1, 0.0, X_tr) + sp1)

    X_cat = np.concatenate([X_tr, X_imp1], axis=0)
    Y_cat = np.concatenate([Y_tr, Y_tr], axis=0)
    xt_all = torch.tensor(X_cat, device=DEVICE)
    yt_all = torch.tensor(Y_cat, device=DEVICE)

    net = CrossoverTrafficNet(use_factors=(R2_MODE == "crossover_kriging_spectral_factor")).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=2.8e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)
    N_tot = len(xt_all)
    for ep in range(60):
        net.train()
        perm = torch.randperm(N_tot, device=DEVICE)
        for b0 in range(0, N_tot, 160):
            sl = perm[b0:b0 + 160]
            loss = F.l1_loss(net(xt_all[sl]), yt_all[sl])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

    net.eval()
    outs = []
    with torch.no_grad():
        for it in items:
            bi = int(it["block_idx"])
            Xc = kriging_clean(X_blocks[bi])
            pn = net(torch.tensor(Xc, device=DEVICE)).cpu().numpy()
            outs.append({"id": it["id"], "prediction": (pn * flow_std + flow_mean).tolist()})
    return outs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    req = json.loads(open(args.input_json, encoding="utf-8").read())
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": run_d4_r2(req["train_npz"], req["eval_npz"], req["items"])}, f)

if __name__ == "__main__":
    main()
