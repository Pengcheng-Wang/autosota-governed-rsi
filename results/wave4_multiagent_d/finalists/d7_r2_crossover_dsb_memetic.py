import argparse, json
import numpy as np

R2_MODE = 'crossover_dsb_memetic'

def polish_1_and_2flip(A, s_in):
    n = len(A)
    s = s_in.copy()
    field = A @ s
    for _ in range(45):
        gains = s * field
        i = int(np.argmax(gains))
        if gains[i] <= 1e-9:
            break
        old = s[i]
        s[i] = -old
        field -= 2.0 * old * A[:, i]

    # Exact vectorized 2-flip check on top-24 borderline/frustrated vertices!
    for _ in range(6):
        gains1 = s * field  # <= 0 at 1-flip local optimum!
        cand_v = np.argsort(-gains1)[:28]
        # Gain of flipping both (u, v) simultaneously is gains1[u] + gains1[v] - 2 * s[u] * s[v] * A[u, v]!
        g_u = gains1[cand_v]
        s_u = s[cand_v]
        A_sub = A[np.ix_(cand_v, cand_v)]
        pair_gain = g_u[:, None] + g_u[None, :] - 2.0 * np.outer(s_u, s_u) * A_sub
        np.fill_diagonal(pair_gain, -1e9)
        bi, bj = np.unravel_index(int(np.argmax(pair_gain)), pair_gain.shape)
        if pair_gain[bi, bj] <= 1e-9:
            break
        u, v = int(cand_v[bi]), int(cand_v[bj])
        ou, ov = s[u], s[v]
        s[u], s[v] = -ou, -ov
        field -= 2.0 * ou * A[:, u] + 2.0 * ov * A[:, v]
        # Re-run 1-flip cascade after escaping 1-flip local optimum!
        for _ in range(25):
            g1 = s * field
            im = int(np.argmax(g1))
            if g1[im] <= 1e-9:
                break
            om = s[im]
            s[im] = -om
            field -= 2.0 * om * A[:, im]
    return s

def solve_d7_r2(A, seed):
    n = len(A)
    rng = np.random.RandomState(7000 + seed)
    tot_w = float(np.sum(A))
    eig_max = float(np.linalg.norm(A, ord=2))
    xi = 0.72 / max(1e-6, eig_max)

    # Ensemble 1: 32 Ballistic + Discrete Simulated Bifurcation chains
    M = 32
    X = rng.uniform(-0.1, 0.1, size=(M, n))
    P = np.zeros((M, n), dtype=np.float64)
    dt = 0.45
    for t in range(130):
        a_p = t / 129.0
        spin_signal = np.sign(X) if (R2_MODE == "crossover_dsb_memetic" and t > 45) else X
        P = P + dt * (-(1.0 - a_p) * X - xi * (spin_signal @ A))
        Xn = X + dt * P
        wall = np.abs(Xn) >= 1.0
        X = np.clip(Xn, -1.0, 1.0)
        P = np.where(wall, 0.0, P)
    S_sb = np.where(X >= 0.0, 1.0, -1.0)

    # Ensemble 2: Rank-10 Burer-Monteiro Riemannian SDP + 16 hyperplanes
    V = rng.normal(0.0, 1.0, size=(n, 10))
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    for it in range(45):
        G = A @ V
        st = 0.65 if it < 28 else 0.35
        Vn = (1.0 - st) * V - st * (G / np.maximum(1e-8, np.linalg.norm(G, axis=1, keepdims=True)))
        V = Vn / np.maximum(1e-8, np.linalg.norm(Vn, axis=1, keepdims=True))
    S_sdp = np.where((V @ rng.normal(0, 1, size=(10, 16))).T >= 0.0, 1.0, -1.0)

    S_all = np.vstack([S_sb, S_sdp])
    cuts = 0.25 * (tot_w - np.sum((S_all @ A) * S_all, axis=1))
    elite = []
    for idx in np.argsort(-cuts)[:10]:
        sp = polish_1_and_2flip(A, S_all[int(idx)])
        cp = 0.25 * (tot_w - float(sp @ A @ sp))
        elite.append((cp, sp))
    elite.sort(key=lambda x: -x[0])

    if R2_MODE == "crossover_dsb_memetic":
        for _ in range(12):
            i1, i2 = rng.choice(min(6, len(elite)), size=2, replace=False)
            s1, s2 = elite[int(i1)][1], elite[int(i2)][1]
            if float(s1 @ s2) < 0.0:
                s2 = -s2
            ch = np.where(s1 == s2, s1, np.where(rng.rand(n) < 0.5, 1.0, -1.0))
            if np.all(s1 == s2):
                flip_i = rng.choice(n, size=max(4, n // 20), replace=False)
                ch[flip_i] *= -1.0
            ch = polish_1_and_2flip(A, ch)
            cc = 0.25 * (tot_w - float(ch @ A @ ch))
            if cc > elite[0][0]:
                elite.insert(0, (cc, ch))
    return elite[0][1].tolist()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    req = json.loads(open(args.input_json, encoding="utf-8").read())
    preds = [{"id": it["id"], "prediction": solve_d7_r2(np.asarray(it["adj"], dtype=np.float64), int(it["seed"]))} for it in req["items"]]
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f)

if __name__ == "__main__":
    main()
