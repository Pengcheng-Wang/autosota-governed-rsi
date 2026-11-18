# [UNSUCCESSFUL / BLOCKED] B2 — Paper #92 `KmeansLocalOpt`: Blocked `15.8x` Runtime Inflation & Lost to Equal-Compute `kmeans++`

- **Status:** `BLOCKED_NOT_CONFIRMED` (`0` Holdout looks consumed — Holdout kept 100% virgin)
- **Wave / Node:** Wave 2 (`Option B`), `node-0`
- **Domain:** Unsupervised Clustering Local Optimality (`5` benchmark datasets, `wall_clock_sec <= 1.40x` baseline)
- **Mandatory Comparators:** `kmeans_lloyd_base`, `lo_kmeans_paper`, `equal_compute_kmeanspp` (3-restart `kmeans++`)

## 1. Dual Governance Catch
1. **Runtime Ceiling Caught `AutoSOTA`'s `20`-Restart Inflation:** Replaying `AutoSOTA`'s `20`-restart configuration inflated wall-clock runtime by **`15.789x`** (`> 1.40x` cap -> immediate **`ROLLBACK`**).
2. **Equal-Compute Comparator Caught Local-Swap Illusion (`b2_cand_01_hartigan_only`):** Within the `1.40x` runtime budget, single-run Hartigan-Wong swaps slightly improved point NMI/objective over 1-run `Lloyd` (`0.3416` vs `0.3440`) and `LO-KMeans` (`0.3422`), **but lost outright to equal-compute 3-restart `kmeans++` (`0.3404`, `signed_gain = -0.0012`, `ROLLBACK`)**!

## 2. Root Cause & RSI Learning
- Spending extra FLOPs on `kmeans++` random restarts escapes poor global basins far more effectively than spending the same FLOPs on local single-point Hartigan swaps inside one basin.
- **RSI Learning (`Opus 5.5, Max` Rule #3):** Every compute-sensitive optimization claim MUST include an **equal-compute standard baseline (`equal_compute_kmeanspp`)** in `required_comparators`.
