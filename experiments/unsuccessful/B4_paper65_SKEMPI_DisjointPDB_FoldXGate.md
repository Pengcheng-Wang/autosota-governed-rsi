# [UNSUCCESSFUL / BLOCKED] B4 — Paper #65 `ProteinBinding` (`SKEMPI`): Beat `StaB-ddG` & `FlexddG`, Blocked by `FoldX` LCB Across Disjoint `#Pdb` Complexes

- **Status:** `BLOCKED_NOT_CONFIRMED` (`0` Holdout looks consumed — Holdout kept 100% virgin)
- **Wave / Node:** Wave 2 (`Option B`), `node-1`
- **Domain:** Protein-Protein Binding Affinity $\Delta\Delta G$ Ranking (`1,286` mutations across `43` strictly disjoint `#Pdb` complexes)
- **Mandatory Comparators:** `stab_ddg_base`, `foldx_base`, `flexddg_base`

## 1. Confirm Gate Results (`b4_cand_04_within_complex_z_rank_blend`, `8` Disjoint PDB-Complex Clusters)
- vs `stab_ddg_base` (`0.4591` per-complex Spearman $\rho$): Candidate achieved **`0.5287`** (`+6.96 pp`, `LCB = +0.0193`, **`PROMOTE`**)
- vs `flexddg_base` (`0.4272`): Candidate achieved **`0.5287`** (`+10.15 pp`, `LCB = +0.0414`, **`PROMOTE`**)
- vs `foldx_base` (`0.4665`): Candidate achieved **`0.5287`** (`+6.22 pp` point gain, **but `LCB = -0.0031 < 0`** at $\alpha/2 = 0.0125$ across `8` heterogeneous PDB clusters -> **`ROLLBACK`**!)
- **`gate-holdout --phase reserve` Verdict:** `BLOCKED_NOT_CONFIRMED` (`missing_or_failed_comparators: ["foldx_base:ROLLBACK"]`).

## 2. Root Cause & RSI Learning
- Grouping strictly by `#Pdb` protein complex (rather than leaking homologous mutations via random row splits) reveals high complex-to-complex heterogeneity: `FoldX`'s physical van-der-Waals clash terms dominate on steric-clash complexes where `StaB-ddG` fails, creating heavy cluster variance across only `8` Confirm complex buckets (`+6.22 pp` mean gain, `SE = 0.030`).
- **RSI Learning:** Enforcing `leakage unit == split unit == cluster unit` (`#Pdb`) prevented false-certainty promotion from row-level pseudoreplication (`N=1,286` mutations vs `G=8` true independent protein interfaces).
