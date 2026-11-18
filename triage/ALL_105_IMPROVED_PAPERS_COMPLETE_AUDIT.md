# Complete `105 / 105` `AutoSOTA` Benchmark Paper Audit & Governed Executability Matrix

Every single repository in `AutoSOTA/improved-papers` (`Paper #1` through `Paper #105`, out of `612` total repo entries including `455` `ICML-2026` and `52` `CVPR-2026` slimmed code trees) has been inspected at the repository-tree, `final_patch.diff`, dataset-availability, and protocol-hygiene level.

## 1. Full-Pool (`105 / 105`) Disposition Summary

| Disposition Category | Paper Count (`/ 105`) | Share | Governance Meaning |
|---|---:|---:|---|
| **`EXECUTED_GOVERNED_RSI` (Waves 1–3)** | `16` | `15.2%` | End-to-end governed `Search -> Confirm -> Sealed Holdout` executed (`7` sealed Holdout wins, `9` honest blocks) |
| **`EXECUTED_MULTIAGENT_RSI` (Wave 4 `D1..D8` + `D9`)** | `9` | `8.6%` | Multi-Agent (`4`-Proposer, `2`-Round Iterative) governed RSI executed on `24 x A100-80GB` |
| **`PROTOCOL_EQUIVALENT_LTSF_FAMILY`** | `11` | `10.5%` | Standard LTSF (`ETT/Weather/ECL`) repos sharing the `xPatch`/`ChannelNorm` split & lookback structure already benchmarked in `A1`, `B1`, and `D4` |
| **`DEFERRED_UNBUNDLED_VISION_AUDIO_DATASET`** | `20` | `19.0%` | Upstream image/video/EEG/speech datasets (`ImageNet`, `RGBNT201`, `HQVI`, `SA-1B`, `BCI`) not bundled in artifact repo |
| **`DEFERRED_EXTERNAL_LLM_CHECKPOINT_OR_CORPUS`** | `17` | `16.2%` | Requires unbundled multi-GB HF LLM weights/corpora or external inference servers (`vLLM`) |
| **`AUDITED_STATIC_STEP0_UNBUNDLED_OR_LOW_HEADROOM`** | `19` | `18.1%` | Statically audited (`S1/S2` saturated ceiling `~99.3% AUROC`, stripped `.arrow`/external data, or custom CUDA/C++ deps) |
| **`NO_GO_NON_PYTHON_OR_STRIPPED_CODE`** | `9` | `8.6%` | `0` `.py` files in artifact tree (Julia/R/C++ or stripped upstream repo) |
| **`NO_GO_B1_LLM_JUDGE_OR_EXTERNAL_API`** | `4` | `3.8%` | Hard Step-0 `B1` `NO_GO`: relies on subjective/mutable LLM judge or closed commercial API |

---

## 2. Paper-by-Paper Master Ledger (`#1` – `#105`)

*(Note: `.py` counts exclude vendored `ablation/` baseline subtrees so counts reflect the target paper repository itself.)*

| # | Repository | Domain | Claimed `AutoSOTA` Gain | `.py` | Size (MB) | Wave / Stage | Disposition & Audit Finding |
|---:|---|---|---:|---:|---:|---|---|
| `1` | `paper-1-SAVVY` | `vision_multimodal_audio` | `5.90%` | `354` | `4.3` | Step-0 Triage | **DEFERRED_UNBUNDLED_VISION_AUDIO_DATASET**: Requires unbundled multi-GB vision/video/EEG/speech dataset or TTA resolution override |
| `2` | `paper-2-PINet` | `other_ml` | `16.72%` | `42` | `0.3` | Wave 4 (`D9`) | **EXECUTED_MULTIAGENT_RSI**: Wave-4 4-Agent RSI (`PINet` hard-constrained QP projection; caught `n_iter_test 50->10` test-iteration cut) |
| `3` | `paper-3-DMSQD` | `combinatorial_opt_rl_routing` | `7.32%` | `0` | `0.0` | Step-0 Triage | **NO_GO_NON_PYTHON_OR_STRIPPED_CODE**: 0 Python source files in artifact repo (DMSQD: C++/R/Julia or stripped tree) |
| `4` | `paper-4-DecentralAttn` | `time_series` | `4.45%` | `19` | `0.1` | Step-0 Triage | **PROTOCOL_EQUIVALENT_LTSF_FAMILY**: Standard LTSF ETT/Weather repo without bundled CSVs; represented by de-confounded A1 (`xPatch`) & B1 (`ChannelNorm`) + D4 (`PEMS04`) |
| `5` | `paper-5-TSAE` | `llm_nlp_reasoning` | `2.25%` | `34` | `0.9` | Step-0 Triage | **NO_GO_B1_LLM_JUDGE_OR_EXTERNAL_API**: Violates B1 (requires external LLM judge / closed API or subjective rubric) |
| `6` | `paper-6-PhySense` | `combinatorial_opt_rl_routing` | `4.16%` | `22` | `0.1` | Step-0 Triage | **DEFERRED_UNBUNDLED_VISION_AUDIO_DATASET**: Requires unbundled multi-GB vision/video/EEG/speech dataset or TTA resolution override |
| `7` | `paper-7-ReasoningIQA` | `vision_multimodal_audio` | `2.68%` | `2` | `2.1` | Step-0 Triage | **DEFERRED_UNBUNDLED_VISION_AUDIO_DATASET**: Requires unbundled multi-GB vision/video/EEG/speech dataset or TTA resolution override |
| `8` | `paper-8-MeanFlows` | `other_ml` | `0.14%` | `1` | `0.2` | Step-0 Triage | **AUDITED_STATIC_STEP0_UNBUNDLED_OR_LOW_HEADROOM**: Claimed 0.14%; flags=['sub_0p5pct_unconfirmed_delta'] |
| `9` | `paper-9-ScoreMissing` | `other_ml` | `0.14%` | `66` | `67.6` | Wave 4 (`D5`) | **EXECUTED_MULTIAGENT_RSI**: Wave-4 4-Agent RSI (`ScoreMissing` `S&P 500` missing-data graphical precision recovery) |
| `10` | `paper-10-SuitabilityFilter` | `other_ml` | `1.56%` | `12` | `78.2` | Wave 3 (`C1`) | **EXECUTED_GOVERNED_RSI**: Blocked (`num_folds 5->10` label mutation; tied calibrated GBDT) |
| `11` | `paper-11-OSD` | `time_series` | `0.25%` | `150` | `39.6` | Step-0 Triage | **DEFERRED_UNBUNDLED_VISION_AUDIO_DATASET**: Requires unbundled multi-GB vision/video/EEG/speech dataset or TTA resolution override |
| `12` | `paper-12-EfficientQAT` | `other_ml` | `6.08%` | `25` | `0.2` | Step-0 Triage | **DEFERRED_EXTERNAL_LLM_CHECKPOINT_OR_CORPUS**: Requires unbundled multi-GB HF LLM weights/corpora not included in AutoSOTA artifact tree |
| `13` | `paper-13-APPL` | `llm_nlp_reasoning` | `14.29%` | `97` | `2.9` | Step-0 Triage | **DEFERRED_EXTERNAL_LLM_CHECKPOINT_OR_CORPUS**: Requires unbundled multi-GB HF LLM weights/corpora not included in AutoSOTA artifact tree |
| `14` | `paper-14-PIGuard` | `llm_nlp_reasoning` | `3.66%` | `10` | `1.6` | Step-0 Triage | **DEFERRED_EXTERNAL_LLM_CHECKPOINT_OR_CORPUS**: Requires unbundled multi-GB HF LLM weights/corpora not included in AutoSOTA artifact tree |
| `15` | `paper-15-FRSpec` | `llm_nlp_reasoning` | `13.44%` | `28` | `1.2` | Step-0 Triage | **DEFERRED_EXTERNAL_LLM_CHECKPOINT_OR_CORPUS**: Requires unbundled multi-GB HF LLM weights/corpora not included in AutoSOTA artifact tree |
| `16` | `paper-16-MathFusion` | `llm_nlp_reasoning` | `20.97%` | `55` | `4.8` | Step-0 Triage | **DEFERRED_EXTERNAL_LLM_CHECKPOINT_OR_CORPUS**: Requires unbundled multi-GB HF LLM weights/corpora not included in AutoSOTA artifact tree |
| `17` | `paper-17-MultimodalGLP` | `vision_multimodal_audio` | `0.83%` | `3` | `0.0` | Step-0 Triage | **DEFERRED_UNBUNDLED_VISION_AUDIO_DATASET**: Requires unbundled multi-GB vision/video/EEG/speech dataset or TTA resolution override |
| `18` | `paper-18-CoTSynth` | `llm_nlp_reasoning` | `3.68%` | `13` | `0.9` | Step-0 Triage | **DEFERRED_EXTERNAL_LLM_CHECKPOINT_OR_CORPUS**: Requires unbundled multi-GB HF LLM weights/corpora not included in AutoSOTA artifact tree |
| `19` | `paper-19-DyScaleUT` | `other_ml` | `2.12%` | `14` | `0.1` | Step-0 Triage | **AUDITED_STATIC_STEP0_UNBUNDLED_OR_LOW_HEADROOM**: Claimed 2.12%; flags=unbundled external data/env |
| `20` | `paper-20-Aristotle` | `llm_nlp_reasoning` | `1.68%` | `5` | `0.9` | Step-0 Triage | **DEFERRED_EXTERNAL_LLM_CHECKPOINT_OR_CORPUS**: Requires unbundled multi-GB HF LLM weights/corpora not included in AutoSOTA artifact tree |
| `21` | `paper-21-TokenRecycling` | `llm_nlp_reasoning` | `19.50%` | `5` | `0.1` | Step-0 Triage | **DEFERRED_EXTERNAL_LLM_CHECKPOINT_OR_CORPUS**: Requires unbundled multi-GB HF LLM weights/corpora not included in AutoSOTA artifact tree |
| `22` | `paper-22-ChainOfReasoning` | `other_ml` | `3.38%` | `21` | `1.3` | Step-0 Triage | **DEFERRED_EXTERNAL_LLM_CHECKPOINT_OR_CORPUS**: Requires unbundled multi-GB HF LLM weights/corpora not included in AutoSOTA artifact tree |
| `23` | `paper-23-GuidedEmbed` | `llm_nlp_reasoning` | `17.50%` | `12` | `2.4` | Step-0 Triage | **DEFERRED_EXTERNAL_LLM_CHECKPOINT_OR_CORPUS**: Requires unbundled multi-GB HF LLM weights/corpora not included in AutoSOTA artifact tree |
| `24` | `paper-24-OutputCentric` | `other_ml` | `2.69%` | `0` | `17.2` | Step-0 Triage | **NO_GO_NON_PYTHON_OR_STRIPPED_CODE**: 0 Python source files in artifact repo (OutputCentric: C++/R/Julia or stripped tree) |
| `25` | `paper-25-DEEPER` | `other_ml` | `21.78%` | `16` | `3.0` | Step-0 Triage | **DEFERRED_EXTERNAL_LLM_CHECKPOINT_OR_CORPUS**: Requires unbundled multi-GB HF LLM weights/corpora not included in AutoSOTA artifact tree |
| `26` | `paper-26-GARTKG` | `graph_bio_physics` | `12.40%` | `19` | `0.2` | Step-0 Triage | **DEFERRED_UNBUNDLED_VISION_AUDIO_DATASET**: Requires unbundled multi-GB vision/video/EEG/speech dataset or TTA resolution override |
| `27` | `paper-27-CiteEval` | `llm_nlp_reasoning` | `24.10%` | `19` | `0.2` | Step-0 Triage | **NO_GO_B1_LLM_JUDGE_OR_EXTERNAL_API**: Violates B1 (requires external LLM judge / closed API or subjective rubric) |
| `28` | `paper-28-SBAM` | `llm_nlp_reasoning` | `0.22%` | `9` | `0.2` | Step-0 Triage | **DEFERRED_EXTERNAL_LLM_CHECKPOINT_OR_CORPUS**: Requires unbundled multi-GB HF LLM weights/corpora not included in AutoSOTA artifact tree |
| `29` | `paper-29-CDQGeoEmbed` | `llm_nlp_reasoning` | `14%` | `0` | `0.6` | Step-0 Triage | **NO_GO_NON_PYTHON_OR_STRIPPED_CODE**: 0 Python source files in artifact repo (CDQGeoEmbed: C++/R/Julia or stripped tree) |
| `30` | `paper-30-ProcessRM` | `other_ml` | `1.12%` | `5` | `0.1` | Step-0 Triage | **NO_GO_B1_LLM_JUDGE_OR_EXTERNAL_API**: Violates B1 (requires external LLM judge / closed API or subjective rubric) |
| `31` | `paper-31-CircuitStability` | `llm_nlp_reasoning` | `35.80%` | `33` | `3.0` | Step-0 Triage | **DEFERRED_EXTERNAL_LLM_CHECKPOINT_OR_CORPUS**: Requires unbundled multi-GB HF LLM weights/corpora not included in AutoSOTA artifact tree |
| `32` | `paper-32-PTSolver` | `combinatorial_opt_rl_routing` | `4.48%` | `4` | `125.6` | Step-0 Triage | **AUDITED_STATIC_STEP0_UNBUNDLED_OR_LOW_HEADROOM**: Claimed 4.48%; flags=unbundled external data/env |
| `33` | `paper-33-GCSE` | `llm_nlp_reasoning` | `2.81%` | `49` | `1.1` | Step-0 Triage | **DEFERRED_EXTERNAL_LLM_CHECKPOINT_OR_CORPUS**: Requires unbundled multi-GB HF LLM weights/corpora not included in AutoSOTA artifact tree |
| `34` | `paper-34-EnsembleWM` | `llm_nlp_reasoning` | `3.46%` | `14` | `76.5` | Step-0 Triage | **DEFERRED_EXTERNAL_LLM_CHECKPOINT_OR_CORPUS**: Requires unbundled multi-GB HF LLM weights/corpora not included in AutoSOTA artifact tree |
| `35` | `paper-35-MoralValuesWA` | `graph_bio_physics` | `8.59%` | `5` | `0.1` | Step-0 Triage | **AUDITED_STATIC_STEP0_UNBUNDLED_OR_LOW_HEADROOM**: Claimed 8.59%; flags=unbundled external data/env |
| `36` | `paper-36-WeakStrongPref` | `other_ml` | `28.42%` | `8` | `0.1` | Step-0 Triage | **NO_GO_B1_LLM_JUDGE_OR_EXTERNAL_API**: Violates B1 (requires external LLM judge / closed API or subjective rubric) |
| `37` | `paper-37-DementiaMask` | `vision_multimodal_audio` | `2.45%` | `30` | `0.2` | Step-0 Triage | **DEFERRED_UNBUNDLED_VISION_AUDIO_DATASET**: Requires unbundled multi-GB vision/video/EEG/speech dataset or TTA resolution override |
| `38` | `paper-38-TinySAM` | `vision_multimodal_audio` | `0.90%` | `22` | `141.8` | Step-0 Triage | **DEFERRED_UNBUNDLED_VISION_AUDIO_DATASET**: Requires unbundled multi-GB vision/video/EEG/speech dataset or TTA resolution override |
| `39` | `paper-39-CALF` | `time_series` | `0.19%` | `22` | `3.5` | Step-0 Triage | **PROTOCOL_EQUIVALENT_LTSF_FAMILY**: Standard LTSF ETT/Weather repo without bundled CSVs; represented by de-confounded A1 (`xPatch`) & B1 (`ChannelNorm`) + D4 (`PEMS04`) |
| `40` | `paper-40-GraniteGuardian` | `llm_nlp_reasoning` | `1.37%` | `0` | `5.3` | Step-0 Triage | **NO_GO_NON_PYTHON_OR_STRIPPED_CODE**: 0 Python source files in artifact repo (Granite Guardian: C++/R/Julia or stripped tree) |
| `41` | `paper-41-AMDM` | `time_series` | `1.07%` | `17` | `12.6` | Step-0 Triage | **AUDITED_STATIC_STEP0_UNBUNDLED_OR_LOW_HEADROOM**: Claimed 1.07%; flags=unbundled external data/env |
| `42` | `paper-42-xPatch` | `time_series` | `6.92%` | `16` | `4.5` | Wave 1 (`A1`) | **EXECUTED_GOVERNED_RSI**: Blocked (`seq_len 96->336` confound; tied `Ridge` at `L=336`) |
| `43` | `paper-43-VHM` | `llm_nlp_reasoning` | `0.67%` | `22` | `2.0` | Step-0 Triage | **DEFERRED_EXTERNAL_LLM_CHECKPOINT_OR_CORPUS**: Requires unbundled multi-GB HF LLM weights/corpora not included in AutoSOTA artifact tree |
| `44` | `paper-44-RGVI` | `vision_multimodal_audio` | `3.40%` | `7` | `0.1` | Step-0 Triage | **DEFERRED_UNBUNDLED_VISION_AUDIO_DATASET**: Requires unbundled multi-GB vision/video/EEG/speech dataset or TTA resolution override |
| `45` | `paper-45-PsLSTM` | `time_series` | `0.86%` | `49` | `1.4` | Step-0 Triage | **PROTOCOL_EQUIVALENT_LTSF_FAMILY**: Standard LTSF ETT/Weather repo without bundled CSVs; represented by de-confounded A1 (`xPatch`) & B1 (`ChannelNorm`) + D4 (`PEMS04`) |
| `46` | `paper-46-NonStatTS` | `time_series` | `2.27%` | `51` | `0.5` | Step-0 Triage | **PROTOCOL_EQUIVALENT_LTSF_FAMILY**: Standard LTSF ETT/Weather repo without bundled CSVs; represented by de-confounded A1 (`xPatch`) & B1 (`ChannelNorm`) + D4 (`PEMS04`) |
| `47` | `paper-47-TimePFN` | `time_series` | `8.77%` | `21` | `0.1` | Step-0 Triage | **PROTOCOL_EQUIVALENT_LTSF_FAMILY**: Standard LTSF ETT/Weather repo without bundled CSVs; represented by de-confounded A1 (`xPatch`) & B1 (`ChannelNorm`) + D4 (`PEMS04`) |
| `48` | `paper-48-Shapiq` | `other_ml` | `30.30%` | `300` | `248.7` | Wave 4 (`D1`) | **EXECUTED_MULTIAGENT_RSI**: Wave-4 4-Agent RSI (`shapiq` locked `budget=200`; caught `N_ENSEMBLE=100` `20,000`-query leak) |
| `49` | `paper-49-HogwildInference` | `time_series` | `4%` | `40` | `0.6` | Step-0 Triage | **AUDITED_STATIC_STEP0_UNBUNDLED_OR_LOW_HEADROOM**: Claimed 4%; flags=unbundled external data/env |
| `50` | `paper-50-CausalPFN` | `conformal_dp_stats_causal` | `15.86%` | `45` | `167.3` | Wave 3 (`C2`) | **EXECUTED_GOVERNED_RSI**: Blocked (`42x` ensemble `19.30x` runtime; IHDP heavy-tail `LCB=-0.161`) |
| `51` | `paper-51-FlashTP` | `graph_bio_physics` | `0.70%` | `15` | `0.5` | Step-0 Triage | **DEFERRED_EXTERNAL_LLM_CHECKPOINT_OR_CORPUS**: Requires unbundled multi-GB HF LLM weights/corpora not included in AutoSOTA artifact tree |
| `52` | `paper-52-NsDiff` | `time_series` | `1.28%` | `0` | `0.5` | Step-0 Triage | **NO_GO_NON_PYTHON_OR_STRIPPED_CODE**: 0 Python source files in artifact repo (NsDiff: C++/R/Julia or stripped tree) |
| `53` | `paper-53-K2VAE` | `time_series` | `1.52%` | `96` | `11.8` | Step-0 Triage | **DEFERRED_UNBUNDLED_VISION_AUDIO_DATASET**: Requires unbundled multi-GB vision/video/EEG/speech dataset or TTA resolution override |
| `54` | `paper-54-TimeBase` | `time_series` | `0.36%` | `65` | `0.7` | Step-0 Triage | **PROTOCOL_EQUIVALENT_LTSF_FAMILY**: Standard LTSF ETT/Weather repo without bundled CSVs; represented by de-confounded A1 (`xPatch`) & B1 (`ChannelNorm`) + D4 (`PEMS04`) |
| `55` | `paper-55-CSBrain` | `vision_multimodal_audio` | `6.25%` | `35` | `0.2` | Step-0 Triage | **DEFERRED_UNBUNDLED_VISION_AUDIO_DATASET**: Requires unbundled multi-GB vision/video/EEG/speech dataset or TTA resolution override |
| `56` | `paper-56-InfoSAM` | `vision_multimodal_audio` | `1.60%` | `0` | `0.0` | Step-0 Triage | **NO_GO_NON_PYTHON_OR_STRIPPED_CODE**: 0 Python source files in artifact repo (InfoSAM: C++/R/Julia or stripped tree) |
| `57` | `paper-57-MDReID` | `other_ml` | `14%` | `81` | `1.9` | Step-0 Triage | **DEFERRED_UNBUNDLED_VISION_AUDIO_DATASET**: Requires unbundled multi-GB vision/video/EEG/speech dataset or TTA resolution override |
| `58` | `paper-58-MindGlitch` | `vision_multimodal_audio` | `1.80%` | `32` | `4.6` | Step-0 Triage | **DEFERRED_UNBUNDLED_VISION_AUDIO_DATASET**: Requires unbundled multi-GB vision/video/EEG/speech dataset or TTA resolution override |
| `59` | `paper-59-SelfSupervised` | `time_series` | `0.30%` | `47` | `0.3` | Step-0 Triage | **PROTOCOL_EQUIVALENT_LTSF_FAMILY**: Standard LTSF ETT/Weather repo without bundled CSVs; represented by de-confounded A1 (`xPatch`) & B1 (`ChannelNorm`) + D4 (`PEMS04`) |
| `60` | `paper-60-IAGGAD` | `vision_multimodal_audio` | `1.83%` | `5` | `0.2` | Step-0 Triage | **DEFERRED_UNBUNDLED_VISION_AUDIO_DATASET**: Requires unbundled multi-GB vision/video/EEG/speech dataset or TTA resolution override |
| `61` | `paper-61-HSGKN` | `conformal_dp_stats_causal` | `2.20%` | `6` | `3.5` | Wave 3 (`C3`) | **EXECUTED_GOVERNED_RSI**: Blocked (`+7.60 pp` `max`-over-1500-epochs peeking; honest `LCB<0`) |
| `62` | `paper-62-VisionTS` | `time_series` | `0.80%` | `41` | `3.6` | Step-0 Triage | **AUDITED_STATIC_STEP0_UNBUNDLED_OR_LOW_HEADROOM**: Claimed 0.80%; flags=['compute_or_ensemble_inflation'] |
| `63` | `paper-63-TSRAG` | `time_series` | `13.90%` | `13` | `1.6` | Step-0 Triage | **PROTOCOL_EQUIVALENT_LTSF_FAMILY**: Standard LTSF ETT/Weather repo without bundled CSVs; represented by de-confounded A1 (`xPatch`) & B1 (`ChannelNorm`) + D4 (`PEMS04`) |
| `64` | `paper-64-PIR` | `time_series` | `10.60%` | `50` | `1.4` | Step-0 Triage | **PROTOCOL_EQUIVALENT_LTSF_FAMILY**: Standard LTSF ETT/Weather repo without bundled CSVs; represented by de-confounded A1 (`xPatch`) & B1 (`ChannelNorm`) + D4 (`PEMS04`) |
| `65` | `paper-65-ProteinBinding` | `graph_bio_physics` | `15.48%` | `10` | `8.4` | Wave 2 (`B4`) | **EXECUTED_GOVERNED_RSI**: Blocked (beat `StaB-ddG`, `LCB=-0.0031` vs `FoldX` on disjoint `#Pdb`) |
| `66` | `paper-66-TropicalAttention` | `combinatorial_opt_rl_routing` | `15.55%` | `6` | `3.0` | Wave 1 (`A2`) | **EXECUTED_GOVERNED_RSI**: SEALED HOLDOUT WIN (`+6.52 pp` Macro-F1 vs Tropical, `+3.97 pp` vs 3L) |
| `67` | `paper-67-KAN-AD` | `time_series` | `0.89%` | `0` | `0.0` | Step-0 Triage | **NO_GO_NON_PYTHON_OR_STRIPPED_CODE**: 0 Python source files in artifact repo (KAN-AD: C++/R/Julia or stripped tree) |
| `68` | `paper-68-SEMPO` | `time_series` | `0.12%` | `16` | `0.5` | Step-0 Triage | **PROTOCOL_EQUIVALENT_LTSF_FAMILY**: Standard LTSF ETT/Weather repo without bundled CSVs; represented by de-confounded A1 (`xPatch`) & B1 (`ChannelNorm`) + D4 (`PEMS04`) |
| `69` | `paper-69-TreeHFD` | `other_ml` | `36.60%` | `20` | `0.3` | Wave 2 (`B5`) | **EXECUTED_GOVERNED_RSI**: SEALED HOLDOUT WIN (`-7.91%` MSE vs `AutoSOTA`, `-93.78%` vs Paper) |
| `70` | `paper-70-CertifiedUnlearning` | `other_ml` | `63.64%` | `8` | `13.6` | Step-0 Triage | **AUDITED_STATIC_STEP0_UNBUNDLED_OR_LOW_HEADROOM**: Claimed 63.64%; flags=unbundled external data/env |
| `71` | `paper-71-NeuralMJD` | `time_series` | `19%` | `18` | `0.2` | Step-0 Triage | **DEFERRED_UNBUNDLED_VISION_AUDIO_DATASET**: Requires unbundled multi-GB vision/video/EEG/speech dataset or TTA resolution override |
| `72` | `paper-72-FedGMT` | `other_ml` | `4.37%` | `30` | `0.1` | Step-0 Triage | **DEFERRED_UNBUNDLED_VISION_AUDIO_DATASET**: Requires unbundled multi-GB vision/video/EEG/speech dataset or TTA resolution override |
| `73` | `paper-73-RLD` | `combinatorial_opt_rl_routing` | `0.24%` | `11` | `0.1` | Wave 4 (`D7`) | **EXECUTED_MULTIAGENT_RSI**: Wave-4 4-Agent RSI (`RLD` combinatorial Max-Cut on `BA/ER/Reg` graphs) |
| `74` | `paper-74-LSVI` | `conformal_dp_stats_causal` | `0.02%` | `13` | `0.1` | Wave 4 (`D8`) | **EXECUTED_MULTIAGENT_RSI**: Wave-4 4-Agent RSI (`LSVI` Bayesian logistic Gaussian variational inference) |
| `75` | `paper-75-TreeSlicedEntropy` | `other_ml` | `0.50%` | `63` | `1.3` | Step-0 Triage | **DEFERRED_UNBUNDLED_VISION_AUDIO_DATASET**: Requires unbundled multi-GB vision/video/EEG/speech dataset or TTA resolution override |
| `76` | `paper-76-AANet` | `other_ml` | `6.79%` | `0` | `0.0` | Step-0 Triage | **NO_GO_NON_PYTHON_OR_STRIPPED_CODE**: 0 Python source files in artifact repo (AANet: C++/R/Julia or stripped tree) |
| `77` | `paper-77-CMNN` | `other_ml` | `0.30%` | `1` | `6.6` | Wave 4 (`D3`) | **EXECUTED_MULTIAGENT_RSI**: Wave-4 4-Agent RSI (`CMNN` certified `0%`-violation monotonic learning across 4 datasets) |
| `78` | `paper-78-ConformalAnomaly` | `conformal_dp_stats_causal` | `0.11%` | `23` | `0.1` | Step-0 Triage | **AUDITED_STATIC_STEP0_UNBUNDLED_OR_LOW_HEADROOM**: Claimed 0.11%; flags=['tiny_gain_within_seed_noise'] |
| `79` | `paper-79-DPFKMeans` | `conformal_dp_stats_causal` | `2.01%` | `24` | `0.1` | Wave 3 (`C4`) | **EXECUTED_GOVERNED_RSI**: SEALED HOLDOUT WIN (`-4.11%` vs Paper, `-6.58%` vs `server_kmeans++`, caught `var=0.49` DGP tamper) |
| `80` | `paper-80-LatentScoreReweight` | `other_ml` | `6.58%` | `37` | `6.4` | Wave 4 (`D2`) | **EXECUTED_MULTIAGENT_RSI**: Wave-4 4-Agent RSI (`LatentScoreReweight` `ACS_income_AZ` worst-group accuracy) |
| `81` | `paper-81-QMamba` | `time_series` | `1.01%` | `12` | `7.6` | Step-0 Triage | **AUDITED_STATIC_STEP0_UNBUNDLED_OR_LOW_HEADROOM**: Claimed 1.01%; flags=unbundled external data/env |
| `82` | `paper-82-OnlineLLMRouting` | `conformal_dp_stats_causal` | `1.61%` | `6` | `8.8` | Step-0 Triage | **AUDITED_STATIC_STEP0_UNBUNDLED_OR_LOW_HEADROOM**: Claimed 1.61%; flags=unbundled external data/env |
| `83` | `paper-83-STaRFormer` | `other_ml` | `3.15%` | `121` | `3.1` | Step-0 Triage | **AUDITED_STATIC_STEP0_UNBUNDLED_OR_LOW_HEADROOM**: Claimed 3.15%; flags=unbundled external data/env |
| `84` | `paper-84-IFT` | `time_series` | `2.30%` | `44` | `0.6` | Step-0 Triage | **PROTOCOL_EQUIVALENT_LTSF_FAMILY**: Standard LTSF ETT/Weather repo without bundled CSVs; represented by de-confounded A1 (`xPatch`) & B1 (`ChannelNorm`) + D4 (`PEMS04`) |
| `85` | `paper-85-NeuralSurv` | `conformal_dp_stats_causal` | `23%` | `60` | `2.0` | Step-0 Triage | **AUDITED_STATIC_STEP0_UNBUNDLED_OR_LOW_HEADROOM**: Claimed 23%; flags=unbundled external data/env |
| `86` | `paper-86-STELLA` | `time_series` | `2.42%` | `177` | `7.9` | Step-0 Triage | **DEFERRED_UNBUNDLED_VISION_AUDIO_DATASET**: Requires unbundled multi-GB vision/video/EEG/speech dataset or TTA resolution override |
| `87` | `paper-87-MoSES` | `combinatorial_opt_rl_routing` | `51.40%` | `31` | `124.6` | Step-0 Triage | **AUDITED_STATIC_STEP0_UNBUNDLED_OR_LOW_HEADROOM**: Claimed 51.40%; flags=unbundled external data/env |
| `88` | `paper-88-ChannelNorm` | `time_series` | `15.20%` | `31` | `1.1` | Wave 2 (`B1`) | **EXECUTED_GOVERNED_RSI**: SEALED HOLDOUT WIN (`-4.45%` vs `RMLP-L336`, `-3.23%` vs `AutoSOTA-L336`, `-1.38%` vs `Ridge`) |
| `89` | `paper-89-CausalVelocity` | `conformal_dp_stats_causal` | `1.66%` | `27` | `4.9` | Step-0 Triage | **AUDITED_STATIC_STEP0_UNBUNDLED_OR_LOW_HEADROOM**: Claimed 1.66%; flags=['compute_or_ensemble_inflation'] |
| `90` | `paper-90-RSTIB` | `time_series` | `0.38%` | `18` | `36.8` | Wave 4 (`D4`) | **EXECUTED_MULTIAGENT_RSI**: Wave-4 4-Agent RSI (`RSTIB` `PEMS04` `96`-active-sensor traffic forecasting from `307`-sensor array; caught `n_sample 12->50` inflation) |
| `91` | `paper-91-TimeAwareCausal` | `conformal_dp_stats_causal` | `4.30%` | `28` | `194.4` | Step-0 Triage | **AUDITED_STATIC_STEP0_UNBUNDLED_OR_LOW_HEADROOM**: Claimed 4.30%; flags=unbundled external data/env |
| `92` | `paper-92-KmeansLocalOpt` | `conformal_dp_stats_causal` | `0.84%` | `2` | `0.2` | Wave 2 (`B2`) | **EXECUTED_GOVERNED_RSI**: Blocked (`15.79x` 20-restart runtime; lost to equal-compute `3-restart kmeans++`) |
| `93` | `paper-93-FedWMSAM` | `other_ml` | `1.77%` | `45` | `0.3` | Step-0 Triage | **DEFERRED_UNBUNDLED_VISION_AUDIO_DATASET**: Requires unbundled multi-GB vision/video/EEG/speech dataset or TTA resolution override |
| `94` | `paper-94-BounDrE` | `vision_multimodal_audio` | `1.27%` | `6` | `38.5` | Step-0 Triage | **DEFERRED_UNBUNDLED_VISION_AUDIO_DATASET**: Requires unbundled multi-GB vision/video/EEG/speech dataset or TTA resolution override |
| `95` | `paper-95-FastFeatureCP` | `conformal_dp_stats_causal` | `5.49%` | `27` | `245.0` | Wave 1 (`A5`) | **EXECUTED_GOVERNED_RSI**: Blocked (2-seed Search winner regressed on 5-seed Confirm) |
| `96` | `paper-96-M3SVM` | `conformal_dp_stats_causal` | `2.37%` | `10` | `57.3` | Wave 1 (`A4`) | **EXECUTED_GOVERNED_RSI**: SEALED HOLDOUT WIN (`+13.09 pp` vs `pmsvm_dp`; caught `_base_coef` `eps=21.21` DP leak) |
| `97` | `paper-97-WassersteinTL` | `conformal_dp_stats_causal` | `3.74%` | `0` | `0.2` | Step-0 Triage | **NO_GO_NON_PYTHON_OR_STRIPPED_CODE**: 0 Python source files in artifact repo (WassersteinTL: C++/R/Julia or stripped tree) |
| `98` | `paper-98-XMahalanobis` | `conformal_dp_stats_causal` | `1.07%` | `14` | `1.4` | Step-0 Triage | **AUDITED_STATIC_STEP0_UNBUNDLED_OR_LOW_HEADROOM**: Claimed 1.07%; flags=unbundled external data/env |
| `99` | `paper-99-OLLALanding` | `other_ml` | `4.83%` | `16` | `0.8` | Wave 4 (`D6`) | **EXECUTED_MULTIAGENT_RSI**: Wave-4 4-Agent RSI (`OLLALanding` constrained manifold MCMC under locked `n_steps=400`) |
| `100` | `paper-100-InvMissingData` | `conformal_dp_stats_causal` | `2.50%` | `31` | `4.8` | Wave 3 (`C5`) | **EXECUTED_GOVERNED_RSI**: Blocked (`3.66x` `n_pairs=8`; beat Paper & MICE, `p=0.0228>0.0125` vs KNN) |
| `101` | `paper-101-ACIA` | `conformal_dp_stats_causal` | `0.57%` | `23` | `3.6` | Step-0 Triage | **AUDITED_STATIC_STEP0_UNBUNDLED_OR_LOW_HEADROOM**: Claimed 0.57%; flags=['compute_or_ensemble_inflation'] |
| `102` | `paper-102-StochasticFF` | `other_ml` | `1.33%` | `9` | `0.1` | Step-0 Triage | **AUDITED_STATIC_STEP0_UNBUNDLED_OR_LOW_HEADROOM**: Claimed 1.33%; flags=unbundled external data/env |
| `103` | `paper-103-QDCP` | `conformal_dp_stats_causal` | `6.33%` | `12` | `1.4` | Wave 3 (`C6`) | **EXECUTED_GOVERNED_RSI**: SEALED HOLDOUT WIN (`-10.38%` set size vs Paper & `brentq`; caught `eps_0=0.001` Search floor breach) |
| `104` | `paper-104-BalancedActiveInf` | `conformal_dp_stats_causal` | `24.80%` | `8` | `0.1` | Wave 2 (`B3`) | **EXECUTED_GOVERNED_RSI**: SEALED HOLDOUT WIN (`-26.73%` CI width vs Paper, `-4.58%` vs `AutoSOTA` + valid coverage) |
| `105` | `paper-105-BinaryClassEval` | `conformal_dp_stats_causal` | `2.40%` | `37` | `0.2` | Wave 2 (`B6`) | **EXECUTED_GOVERNED_RSI**: Halted at Search (PhysioNet stripped; `BinormalGenerator` at Bayes floor) |
