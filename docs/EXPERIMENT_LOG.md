# Experimental log: Track B, guaranteed-quality screening of SWOT river water levels

Owner: Ravi. Period: 13-14 Oct 2026. Workspace: DGX `/workspace/ravi/swot` (code/, data/, results/, logs/, docs/).
This log records every experiment in order: purpose, design, scripts, outputs, key results and decisions. Exact file
timestamps, sizes and checksums are in the automatically generated appendix at the end. Final numbers are those of the
frozen pipeline (E14 onwards); earlier entries are kept as the record of how the design evolved.
Times are IST (UTC+5:30) unless stated. Design notes: docs/design_note_v0.1 .. v0.6 (v0.6 is current).

## 0. Environment
- DGX: NVIDIA B200 in MIG mode (slice with about 45 GB, 30 SMs), driver 570.124.06, CUDA 12.8. Access via
  `C:\Users\admin\dgx_tools\dgx.py` (Jupyter terminal API); long jobs under nohup.
- Python 3.12; pandas 2.2.3, scikit-learn 1.6.1, xgboost 2.1.3 (CUDA build), cupy 13.3.0, torch 2.7.0a0 (CUDA),
  pyarrow 19.0.1, matplotlib 3.10.1; h5py 3.16.0 installed project-locally in external/pylib (system untouched).
- Third-party code: SCoRE (Bai and Jin 2026, arXiv 2603.24704), github.com/Tian-Bai/SCoRE, MIT licence, commit 401b7ca,
  cloned to external/SCoRE and called unchanged or through a verified GPU port.

## 1. Data sources
| Source | Identifier | Verification |
|---|---|---|
| USGS paired SWOT-gauge release (Harlan et al. 2026), 9 files | doi:10.5066/P1FE9W9E, ScienceBase 688ce6bdd4be02086276c296 | MD5 of all 9 files vs publisher (docs/usgs_paired_MANIFEST.tsv) |
| SWORD v17b netCDF (Altenau et al.) | Zenodo 15299138 | MD5 ebab786a5c790eb571ac86c9adf1aca3 |
| SWOT RiverSP Version D, Brazil reaches and nodes | Hydrocron API (SWOT_L2_HR_RiverSP_D) | raw CSV per feature |
| ANA telemetric stage, Brazil (15 min) | ANA legacy telemetry service (no login) | raw records per station (Parquet) |

## 2. Chronological experiment entries

### E00 Data acquisition and integrity (13 Oct)
Purpose: obtain and verify the USGS release. ScienceBase blocks scripted access (Cloudflare, HTTP 403) from the DGX and
the laptop; files were downloaded in a browser and uploaded through JupyterLab. Script: code/verify_usgs_paired.py
(sizes, MD5, sha256). Result: 9/9 files match. Outputs: docs/sciencebase_item_*.json (transcribed record),
docs/usgs_paired_MANIFEST.tsv, docs/usgs_paired.md5, docs/metadata_digest.md (code/digest_metadata.py).
Finding: USGS headline accuracies are median-normalised per site (bias removed).

### E01 Audit v0 (code/audit_data_v0.py; results/audit_v0)
Counts by type and version, label definitions (B bias-relative, A absolute EGM2008, U USGS-style), error tails,
wse_u calibration, C/D overlap, implausible values, USGS grid preview. Key results: 2,662 gauges; 35-45% of pairs exceed
25 cm under label B; reach_q non-monotone in error and shifting between versions; wse_u ranks errors but is miscalibrated.
Decision: label B (fixed per-site offset) primary, label A secondary.

### E02 Audit v1 / v1b (code/audit_data_v1.py, audit_data_v1b.py, swotio.py; results/audit_v1)
Duplicates (29,670 groups; mostly the same SWOT observation paired with two gauge series), pairs > 1 h (12,666, all BOR
daily lake values), C/D re-match by gauge+cycle+pass (176,406 overpasses), decoding of the USGS filter grid.
Fidelity anchor: river grid rows reproduced within 0.33 pp (reach) and 0.85 pp (node) retention and 0.9 / 2 cm q68 under
R-style NaN handling and duplicate collapse; lakes not reproducible (missing lake prior in Version C).

### E03 Cleaning v1 (code/clean_v1.py; docs/cleaning_log_v1.md; data/processed/paired_clean_v1.parquet, cd_crosswalk_v1.parquet)
Rules: drop BOR (daily) and pairs > 1 h; one row per overpass (dominant datum series; PGC0 over PIC0; unresolved conflicts
dropped); flags (calibration phase, tidal gauges, mixed datum, shared SWOT observation, EGM offset stability); fill and
impossible values to NaN; labels B and A. Result: 633,541 -> 583,399 rows; C/D crosswalk 172,003 overpasses.

### E04 Failure check (code/check_failure_mode.py; results/failure_check_v1)
Flag grid in-sample (M1), observation-level conformal (M2), site-disjoint conformal (M3). Result: M2 reports 10% but
delivers 11-12% at new sites (leakage); pooled control leaves the average held-out site at about twice alpha.

### E05 Site-level prototypes (code/prototype_site_level.py, prototype_selective.py, prototype_score_sdr*.py;
results/site_level_proto_v1..v4)
S1 site-level CRC, S2 Learn-then-Test, S3 gate + CRC (abstention counted as zero: inflates error where data are kept),
S4 selective LTT (valid but too conservative), S5 e-BH mixture (failed: mixture dilutes e-values; diagnosed),
S5b single threshold, S6 SCoRE-SDR (authors' code; 'homo' pruning). S6 adopted.

### E06 Literature check (13 Oct)
Selective risk control already exists (SCRC 2025, SCoRE 2026, SCoRC 2026; hierarchical selection and group-weighted CP).
Positioning: domain-methodological contribution; statistical engine credited.

### E07 GPU port of SCoRE (code/score_sdr_gpu.py; logs/score_sdr_gpu_equivalence*.log)
Vectorised e-values (PyTorch, float64), authors' pruning and e-BH reused. Equivalence: 72/72 unweighted and 72/72 weighted
cases bit-exact, identical selections. One fix: scalar/tensor division replaced by tensor/tensor (1-ulp differences).

### E08 Frontier v5 (code/run_frontier_v5.py, features_v2.py; results/frontier_v5)
XGBoost on CUDA, extended label-free features, 4 x 5-fold over units, tau {0.10, 0.25, 0.50}, alpha {0.10, 0.20}.

### E09 Robustness v6 (code/run_robustness_v6.py; results/robustness_v6)
Unit/basin folds, no-twin, causal features, calibration phase, tidal, m, cross-version (stale / recal / native).

### E10 Semi-synthetic shift v7 (code/run_shift_v7.py; results/shift_v7)
Width and latitude shifts; unweighted vs estimated and oracle density-ratio weights (weighted SCoRE, GPU port).

### E11 SWORD v17b and support (code/download_range.py, sword_support.py, sword_domain_v1.py, sword_domain_v2.py)
curl resume restarted the file from zero after a dropped connection; replaced by a verified 6-part range downloader
(MD5 OK). Support: AUC 0.80 (location) / 0.79 (physical); continent-wide ESS 132 / 188 of 1,795; domain within the
99th percentile of gauged weights covers about 76% of ungauged reaches (ESS 530 / 571); transferability map.

### E12 Real labelled shifts (code/real_shift_v8.py location weights, real_shift_v9.py physical weights)
US -> Canada, Canada -> US, south -> north of 55 N, east -> west of 110 W. Location weights: no overlap, no certification.
Physical weights (width, elevation, slope, flow accumulation, length, channels, obstruction, ice fractions, stream order,
distance to outlet): empirical SDR control in all evaluated settings (estimated weights).

### E13 Brazil external set (code/amazon_step1.py, amazon_match_qc.py, amazon_tz_check.py, amazon_tz_check2.py,
amazon_download_full.py, swot_reach_download.py, pair_brazil.py, gauge_qc_brazil.py, external_test_v1.py, external_test_v2.py)
AmazonWSE data not released. 4,666 ANA fluviometric telemetric stations; 1,829 within 1 km of a SWORD river node.
ANA timestamps verified as Brasilia time (UTC-3). Download redesigned after timing tests (yearly requests, about 10x
faster). 110,207 reach pairs from 1,585 gauges; gauge-only QC (1,065 stations pass). ANA basin codes verified.

### E14 Final frozen pipeline, 14 Oct (code/final_chain.sh, tune_score.py, final_main.py; results/final)
Tuning on Version C (grouped CV): depth 8, learning rate 0.03, 700 trees, min child weight 10 (log-loss 0.3698 vs
0.3742); monotone constraints rejected. Pre-final results archived in results/archive_prefinal. Main results (Version D,
tau 0.25): S6 homo reach 0.47 / 0.62 / 0.100 (kept / coverage / realised SDR) at alpha 0.10 and 0.73 / 0.79 / 0.196 at
0.20; node 0.42 / 0.56 / 0.098 and 0.62 / 0.73 / 0.198. Stop rule 1: 0 of 144 S6 configurations flagged. Validity curve
within 0.01 of alpha for alpha 0.05-0.30. Baselines: flags and USGS-style grid infeasible at 0.10 and 0.255-0.259 at
0.20; pooled conformal 0.141-0.149 / 0.256-0.271; S1 valid but 10-15 points less retention. tau 0.50: 0.80 / 0.94 kept
(reach). Label A: nodes certifiable at 0.50 m and 1 m; reaches not (reach-mean vs point-gauge offset on sloping rivers).
Re-runs with the tuned model: robustness 4/176 flagged (all C -> D stale); real shifts unweighted 11/16 fail, physical
weights 16/16 empirical SDR control; Brazil unweighted 23/56 fail (reach tau 0.25: 1/14; 0.50: 10/14; 0.10: 1/14; node 0.25: 11/14),
physical weights 56/56 empirical SDR control (estimated weights, not an exact guarantee); diverse training inconclusive.

### E15 Reviewer checks (code/final_extras.py; results/final/extras_*)
Named filters: Confluence Permissive Relaxed (approx.) 0.57 kept, site error 0.268; composite strict filter 0.56, 0.291.
Temporal holdout (calibrate < 2025, test 2025): alpha 0.10 valid, alpha 0.20 0.207-0.208 (marginal). Conditional:
narrow nodes and far-north reaches above alpha by 1-3 points; extreme flows worse among kept observations. 'hete' valid
but less efficient. Seed stability: reach decisions identical over 50 seeds; 13.5% of nodes flip. Runtime: reference
SCoRE 0.71 s vs GPU about 1 ms (650-700x); XGBoost fit 1.6-1.7 s.

### E16 Group-conditional calibration (code/final_mondrian.py; results/final/mondrian_*)
Groups: prior width class, width x north. Overall valid in all schemes. Narrow nodes 0.113 / 0.233 -> 0.096 / 0.193;
far-north reaches 0.118 / 0.222 -> 0.101 / 0.185 (width x north). Retention cost 2-9 points.

## 3. Decisions log
| Date | Decision | Reason |
|---|---|---|
| 13 Oct | Label B with fixed per-site offset primary; label A secondary | datum-free; labels independent of screening |
| 13 Oct | Cleaning rules v1 (BOR excluded, one row per overpass, PGC0 over PIC0) | daily values; duplicate series |
| 13 Oct | Headline tau 0.25 m, alpha 0.20, frontier reported | user decision |
| 13 Oct | Site-level selective target (SDR over certified reaches), engine SCoRE | pooled and abstain-as-zero fail |
| 13 Oct | C3 twin recertification moved to discussion (stop rule 4) | no label-cost advantage |
| 13 Oct | GPU-bound code; long runs under nohup | user preference |
| 13 Oct | Physical (not location) weights for transfer | location weights give no overlap |
| 14 Oct | ANA timestamps read as UTC-3 | tz checks |
| 14 Oct | Tuned model from Version C; monotone constraints rejected | disjoint tuning; worse log-loss |
| 14 Oct | Lakes excluded from the paper (future work) | BOR daily values; missing lake prior |

## 4. Corrections log
- Brazil (E13 -> E14): the earlier conclusion that unweighted transfer is valid after station QC held only for reach,
  tau 0.25 with the pre-final model; final results show unweighted failures in 23/56 configurations. Final claim:
  transfer to new regions requires physical weights (valid 72/72 regional and continental tests).
- S1 (E05 -> E14): in the final design S1 is valid; the abstention-as-zero failure is shown by S3 (prototype).
- Diverse training: no consistent gain; the earlier reading that the score is the bottleneck is not supported.

## 5. Limitations to report
Marginal guarantee (subgroups can exceed alpha unless group-conditional calibration is used); extreme flows worse among
kept observations; temporal holdout 0.207-0.208 at alpha 0.20 (E15); prospective offsets up to 0.238 without drift-aware calibration (E17) and below alpha with it (E18); randomised certification at node level;
transfer to new regions needs physical weights and keeps less; product-version change needs recalibration; Brazil
Amazon stratum small and dominated by southern tributaries and hydropower sites; ANA telemetry is provisional.

## 6. How to reproduce
1. Raw data in data/raw (verify: `python3 code/verify_usgs_paired.py`).
2. `python3 code/clean_v1.py` (cleaned table and crosswalk).
3. Brazil set: amazon_step1.py, amazon_match_qc.py, swot_reach_download.py, amazon_download_full.py, pair_brazil.py,
   gauge_qc_brazil.py, brazil_nodes.py.
4. `sh code/final_chain.sh` (tuning, final main, robustness, shifts, Brazil, node), then `python3 code/final_extras.py`,
   `python3 code/final_mondrian.py`, `python3 code/final_condense.py`.

## 7. E17 Audit response (14 Oct; code/audit_response.py, group_sdr_convention.py, final_mondrian.py VERSION=C)
Prompted by an external log-based audit (SWOT_Experimental_Audit_Final_Recommendations.pdf). Documents:
docs/AUDIT_RESPONSE.md, docs/GUARANTEE_AND_DEFINITIONS.md; outputs results/audit_response/, results/final/mondrian_C/.
- Leakage matrix: 0 shared gauges, units, features and SWOT observations between training, calibration and test in
  all 20 splits (reach, node); overpasses shared as expected.
- Prospective-offset sensitivity (temporal design): offsets from pre-2025 data only give realised SDR 0.119 / 0.238
  (reach) and 0.121 / 0.224 (node) at alpha 0.10 / 0.20, versus 0.103 / 0.209 and 0.108 / 0.203 with full-record
  offsets. The guarantee concerns same-period bias-adjusted error; prospective use with a fixed earlier offset
  overshoots by 2-4 points.
- Weight diagnostics: AUC 0.92-0.98 between calibration and evaluation sites; ESS 24-145; dominance explains where
  certification collapses (Canada -> US, Amazon).
- Brazil QC flow table, match-distance and pairing-gap quantiles; support map relabelled as covariate support.
- E16 pattern reproduced on Version C (supportive only: Version C labels were used for tuning); group SDR recomputed under the SDR convention (0 when nothing certified): width x north
  has no group above alpha beyond 2 SE in either version; far north achieved largely by abstention.
- Main table with SDs and paired retention differences (results/audit_response/G_main_table.md).

## 8. Corrections added after the audit
- Group-level SDR in E15/E16 tables was a conditional average over splits with certification; replaced by the SDR
  convention (conditional values kept alongside).
- "Per-group guarantees imply the overall one" (final_mondrian.py docstring) withdrawn; overall validity is empirical.
- Counts: 4,666 unique fluviometric codes (4,964 rows); 242 stations without usable gauge data (log said 240).
- Support-map classes renamed from "likely certifiable / partial / abstention likely" to covariate-support classes.
- Scope: label B results are retrospective same-period bias-adjusted accuracy, not prospective or absolute accuracy.

## 9. E18 Prospective protocol (14 Oct; code/prospective_test.py; results/audit_response/H_prospective_*)
Purpose: support a prospective-use claim. Causal features; test = held-out units, 2025 observations labelled against
offsets from data before 2025-01-01; calibration either naive (same-period labels before 2025) or drift-aware
(offsets before 2024, labels for 2024). tau 0.25 m, 20 splits, Version D.
Results (realised SDR / kept): naive reach 0.115 / 0.343 and 0.219 / 0.666, node 0.106 / 0.325 and 0.198 / 0.539
(alpha 0.10 and 0.20); drift-aware reach 0.090 / 0.186 and 0.184 / 0.562, node 0.056 / 0.130 and 0.165 / 0.453.
Decision: the paper reports two modes. Retrospective screening carries the SCoRE guarantee; prospective deployment uses
causal features with drift-aware calibration and is empirically valid in time-split tests (not theorem-backed).

## 10. Pre-writing corrections (14 Oct 2026; code/apply_prewriting_fixes.py)
Prompted by SWOT_PreWriting_Guide.pdf. Wording only, no new experiments: prospective protocol recorded as completed;
transfer results stated as empirical SDR control with estimated weights; temporal analyses E15 / E17 / E18 reported
separately; Version C results supportive (Version C used for tuning); canonical term "within-site bias-adjusted WSE
error"; support-map ratio and thresholds defined (GUARANTEE_AND_DEFINITIONS.md Sections 7-11). Outputs frozen
(docs/FREEZE_MANIFEST_2026-10-14.tsv; frozen/code_freeze_2026-10-14.tar.gz). Experimentation closed.

## Appendix A. Environment (generated 2026-10-15 05:27 UTC)
- pandas 2.2.3
- numpy 1.26.4
- sklearn 1.6.1
- xgboost 2.1.3
- cupy 13.3.0
- torch 2.7.0a0+7c8ec84dab.nv25.03
- pyarrow 19.0.1
- h5py 3.16.0
- scipy 1.15.2
- NVIDIA-SMI 570.124.06             Driver Version: 570.124.06     CUDA Version: 12.8
- 0  NVIDIA B200                    On  |   00000000:C3:00.0 Off |                   On

## Appendix B. Code inventory (code/)
| File | Lines | Modified | Summary |
|---|---|---|---|
| access_tests.py | 28 | 2026-10-13 10:43 UTC (16:13 IST) | Access tests for the Amazon external set: (1) Hydrocron SWOT RiverSP Version D time series (open API), |
| after_sword_download.sh | 7 | 2026-10-13 09:39 UTC (15:09 IST) | Verified range download of SWORD v17b (netCDF), then the support analysis; each step runs only if the previous passed. |
| amazon_download.py | 136 | 2026-10-13 14:38 UTC (20:08 IST) | External set bulk download (13 Oct 2026). Usage: python3 amazon_download.py [amazon/brazil] |
| amazon_download_full.py | 61 | 2026-10-13 15:58 UTC (21:28 IST) | External set, full-record ANA download (13 Oct 2026). Replaces the overpass-window method after timing tests: |
| amazon_match_qc.py | 71 | 2026-10-13 14:38 UTC (20:08 IST) | Match QC for ANA station -> SWORD reach matches (13 Oct 2026). |
| amazon_step1.py | 125 | 2026-10-13 10:49 UTC (16:19 IST) | Amazon / Brazil external set, step 1 (13 Oct 2026): ANA telemetric stations -> SWORD v17b river nodes -> SWOT D coverage. |
| amazon_tz_check.py | 115 | 2026-10-13 11:00 UTC (16:30 IST) | Empirical time-zone check of ANA legacy telemetry timestamps (13 Oct 2026). |
| amazon_tz_check2.py | 78 | 2026-10-13 15:58 UTC (21:28 IST) | ANA time-zone discrimination, check 2 (13 Oct 2026): Brasilia time (UTC-3) everywhere, or local legal time? |
| amazonwse_check.py | 28 | 2026-10-13 10:42 UTC (16:12 IST) | Inspect the AmazonWSE repository via the GitHub API: licence, README, top-level and data-related files. |
| ana_sample.py | 17 | 2026-10-13 10:43 UTC (16:13 IST) | Parse a sample of ANA legacy telemetry (Rio Negro at Manaus, 14990000): fields, time step, stage values. |
| ana_timing.py | 10 | 2026-10-13 15:56 UTC (21:26 IST) | Time ANA legacy telemetry requests of different window lengths for one station (Rio Negro at Manaus). |
| ana_timing2.py | 15 | 2026-10-13 15:57 UTC (21:27 IST) | Time long ANA legacy telemetry windows (90 and 365 days) for one station; check the record span returned. |
| apply_prewriting_fixes.py | 194 | 2026-10-15 05:27 UTC (10:57 IST) | Apply the pre-writing documentation fixes (14 Oct 2026) and freeze outputs before manuscript drafting. |
| audit_data_v0.py | 178 | 2026-10-13 06:14 UTC (11:44 IST) | Data audit v0 for the USGS paired SWOT-gage release (Harlan et al. 2026, doi:10.5066/P1FE9W9E). |
| audit_data_v1.py | 191 | 2026-10-13 06:19 UTC (11:49 IST) | Data audit v1: duplicates, >1 h pairs, C/D re-match, and decoding + reproducing the USGS filter grid. |
| audit_data_v1b.py | 99 | 2026-10-13 06:22 UTC (11:52 IST) | Audit v1b: loose ends from v1 before fixing cleaning rules. |
| audit_headers.py | 22 | 2026-10-13 06:09 UTC (11:39 IST) | Print row count, column count and header of every CSV in the USGS release, |
| audit_response.py | 236 | 2026-10-14 09:55 UTC (15:25 IST) | Audit response (14 Oct 2026): checks requested by the external experimental audit. GPU-bound where models are fitted. |
| brazil_nodes.py | 99 | 2026-10-14 06:09 UTC (11:39 IST) | Brazil node-level pairs (14 Oct 2026): Hydrocron SWOT_L2_HR_RiverSP_D node series for each station's matched node, |
| chain_amazon.sh | 7 | 2026-10-13 11:00 UTC (16:30 IST) | Wait for amazon_step1 (station matching + Hydrocron counts) to finish, then run the ANA time-zone check. |
| chain_amazon2.sh | 13 | 2026-10-13 14:38 UTC (20:08 IST) | External-set chain (13 Oct 2026): match QC -> Amazon download -> time-zone check 2 -> rest-of-Brazil download. |
| chain_amazon3.sh | 11 | 2026-10-13 15:58 UTC (21:28 IST) | External-set chain v3 (13 Oct 2026): SWOT reach series for all Brazilian matches -> full-record ANA download |
| chain_overnight.sh | 12 | 2026-10-13 16:21 UTC (21:51 IST) | Overnight chain (13-14 Oct 2026): waits for chain_amazon3 (ANA download + time-zone check 2), then |
| chain_v8.sh | 6 | 2026-10-13 10:17 UTC (15:47 IST) | Certifiable domain (fast), then the real-shift experiment; logs per step. |
| check_failure_mode.py | 194 | 2026-10-13 06:42 UTC (12:12 IST) | Failure-mode check (13 Oct 2026): does tuning a screening rule without site-level separation |
| clean_v1.py | 166 | 2026-10-13 06:26 UTC (11:56 IST) | Freeze the cleaned analysis table v1 (rules agreed 13 Oct 2026). Raw data are only read. |
| condense_v6.py | 42 | 2026-10-13 09:01 UTC (14:31 IST) | Condense results/robustness_v6/summary.csv into compact tables (retention / site coverage / realised SDR). |
| digest_metadata.py | 66 | 2026-10-13 06:07 UTC (11:37 IST) | Digest the FGDC metadata XML of the USGS paired SWOT-gage release. |
| download_range.py | 65 | 2026-10-13 09:39 UTC (15:09 IST) | Resumable, verified parallel download of one Zenodo file (written 13 Oct 2026 after curl restarted from zero). |
| external_diverse.py | 90 | 2026-10-14 06:09 UTC (11:39 IST) | Diverse-training test (14 Oct 2026), GPU-bound: does adding Brazilian reaches to score training help on held-out |
| external_test_v1.py | 182 | 2026-10-14 06:09 UTC (11:39 IST) | External test v1 (13-14 Oct 2026), GPU-bound: does a guarantee built on North America hold on Brazilian reaches? |
| external_test_v2.py | 135 | 2026-10-14 06:09 UTC (11:39 IST) | External test v2 (14 Oct 2026): sensitivity of the North America -> Brazil result to station-match and gauge QC. |
| features_v2.py | 78 | 2026-10-13 08:35 UTC (14:05 IST) | Extended label-free features (v2) for the screening score. Everything is computed from SWOT data only |
| final_chain.sh | 28 | 2026-10-14 06:10 UTC (11:40 IST) | Final experiment chain (14 Oct 2026). Pre-final results are archived; the node download runs in parallel (network); |
| final_condense.py | 44 | 2026-10-14 08:46 UTC (14:16 IST) | Condense all final results into results/final/condensed_all.txt (14 Oct 2026). |
| final_extras.py | 211 | 2026-10-14 09:09 UTC (14:39 IST) | Final extras (14 Oct 2026), GPU-bound, tuned model, same splits/seeds as final_main.py. Version D, tau 0.25. |
| final_main.py | 175 | 2026-10-14 06:10 UTC (11:40 IST) | Final main run (14 Oct 2026), GPU-bound, frozen settings (tuned model from results/final/model_config.json). |
| final_mondrian.py | 113 | 2026-10-14 10:09 UTC (15:39 IST) | Group-conditional (Mondrian-type) calibration (14 Oct 2026), GPU-bound, tuned model, same splits/seeds as final_main. |
| gauge_qc_brazil.py | 31 | 2026-10-14 05:54 UTC (11:24 IST) | Gauge-only QC for ANA stations (14 Oct 2026): checks on the 15-min stage records alone, no SWOT data involved. |
| gpu_check.py | 19 | 2026-10-13 08:04 UTC (13:34 IST) | Report GPUs and GPU-capable Python libraries on the DGX. |
| group_sdr_convention.py | 30 | 2026-10-14 10:08 UTC (15:38 IST) | Recompute group-level realised SDR under the SDR convention (0 when a group certifies nothing in a split), alongside the |
| inspect_sword_vars.py | 16 | 2026-10-13 10:27 UTC (15:57 IST) | Inspect SWORD v17b reach variables needed for physical (non-geographic) weight covariates. |
| lib_check.py | 9 | 2026-10-13 09:22 UTC (14:52 IST) | Check readers needed for SWORD netCDF and for maps. |
| make_log_appendix.py | 90 | 2026-10-14 09:29 UTC (14:59 IST) | Build the automatic appendix of the experiment log and assemble docs/EXPERIMENT_LOG.md (part 1 + appendix). |
| mondrian_c_summary.py | 10 | 2026-10-14 10:07 UTC (15:37 IST) | Print the Version C group-conditional (E16 confirmation) summary. |
| pair_brazil.py | 122 | 2026-10-13 16:19 UTC (21:49 IST) | Build Brazil SWOT-gauge pairs (13 Oct 2026). Usage: python3 pair_brazil.py [limit] |
| prospective_test.py | 93 | 2026-10-14 10:13 UTC (15:43 IST) | Prospective protocol test (14 Oct 2026), GPU-bound, tuned model, same unit splits/seeds as final_main. |
| prototype_score_sdr.py | 148 | 2026-10-13 07:23 UTC (12:53 IST) | Prototype v3 (13 Oct 2026): site-level selective deployment risk (SDR) via conformal risk-adjusted e-values + e-BH. |
| prototype_score_sdr_v3b.py | 97 | 2026-10-13 07:31 UTC (13:01 IST) | Prototype v3b: site-level SDR e-BH with a SINGLE threshold (v3's uniform mixture diluted the e-values; diagnosed 13 Oct). |
| prototype_score_sdr_v4.py | 111 | 2026-10-13 07:44 UTC (13:14 IST) | Prototype v4 (13 Oct 2026): site-level SDR with the authors' SCoRE implementation. |
| prototype_selective.py | 148 | 2026-10-13 07:09 UTC (12:39 IST) | Prototype v2 (13 Oct 2026): SELECTIVE site-level risk control. |
| prototype_site_level.py | 145 | 2026-10-13 06:58 UTC (12:28 IST) | Prototype (13 Oct 2026): what does a SITE-level guarantee cost in retention? |
| real_shift_v8.py | 129 | 2026-10-13 10:17 UTC (15:47 IST) | Real-shift experiment v8 (13 Oct 2026), GPU-bound: do site-level guarantees transfer across real, labelled shifts |
| real_shift_v9.py | 133 | 2026-10-13 10:28 UTC (15:58 IST) | Real-shift experiment v9 (13 Oct 2026): same designs, splits and seeds as v8, but density-ratio weights from PHYSICAL |
| run_frontier_v5.py | 139 | 2026-10-14 06:09 UTC (11:39 IST) | Frontier run v5 (13 Oct 2026), GPU-bound: SCoRE-SDR at the site level vs pooled conformal selection. |
| run_robustness_v6.py | 220 | 2026-10-13 08:35 UTC (14:05 IST) | Robustness run v6 (13 Oct 2026), GPU-bound. tau = 0.25 m, alpha in {0.10, 0.20}; S6 = SCoRE-SDR at the site level |
| run_shift_v7.py | 141 | 2026-10-13 09:08 UTC (14:38 IST) | Shift experiment v7 (13 Oct 2026), GPU-bound: does the site-level guarantee survive a gauged -> ungauged covariate shift, |
| score_sdr_gpu.py | 191 | 2026-10-13 09:05 UTC (14:35 IST) | GPU (PyTorch) port of SCoRE_SDR e-values (Bai & Jin 2026; github.com/Tian-Bai/SCoRE, MIT, commit 401b7ca). |
| sword_domain_v1.py | 71 | 2026-10-13 10:17 UTC (15:47 IST) | Certifiable domain v1 (13 Oct 2026): restrict the ungauged target to reaches inside gauged support and recompute |
| sword_domain_v2.py | 77 | 2026-10-13 10:39 UTC (16:09 IST) | Certifiable domain v2 (13 Oct 2026): support of ungauged North American river reaches with PHYSICAL covariates |
| sword_list.py | 8 | 2026-10-13 08:33 UTC (14:03 IST) | List the files of the SWORD v17b Zenodo record (15299138) with sizes and checksums. |
| sword_support.py | 95 | 2026-10-13 09:29 UTC (14:59 IST) | SWORD v17b support analysis (13 Oct 2026): how far are ungauged North American river reaches from the gauged ones? |
| swot_reach_download.py | 13 | 2026-10-13 15:58 UTC (21:28 IST) | Hydrocron SWOT D reach series for every matched Brazilian reach (skips reaches already saved). |
| swotio.py | 62 | 2026-10-13 06:26 UTC (11:56 IST) | Shared loader for the USGS paired SWOT-gage release (Harlan et al. 2026, doi:10.5066/P1FE9W9E). |
| tune_score.py | 78 | 2026-10-14 06:09 UTC (11:39 IST) | Score-model tuning (14 Oct 2026): grouped 5-fold CV over river units on Version C (reach and node), so tuning data are |
| verify_bacia.py | 10 | 2026-10-13 16:21 UTC (21:51 IST) | Verify ANA basin codes: most frequent rivers and median location per Bacia code among matched stations. |
| verify_usgs_paired.py | 55 | 2026-10-13 06:06 UTC (11:36 IST) | Verify the USGS paired SWOT-gage release against the ScienceBase record. |

## Appendix C. Results inventory (results/)

**amazon_v1/** (11 files, 2 MB)
- ana_stations.csv: 1 MB, 2026-10-13 10:49 UTC (16:19 IST)
- gauge_qc.csv: 84 KB, 2026-10-14 05:54 UTC (11:24 IST)
- matches.csv: 560 KB, 2026-10-13 11:25 UTC (16:55 IST)
- matches_qc.csv: 649 KB, 2026-10-13 14:38 UTC (20:08 IST)
- qc_report.txt: 8 KB, 2026-10-13 14:38 UTC (20:08 IST)
- qc_sample.csv: 5 KB, 2026-10-13 14:38 UTC (20:08 IST)
- report.txt: 2 KB, 2026-10-13 11:25 UTC (16:55 IST)
- tz_check.csv: 2 KB, 2026-10-13 11:47 UTC (17:17 IST)
- tz_check2.csv: 2 KB, 2026-10-13 18:38 UTC (00:08 IST)
- tz_check2_report.txt: 2 KB, 2026-10-13 18:38 UTC (00:08 IST)
- tz_check_report.txt: 2 KB, 2026-10-13 11:47 UTC (17:17 IST)

**archive_prefinal/** (20 files, 1 MB)
- external_v1/per_split.csv: 135 KB, 2026-10-14 06:10 UTC (11:40 IST)
- external_v1/report.txt: 9 KB, 2026-10-14 06:10 UTC (11:40 IST)
- external_v1/summary.csv: 8 KB, 2026-10-14 06:10 UTC (11:40 IST)
- external_v2/per_split.csv: 87 KB, 2026-10-14 06:10 UTC (11:40 IST)
- external_v2/report.txt: 6 KB, 2026-10-14 06:10 UTC (11:40 IST)
- external_v2/summary.csv: 5 KB, 2026-10-14 06:10 UTC (11:40 IST)
- frontier_v5/per_split.csv: 285 KB, 2026-10-14 06:10 UTC (11:40 IST)
- frontier_v5/report.txt: 18 KB, 2026-10-14 06:10 UTC (11:40 IST)
- frontier_v5/summary.csv: 15 KB, 2026-10-14 06:10 UTC (11:40 IST)
- real_shift_v9/per_split.csv: 120 KB, 2026-10-14 06:10 UTC (11:40 IST)
- real_shift_v9/report.txt: 7 KB, 2026-10-14 06:10 UTC (11:40 IST)
- real_shift_v9/summary.csv: 7 KB, 2026-10-14 06:10 UTC (11:40 IST)
- robustness_v6/condensed.txt: 5 KB, 2026-10-14 06:10 UTC (11:40 IST)
- robustness_v6/per_split.csv: 498 KB, 2026-10-14 06:10 UTC (11:40 IST)
- robustness_v6/report.txt: 26 KB, 2026-10-14 06:10 UTC (11:40 IST)
- robustness_v6/stoprule1.csv: 20 KB, 2026-10-14 06:10 UTC (11:40 IST)
- robustness_v6/summary.csv: 32 KB, 2026-10-14 06:10 UTC (11:40 IST)
- shift_v7/per_split.csv: 222 KB, 2026-10-14 06:10 UTC (11:40 IST)
- shift_v7/report.txt: 10 KB, 2026-10-14 06:10 UTC (11:40 IST)
- shift_v7/summary.csv: 12 KB, 2026-10-14 06:10 UTC (11:40 IST)

**audit_response/** (15 files, 3 MB)
- A_leakage_matrix.csv: 460 B, 2026-10-14 09:55 UTC (15:25 IST)
- B_prospective_offset.csv: 1 KB, 2026-10-14 10:03 UTC (15:33 IST)
- B_prospective_offset_per_split.csv: 21 KB, 2026-10-14 10:03 UTC (15:33 IST)
- C_weight_diagnostics.csv: 3 KB, 2026-10-14 09:55 UTC (15:25 IST)
- C_weight_diagnostics_per_rep.csv: 14 KB, 2026-10-14 09:55 UTC (15:25 IST)
- D_brazil_qc_flow.md: 920 B, 2026-10-14 09:55 UTC (15:25 IST)
- E_support_classes.csv: 3 MB, 2026-10-14 09:55 UTC (15:25 IST)
- F_group_sdr_convention.csv: 54 KB, 2026-10-14 10:08 UTC (15:38 IST)
- F_group_sdr_convention.md: 8 KB, 2026-10-14 10:08 UTC (15:38 IST)
- G_main_table.md: 2 KB, 2026-10-14 09:55 UTC (15:25 IST)
- H_prospective_per_split.csv: 18 KB, 2026-10-14 10:20 UTC (15:50 IST)
- H_prospective_report.txt: 1 KB, 2026-10-14 10:20 UTC (15:50 IST)
- H_prospective_summary.csv: 1 KB, 2026-10-14 10:20 UTC (15:50 IST)
- report.txt: 8 KB, 2026-10-14 10:03 UTC (15:33 IST)
- support_map_physical.png: 279 KB, 2026-10-14 09:55 UTC (15:25 IST)

**audit_v0/** (5 files, 15 KB)
- audit_report.txt: 12 KB, 2026-10-13 06:14 UTC (11:44 IST)
- err_A_by_type_version.csv: 301 B, 2026-10-13 06:14 UTC (11:44 IST)
- err_B_by_qflag.csv: 2 KB, 2026-10-13 06:14 UTC (11:44 IST)
- err_B_by_type_version.csv: 424 B, 2026-10-13 06:14 UTC (11:44 IST)
- err_U_by_type_version.csv: 419 B, 2026-10-13 06:14 UTC (11:44 IST)

**audit_v1/** (7 files, 29 MB)
- audit_v1_report.txt: 23 KB, 2026-10-13 06:20 UTC (11:50 IST)
- audit_v1b_report.txt: 3 KB, 2026-10-13 06:23 UTC (11:53 IST)
- cd_matched_overpasses.csv: 25 MB, 2026-10-13 06:19 UTC (11:49 IST)
- duplicate_groups.csv: 4 MB, 2026-10-13 06:19 UTC (11:49 IST)
- grid_directions.csv: 3 KB, 2026-10-13 06:19 UTC (11:49 IST)
- grid_reproduction_single_factor.csv: 31 KB, 2026-10-13 06:20 UTC (11:50 IST)
- grid_variants.csv: 25 KB, 2026-10-13 06:23 UTC (11:53 IST)

**external_diverse/** (3 files, 44 KB)
- per_split.csv: 39 KB, 2026-10-14 08:41 UTC (14:11 IST)
- report.txt: 3 KB, 2026-10-14 08:41 UTC (14:11 IST)
- summary.csv: 2 KB, 2026-10-14 08:41 UTC (14:11 IST)

**external_v1/** (3 files, 152 KB)
- per_split.csv: 135 KB, 2026-10-13 18:41 UTC (00:11 IST)
- report.txt: 9 KB, 2026-10-13 18:41 UTC (00:11 IST)
- summary.csv: 8 KB, 2026-10-13 18:41 UTC (00:11 IST)

**external_v2/** (3 files, 136 KB)
- per_split.csv: 120 KB, 2026-10-14 08:27 UTC (13:57 IST)
- report.txt: 8 KB, 2026-10-14 08:27 UTC (13:57 IST)
- summary.csv: 7 KB, 2026-10-14 08:27 UTC (13:57 IST)

**external_v2_node/** (3 files, 140 KB)
- per_split.csv: 125 KB, 2026-10-14 08:43 UTC (14:13 IST)
- report.txt: 8 KB, 2026-10-14 08:43 UTC (14:13 IST)
- summary.csv: 7 KB, 2026-10-14 08:43 UTC (14:13 IST)

**external_v2_tau0.1/** (3 files, 112 KB)
- per_split.csv: 97 KB, 2026-10-14 08:32 UTC (14:02 IST)
- report.txt: 8 KB, 2026-10-14 08:32 UTC (14:02 IST)
- summary.csv: 7 KB, 2026-10-14 08:32 UTC (14:02 IST)

**external_v2_tau0.5/** (3 files, 153 KB)
- per_split.csv: 137 KB, 2026-10-14 08:30 UTC (14:00 IST)
- report.txt: 8 KB, 2026-10-14 08:30 UTC (14:00 IST)
- summary.csv: 8 KB, 2026-10-14 08:30 UTC (14:00 IST)

**failure_check_v1/** (3 files, 55 KB)
- per_split.csv: 48 KB, 2026-10-13 06:52 UTC (12:22 IST)
- report.txt: 4 KB, 2026-10-13 06:52 UTC (12:22 IST)
- summary.csv: 3 KB, 2026-10-13 06:52 UTC (12:22 IST)

**final/** (26 files, 2 MB)
- condensed_all.txt: 22 KB, 2026-10-14 08:46 UTC (14:16 IST)
- extras_conditional.csv: 146 KB, 2026-10-14 09:17 UTC (14:47 IST)
- extras_filters.csv: 6 KB, 2026-10-14 09:17 UTC (14:47 IST)
- extras_report.txt: 361 B, 2026-10-14 09:17 UTC (14:47 IST)
- extras_runtime.csv: 268 B, 2026-10-14 09:17 UTC (14:47 IST)
- extras_seeds.csv: 3 KB, 2026-10-14 09:17 UTC (14:47 IST)
- extras_summary.txt: 8 KB, 2026-10-14 09:17 UTC (14:47 IST)
- extras_temporal.csv: 24 KB, 2026-10-14 09:17 UTC (14:47 IST)
- labelA_per_split.csv: 79 KB, 2026-10-14 06:54 UTC (12:24 IST)
- labelA_summary.csv: 6 KB, 2026-10-14 06:54 UTC (12:24 IST)
- main_per_split.csv: 1006 KB, 2026-10-14 06:42 UTC (12:12 IST)
- main_report.txt: 7 KB, 2026-10-14 06:54 UTC (12:24 IST)
- main_summary.csv: 68 KB, 2026-10-14 06:42 UTC (12:12 IST)
- model_config.json: 250 B, 2026-10-14 06:15 UTC (11:45 IST)
- mondrian_C/mondrian_groups.csv: 20 KB, 2026-10-14 10:07 UTC (15:37 IST)
- mondrian_C/mondrian_groups_per_split.csv: 281 KB, 2026-10-14 10:07 UTC (15:37 IST)
- mondrian_C/mondrian_overall.csv: 2 KB, 2026-10-14 10:07 UTC (15:37 IST)
- mondrian_C/mondrian_per_split.csv: 30 KB, 2026-10-14 10:07 UTC (15:37 IST)
- mondrian_C/mondrian_report.txt: 11 KB, 2026-10-14 10:07 UTC (15:37 IST)
- mondrian_groups.csv: 20 KB, 2026-10-14 09:25 UTC (14:55 IST)
- mondrian_groups_per_split.csv: 281 KB, 2026-10-14 09:25 UTC (14:55 IST)
- mondrian_overall.csv: 2 KB, 2026-10-14 09:25 UTC (14:55 IST)
- mondrian_per_split.csv: 30 KB, 2026-10-14 09:25 UTC (14:55 IST)
- mondrian_report.txt: 11 KB, 2026-10-14 09:25 UTC (14:55 IST)
- monotone.csv: 223 B, 2026-10-14 06:15 UTC (11:45 IST)
- tuning.csv: 3 KB, 2026-10-14 06:14 UTC (11:44 IST)

**frontier_v5/** (3 files, 319 KB)
- per_split.csv: 285 KB, 2026-10-13 08:25 UTC (13:55 IST)
- report.txt: 18 KB, 2026-10-13 08:25 UTC (13:55 IST)
- summary.csv: 15 KB, 2026-10-13 08:25 UTC (13:55 IST)

**real_shift_v8/** (3 files, 111 KB)
- per_split.csv: 99 KB, 2026-10-13 10:24 UTC (15:54 IST)
- report.txt: 6 KB, 2026-10-13 10:24 UTC (15:54 IST)
- summary.csv: 6 KB, 2026-10-13 10:24 UTC (15:54 IST)

**real_shift_v9/** (3 files, 134 KB)
- per_split.csv: 120 KB, 2026-10-14 08:25 UTC (13:55 IST)
- report.txt: 7 KB, 2026-10-14 08:25 UTC (13:55 IST)
- summary.csv: 7 KB, 2026-10-14 08:25 UTC (13:55 IST)

**robustness_v6/** (5 files, 582 KB)
- condensed.txt: 5 KB, 2026-10-13 09:01 UTC (14:31 IST)
- per_split.csv: 499 KB, 2026-10-14 07:50 UTC (13:20 IST)
- report.txt: 26 KB, 2026-10-14 07:50 UTC (13:20 IST)
- stoprule1.csv: 20 KB, 2026-10-14 07:50 UTC (13:20 IST)
- summary.csv: 32 KB, 2026-10-14 07:50 UTC (13:20 IST)

**shift_v7/** (3 files, 243 KB)
- per_split.csv: 220 KB, 2026-10-14 08:09 UTC (13:39 IST)
- report.txt: 10 KB, 2026-10-14 08:09 UTC (13:39 IST)
- summary.csv: 12 KB, 2026-10-14 08:09 UTC (13:39 IST)

**site_level_proto_v1/** (3 files, 101 KB)
- per_split.csv: 90 KB, 2026-10-13 07:03 UTC (12:33 IST)
- report.txt: 6 KB, 2026-10-13 07:03 UTC (12:33 IST)
- summary.csv: 5 KB, 2026-10-13 07:03 UTC (12:33 IST)

**site_level_proto_v2/** (3 files, 116 KB)
- per_split.csv: 101 KB, 2026-10-13 07:14 UTC (12:44 IST)
- report.txt: 10 KB, 2026-10-13 07:14 UTC (12:44 IST)
- summary.csv: 6 KB, 2026-10-13 07:14 UTC (12:44 IST)

**site_level_proto_v3/** (6 files, 228 KB)
- per_split.csv: 87 KB, 2026-10-13 07:28 UTC (12:58 IST)
- per_split_with_v3b.csv: 117 KB, 2026-10-13 07:36 UTC (13:06 IST)
- report.txt: 7 KB, 2026-10-13 07:28 UTC (12:58 IST)
- report_v3b.txt: 8 KB, 2026-10-13 07:36 UTC (13:06 IST)
- summary.csv: 4 KB, 2026-10-13 07:28 UTC (12:58 IST)
- summary_with_v3b.csv: 6 KB, 2026-10-13 07:36 UTC (13:06 IST)

**site_level_proto_v4/** (3 files, 135 KB)
- per_split.csv: 122 KB, 2026-10-13 07:57 UTC (13:27 IST)
- report.txt: 8 KB, 2026-10-13 07:57 UTC (13:27 IST)
- summary.csv: 6 KB, 2026-10-13 07:57 UTC (13:27 IST)

**sword_domain_v1/** (3 files, 2 MB)
- domain_map.png: 272 KB, 2026-10-13 10:17 UTC (15:47 IST)
- domain_q99.csv: 2 MB, 2026-10-13 10:17 UTC (15:47 IST)
- report.txt: 455 B, 2026-10-13 10:17 UTC (15:47 IST)

**sword_domain_v2/** (3 files, 2 MB)
- domain_map_physical.png: 275 KB, 2026-10-13 10:39 UTC (16:09 IST)
- reach_support_physical.csv: 2 MB, 2026-10-13 10:39 UTC (16:09 IST)
- report.txt: 799 B, 2026-10-13 10:39 UTC (16:09 IST)

**sword_support_v1/** (3 files, 2 MB)
- reach_weights.csv: 2 MB, 2026-10-13 10:03 UTC (15:33 IST)
- report.txt: 2 KB, 2026-10-13 10:03 UTC (15:33 IST)
- weight_map.png: 275 KB, 2026-10-13 10:03 UTC (15:33 IST)

## Appendix D. Logs (logs/)
| Log | Size | Modified |
|---|---|---|
| access_tests.log | 2 KB | 2026-10-13 10:43 UTC (16:13 IST) |
| amazon_download_amazon.log | 440 B | 2026-10-13 15:56 UTC (21:26 IST) |
| amazon_download_full.log | 4 KB | 2026-10-13 18:37 UTC (00:07 IST) |
| amazon_match_qc.log | 8 KB | 2026-10-13 14:38 UTC (20:08 IST) |
| amazon_step1.log | 2 KB | 2026-10-13 11:25 UTC (16:55 IST) |
| amazon_tz_check.log | 2 KB | 2026-10-13 11:47 UTC (17:17 IST) |
| amazon_tz_check2.log | 2 KB | 2026-10-13 18:38 UTC (00:08 IST) |
| amazonwse_check.log | 262 B | 2026-10-13 10:42 UTC (16:12 IST) |
| audit_data_v0.log | 12 KB | 2026-10-13 06:14 UTC (11:44 IST) |
| audit_data_v1.log | 23 KB | 2026-10-13 06:20 UTC (11:50 IST) |
| audit_data_v1b.log | 3 KB | 2026-10-13 06:23 UTC (11:53 IST) |
| audit_response.log | 8 KB | 2026-10-14 10:07 UTC (15:37 IST) |
| brazil_nodes.log | 260 B | 2026-10-14 06:24 UTC (11:54 IST) |
| chain_amazon.log | 33 B | 2026-10-13 11:47 UTC (17:17 IST) |
| chain_amazon2.log | 40 B | 2026-10-13 14:38 UTC (20:08 IST) |
| chain_amazon3.log | 127 B | 2026-10-13 18:38 UTC (00:08 IST) |
| chain_overnight.log | 123 B | 2026-10-13 18:41 UTC (00:11 IST) |
| chain_v8.log | 33 B | 2026-10-13 10:24 UTC (15:54 IST) |
| check_failure_mode.log | 4 KB | 2026-10-13 06:52 UTC (12:22 IST) |
| clean_v1.log | 3 KB | 2026-10-13 06:27 UTC (11:57 IST) |
| external_diverse.log | 3 KB | 2026-10-14 08:41 UTC (14:11 IST) |
| external_test_v1.log | 9 KB | 2026-10-13 18:41 UTC (00:11 IST) |
| external_test_v2.log | 6 KB | 2026-10-14 05:56 UTC (11:26 IST) |
| external_v2_final.log | 8 KB | 2026-10-14 08:27 UTC (13:57 IST) |
| external_v2_node.log | 8 KB | 2026-10-14 08:43 UTC (14:13 IST) |
| external_v2_tau01.log | 8 KB | 2026-10-14 08:32 UTC (14:02 IST) |
| external_v2_tau05.log | 8 KB | 2026-10-14 08:30 UTC (14:00 IST) |
| final_chain.log | 214 B | 2026-10-14 08:43 UTC (14:13 IST) |
| final_extras.log | 383 B | 2026-10-14 09:17 UTC (14:47 IST) |
| final_main.log | 7 KB | 2026-10-14 06:54 UTC (12:24 IST) |
| final_mondrian.log | 11 KB | 2026-10-14 09:25 UTC (14:55 IST) |
| final_mondrian_C.log | 11 KB | 2026-10-14 10:07 UTC (15:37 IST) |
| gauge_qc_brazil.log | 92 B | 2026-10-14 05:54 UTC (11:24 IST) |
| pair_brazil.log | 510 B | 2026-10-13 18:40 UTC (00:10 IST) |
| prospective_test.log | 1 KB | 2026-10-14 10:20 UTC (15:50 IST) |
| prototype_score_sdr.log | 7 KB | 2026-10-13 07:28 UTC (12:58 IST) |
| prototype_score_sdr_v3b.log | 8 KB | 2026-10-13 07:36 UTC (13:06 IST) |
| prototype_score_sdr_v4.log | 8 KB | 2026-10-13 07:57 UTC (13:27 IST) |
| prototype_selective.log | 10 KB | 2026-10-13 07:14 UTC (12:44 IST) |
| prototype_site_level.log | 6 KB | 2026-10-13 07:03 UTC (12:33 IST) |
| real_shift_v8.log | 7 KB | 2026-10-13 10:24 UTC (15:54 IST) |
| real_shift_v9.log | 7 KB | 2026-10-13 10:35 UTC (16:05 IST) |
| real_shift_v9_final.log | 7 KB | 2026-10-14 08:25 UTC (13:55 IST) |
| run_frontier_v5.log | 18 KB | 2026-10-13 08:25 UTC (13:55 IST) |
| run_robustness_v6.log | 26 KB | 2026-10-13 09:00 UTC (14:30 IST) |
| run_robustness_v6_final.log | 26 KB | 2026-10-14 07:50 UTC (13:20 IST) |
| run_shift_v7.log | 10 KB | 2026-10-13 09:17 UTC (14:47 IST) |
| run_shift_v7_final.log | 10 KB | 2026-10-14 08:09 UTC (13:39 IST) |
| score_sdr_gpu_equivalence.log | 157 B | 2026-10-13 08:10 UTC (13:40 IST) |
| score_sdr_gpu_equivalence_v2.log | 233 B | 2026-10-13 09:07 UTC (14:37 IST) |
| sword_domain_v1.log | 1 KB | 2026-10-13 10:17 UTC (15:47 IST) |
| sword_domain_v2.log | 2 KB | 2026-10-13 10:39 UTC (16:09 IST) |
| sword_download.log | 88 B | 2026-10-13 09:33 UTC (15:03 IST) |
| sword_pipeline_v2.log | 2 KB | 2026-10-13 10:03 UTC (15:33 IST) |
| sword_support_v1.log | 22 B | 2026-10-13 09:29 UTC (14:59 IST) |
| swot_reach_download.log | 164 B | 2026-10-13 16:04 UTC (21:34 IST) |
| tune_score.log | 1 KB | 2026-10-14 06:15 UTC (11:45 IST) |
| verify_bacia.log | 2 KB | 2026-10-13 18:38 UTC (00:08 IST) |

## Appendix E. Raw data inventory (data/raw/)
- data/raw/ana_brazil/ana_stage: 104 files, 33 MB
- data/raw/ana_brazil/ana_stage_full: 3654 files, 461 MB
- data/raw/ana_brazil/swot_node_D: 1728 files, 43 MB
- data/raw/ana_brazil/swot_reach_D: 1330 files, 41 MB
- data/raw/sword_v17b: 3 files, 5 GB
- data/raw/usgs_paired: 9 files, 265 MB

## Appendix F. sha256 of key outputs
- data/processed/brazil_pairs_nodes_v1.parquet (13 MB): `8397396a1e6e35df665494de41f588f13b79143d2d575d0404ab90bb28713891`
- data/processed/brazil_pairs_smoke.parquet (257 KB): `5ac67ddd4d7ff1bcf3361c769e0c61c7ca5cc5291b668dea14fe16c370b8c230`
- data/processed/brazil_pairs_v1.parquet (20 MB): `b653984a26b9c4cd09fff1ff25778c017b797f71b72080696fc21ccf53edb8d9`
- data/processed/cd_crosswalk_v1.parquet (10 MB): `1e81abcf691b7b6c81d4da1bf7102253922e6c4ea94e714ebe2aceef5191071d`
- data/processed/paired_clean_v1.parquet (86 MB): `384b314fc130fabb2f34df0bb598345610a0a9eec1d424bef06a2806a321c3cc`
- docs/design_note_v0.6.md (27 KB): `8ee0ab12b04a2bcd04ddd4ef676f3560f11474b3d0e383c8197681fadec6a827`
- results/final/condensed_all.txt (22 KB): `bfb9b80b379d8d59f9fb0a9cee4738bfae4aecf18d5ad569f99b4e529b780781`
- results/final/labelA_summary.csv (6 KB): `58d78dbf0f015d75dd8cdb6bc94d6d7ff59d832b0012c14e0e5edafafc42fc76`
- results/final/main_summary.csv (68 KB): `9976c9c8d2e2005c2bad335f847ef41c86fa7f4d50536a9efcb81bb98c59b36f`
- results/final/model_config.json (250 B): `aabdcc6c6946585545a5c5038bd1843ea947f27f1f17ad63bce7e3f838f58ee3`

### E19 Single-site sensitivity (code/final_single_site.py; results/sensitivity_single_site/) - 16 Oct 2026
Purpose: v21 external audit item 2. In the main design the partition is drawn over river units and ~10% of sites
share a unit (reach 90 of 1,899 units, 188 of 1,997 sites; node 96 of 1,941 units, 201 of 2,046 sites), so SCoRE's
site-level exchangeability is approximate. E19 keeps only single-site units (reach 1,809 units/sites, 90,686 pairs;
node 1,845, 124,251 pairs), where unit and site exchangeability coincide. Frozen final_main functions imported
unchanged; same seed formulas; 4 x 5 splits; Version D, tau 0.25 m, alpha 0.10/0.20. Report pre-specified.
Results (retention / realized SDR, proposed 'homo'): reach 0.479/0.097 and 0.747/0.199; node 0.426/0.099 and
0.630/0.202 (marginal, not flagged). Flagged (excess > 2 SE): 0 of 8. Runtime 339 s. Main results unchanged.
