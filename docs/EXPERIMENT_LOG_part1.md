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
