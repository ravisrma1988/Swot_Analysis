
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
