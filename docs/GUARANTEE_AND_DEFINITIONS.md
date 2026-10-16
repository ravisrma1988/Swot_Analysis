# Formal guarantee, assumptions and reporting definitions (Track B)

Version 1, 14 Oct 2026. Written in response to the external experimental audit (P0-A, P0-D, Section 5).
Theorem numbers in Bai and Jin (2026) are to be checked against the paper before citation; the statement below
follows the authors' definitions of selective deployment risk (SDR) and risk-adjusted e-values with e-BH.

## 1. Objects
- Unit of inference: a site j = (feature, gauge, product version); feature = SWOT reach or node.
- Observations i of site j: SWOT passes paired with the gauge within 1 h.
- Label (primary, B): e_ij = d_ij - median_i(d_ij), d_ij = WSE_ij - gauge stage_ij; the median uses every pair of site
  j in the analysis subset (full record, retrospective). Bad observation: |e_ij| > tau.
- Score: mu_ij = P(good) from a gradient-boosted classifier f trained on training units T only.
- Within-site rule: keep observations with mu_ij >= lambda; lambda chosen from out-of-fold predictions inside T only.
- Eligibility: site j is eligible if at least m = 3 observations are kept (depends on scores only).
- Site score: s_j = mean over kept observations of (1 - mu_ij). Site loss: L_j = share of kept observations that are bad,
  L_j in [0, 1].
- Certification: psi_j in {0, 1} from SCoRE-SDR (risk-adjusted e-values + e-BH) using calibration sites' (L, s) and
  test sites' s; gamma = alpha; pruning 'homo' (one uniform draw) unless stated.

## 2. Guarantee used
Under assumptions A1-A4, with f, lambda and m fixed independently of calibration and test labels,
  SDR = E[ sum_j L_j psi_j / max(1, sum_j psi_j) ] <= alpha,
the expectation being over calibration and test data (in our evaluation: the random partition of units) and over the
pruning draw. This is a statement about an expectation of the average within-site error over certified test sites.

It does NOT state: that every split, site, subgroup, flow condition or configuration has error <= alpha; an
observation-level false-discovery rate; absolute (datum-referenced) accuracy; validity under arbitrary shift.

Weighted version (SCoRE_SDR_w): exact validity assumes the true site-level density ratio w(x) = dP_target/dP_source.
Our weights are estimated; Bai and Jin report robustness to estimated weights, but no exact finite-sample guarantee is
claimed for our implementation. All transfer results are "empirically valid in the evaluated configurations".

Group-conditional calibration (E16): SCoRE applied within each pre-specified, label-free group gives the guarantee
within each group under within-group exchangeability. Overall validity is NOT implied in general (the overall ratio
weights groups by random certified counts); it was observed empirically. CORRECTION: design note v0.6 Section 14 and
the final_mondrian.py docstring said it is implied; that statement is withdrawn.

## 3. Assumptions
- A1 Exchangeability of calibration and test sites. In the evaluation it is created by a uniformly random partition of
  units of a fixed population (finite-population argument, conditional on all data; dependence between sites, e.g.
  shared passes, does not break it). For deployment to ungauged reaches it is an assumption; weights address covariate
  shift only.
- A2 Score model, hyperparameters, lambda, m and eligibility are fixed without using calibration or test labels.
  Hyperparameters were tuned on Version C without reference to Version D splits, so A2 holds for Version D; Version C
  results reuse their own labels for tuning (stated as mild optimism).
- A3 Bounded loss: L_j in [0, 1].
- A4 The label definition is the same function of each site's data for calibration and test sites (full-record
  offset). The offset never enters screening decisions.
- A5 (weighted) Known density ratio for exact validity; otherwise approximate.

## 4. Theorem-to-code map
| Quantity | Code |
|---|---|
| Label B | swotio.label_B |
| Score model f | run_frontier_v5.mu_good (XGBoost CUDA; tuned settings results/final/model_config.json) |
| lambda | run_robustness_v6.oof + choose_lam (training units only) |
| Eligibility, s_j, L_j | prototype_score_sdr.site_scores |
| e-values | score_sdr_gpu.sdr_evalues / sdr_evalues_w (bit-exact to SCoRE.SCoRE.SCoRE_SDR / _w; logs/score_sdr_gpu_equivalence_v2.log) |
| Pruning, e-BH | SCoRE._uniform_random, SCoRE.utility.eBH (authors' code, commit 401b7ca) |
| Splits | final_main.main and run_split (seeds 5000q + 17[node] + 3[C]; 100q + f) |
| Density ratio | run_shift_v7.ratio_weights (classifier odds x pool-size ratio; p clipped to [1e-4, 1 - 1e-4]) |
| Realised SDR, retention, coverage | prototype_site_level.evaluate |

## 5. Reporting definitions
- Retention: share of all test observations that are kept.
- Site coverage: share of test sites with at least one kept observation (= certified sites).
- Realised SDR (per split): mean over certified test sites of their bad share among kept observations; 0 if no site
  is certified (SDR convention). Reported as mean +- SD over splits.
- Pooled FDP: bad share among all kept test observations (observation-level; not the guaranteed quantity).
- Split: one of 4 repeats x 5 folds over units (20). Splits share units across repeats: SDs understate fully
  independent replication; differences between methods are paired on identical splits.
- Configuration (0 of 144): S6 x (reach, node) x (Version C, D) x tau {0.10, 0.25, 0.50} x alpha {0.05..0.30, 6 values}
  x pruning {None, homo} = 144. Diagnostic flag: mean realised SDR - alpha > 2 SD / sqrt(20). A pass is a diagnostic,
  not a proof; configurations are dependent.
- 16 regional tests: real_shift_v9, physical weights, 4 designs (US -> Canada, Canada -> US, south -> north of 55 N,
  east -> west of 110 W) x (reach, node) x alpha {0.10, 0.20}.
- 56 continental tests: external_v2 family (Brazil), physical weights, 7 strata x alpha {0.10, 0.20} x 4 settings
  (reach tau 0.10, 0.25, 0.50; node tau 0.25). "72 regional and continental transfer tests" = 16 + 56.
- Temporal holdout: report mean realised SDR with its SE over splits (alpha 0.20: 0.207 node, 0.208 reach; SD 0.011
  and 0.014; overshoot about 2-3 SE).
- Node seed flips: share of eligible nodes whose certification is not identical across 50 seeds (13.5%), i.e. the
  fraction that ever changes; mean pairwise Jaccard of certified sets 0.977-0.985; reaches: 0% and 1.000.
- Runtime: same DGX MIG slice; reference = authors' Python SCoRE_SDR on CPU; GPU port timed end-to-end per call
  (host-to-device transfer and result copy included, CUDA already warm); kernel-only time not measured.
- Version C tuning: objective log-loss; grouped 5-fold by unit; Version D data not used. Efficiency effect on Version D
  is negligible (untuned frontier_v5: reach 0.465 / 0.729 retention at alpha 0.10 / 0.20; tuned: 0.465 / 0.725).
- Percentages vs fractions: tables use fractions with three decimals; text uses percentages with the same precision.
- tau must accompany every number; tau 0.25 m is the headline.

## 6. Scope statements for the manuscript
- Two deployment modes, reported separately:
  (1) Retrospective screening: label B with same-period offsets, full-record features; guarantee E[SDR] <= alpha under
      A1-A4 (random-partition exchangeability).
  (2) Prospective deployment: causal features; test labels against offsets estimated before deployment; drift-aware
      calibration (calibration labels built with the same lag). Empirically valid in time-split tests
      (results/audit_response/H_prospective_summary.csv); not theorem-backed, because calibration and deployment
      periods are not exactly exchangeable. Naive forward use (same-period calibration) can exceed alpha.
- Neither mode certifies absolute (datum-referenced) WSE accuracy.
- Support maps show estimated covariate support, not certified accuracy.
- E16 group definitions (width x north) were chosen after E15: exploratory mitigation; the pattern was reproduced on
  Version C (results/final/mondrian_C), which is supportive rather than independent because Version C labels were
  used for hyperparameter tuning.
- Main pipeline features use each site's whole SWOT record (retrospective screening); the causal feature set covers
  near-real-time use.

## 7. Four-mode claim framework (use in every section, caption and abstract)
| Mode | What can be claimed | What must not be claimed | Wording |
|---|---|---|---|
| Retrospective screening | E[SDR] <= alpha under A1-A4 for the same-period, within-site bias-adjusted WSE error | absolute (datum-referenced) accuracy; per-site or per-split guarantee; validity under arbitrary shift | "Retrospective screening controls expected site-level SDR under A1-A4 for the same-period bias-adjusted target." |
| Transfer with physical weights | empirical SDR control in the 72 evaluated regional (16) and continental (56) configurations | exact finite-sample guarantee with estimated weights | "Estimated physical-covariate weighting yielded empirical SDR control across all evaluated transfer configurations." |
| Prospective deployment | empirical validity of the drift-aware causal protocol in the 2024 -> 2025 time-split test | theorem-backed prospective guarantee | "Prospective deployment was empirically valid in the time-split experiment, but is not covered by the retrospective theorem." |
| Support map | overlap of ungauged reaches with the gauged physical-covariate domain | certification or prediction of WSE accuracy | "The map visualises estimated covariate support only; it does not certify WSE accuracy." |

## 8. Terminology (one term throughout)
- Canonical term for label B: "within-site bias-adjusted WSE error" (defined once in Methods as e_ij = d_ij - median_i d_ij,
  d_ij = SWOT WSE - gauge stage); add "same-period" in the retrospective mode.
- Do not use "level-change accuracy" or "absolute accuracy" for label B. Label A is "absolute (EGM2008) error".

## 9. Support-map statistic (Methods and caption)
- Populations: reference = 1,795 gauged SWORD v17b river reaches (Version D, science phase, non-tidal); target = 20,321
  ungauged North American SWORD v17b river reaches (type digit 1, lakeflag 0).
- Covariates z (physical, no location): log width, prior WSE, slope, log flow accumulation, log reach length, channel
  count, obstruction flag, ice fraction, full-ice fraction, stream order, log distance to outlet.
- Estimated density ratio of target to reference: w(z) = [P(ungauged | z) / P(gauged | z)] x (n_gauged / n_ungauged),
  from a 5-fold cross-fitted XGBoost classifier (out-of-fold probabilities clipped to [1e-6, 1 - 1e-6]).
- Plotted statistic for ungauged reach j: ratio_j = w_j / sum over gauged reaches of w_i, i.e. the weight reach j would
  carry relative to the total gauged weight if added as a test site in weighted SCoRE. Large values mean little
  comparable gauged evidence (the reach would dominate the calibration mass).
- Classes: <= 0.01 within support (16,684; 82.1%), 0.01-0.1 weak support (2,817; 13.9%), > 0.1 outside support (820;
  4.0%); total 20,321. Thresholds are descriptive, set from the real-shift diagnostics (median test-weight /
  calibration-total ratio about 0.01 where weighted certification retained data, 0.2 or more where it mostly
  abstained); not calibrated decision rules. The denominator uses all gauged reaches, which is optimistic relative to
  a calibration subset.
- Caption ending: "Classes summarise estimated overlap in physical-covariate space relative to the gauged reference
  set. They are diagnostic of feature support only and do not constitute certification of water-level accuracy."

## 10. Temporal analyses: report separately
| Analysis | Design | Result |
|---|---|---|
| E15 temporal holdout | calibrate < 2025, test 2025 at held-out units; full-record offsets; full-record features | alpha 0.10: 0.101-0.102; alpha 0.20: 0.207-0.208 (about 2-3 SE above) |
| E17 prospective-offset sensitivity | as E15, test labels against pre-2025 offsets | up to 0.119-0.121 (alpha 0.10) and 0.224-0.238 (alpha 0.20) |
| E18 drift-aware prospective protocol | causal features; calibration on 2024 labels with pre-2024 offsets; test 2025 vs pre-2025 offsets | reach 0.090 / 0.184, node 0.056 / 0.165 (alpha 0.10 / 0.20); kept reach 0.19 / 0.56, node 0.13 / 0.45 |

## 11. Reporting guardrails
1. Attach tau to every performance number (headline tau = 0.25 m).
2. Keep retention, site coverage, realised SDR and pooled FDP distinct.
3. "0 of 144" is a diagnostic robustness result, not a proof across configurations.
4. Width x north grouping was selected after E15: exploratory, even though reproduced on Version C.
5. Far-north SDR is achieved partly through abstention (low certification and coverage); never "high performance
   everywhere".
6. Node seed sensitivity: report 13.5% ever-flip together with Jaccard 0.977-0.985.
7. GPU speed: authors' Python SCoRE_SDR on CPU vs the verified GPU port; GPU timing includes host-device transfer with a
   warm CUDA context.
8. Version D is the primary evaluation; Version C results are supportive.
