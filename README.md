# Site-Level Selective Risk Control for SWOT River Water Surface Elevations

Code and frozen results for the manuscript

> R. Sharma, G. Rathi, R. R. Upadhyay and V. Gupta, "Site-level selective risk control for screening SWOT river water
> surface elevations and its transferability across regions and time," manuscript submitted for publication, 2026.

The study screens SWOT river water surface elevations (WSE) at the level of river sites. A gradient-boosted score,
trained on gauge-derived labels but using only SWOT and SWORD predictors, ranks observations within each site, and the
SCoRE selective risk procedure certifies sites so that the expected share of erroneous observations among certified
sites stays below a chosen level. Every number, table and figure in the manuscript can be traced to a file in `results/`.

## Repository layout

| Path | Content |
| --- | --- |
| `code/` | All scripts of the study (E00 to E19 in the experiment log) |
| `results/` | Frozen result tables behind every number in the paper |
| `results/final/` | Frozen main pipeline (Table 1 and Supplementary Table S1, Fig. 3, seeds, filters, runtime) |
| `results/manuscript/` | Plot tables and figure files of the manuscript |
| `results/sensitivity_single_site/` | E19 single-site sensitivity, with `MANIFEST.sha256` |
| `docs/EXPERIMENT_LOG.md` | Dated log of every experiment, decision and frozen output |
| `docs/FREEZE_MANIFEST_2026-10-14.tsv` | SHA-256 checksums of the frozen outputs |
| `docs/GUARANTEE_AND_DEFINITIONS.md` | Definitions, claim strength and terminology used in the paper |
| `figures_source/fig2_pipeline_source.tex` | TikZ source of Fig. 2 |

## Data (not redistributed)

All inputs are public, but third-party data are not redistributed here. Download them into `data/raw/` as described in
the experiment log (entries E00, E11 and E13).

| Data | Source | Used for |
| --- | --- | --- |
| Paired SWOT and gauge data, United States and Canada, 2023 to 2025 | USGS data release, doi 10.5066/P1FE9W9E | Main analysis; verify files with `code/verify_usgs_paired.py` against `docs/usgs_paired_MANIFEST.tsv` |
| SWOT River Database (SWORD) v17b | Zenodo, doi 10.5281/zenodo.15299138 | Physical attributes, transfer weights, support map |
| Brazilian gauge stage records | ANA telemetry service, http://telemetriaws1.ana.gov.br/ServiceANA.asmx | External test (`code/amazon_download_full.py`) |
| SWOT Version D reach and node series for Brazil | PO.DAAC Hydrocron, https://podaac.github.io/hydrocron/ | External test (`code/swot_reach_download.py`, `code/brazil_nodes.py`) |

Expected layout: `data/raw/usgs_paired/`, `data/raw/sword_v17b/`, `data/raw/ana_brazil/`, and `data/processed/` for the
cleaned tables written by the pipeline.

## Environment

- Python 3.12.3 in the NVIDIA NGC PyTorch container 25.03, one NVIDIA B200 partition, CUDA 12
- Packages in `requirements.txt` (numpy 1.26.4, pandas 2.2.3, pyarrow 19.0.1, scikit-learn 1.6.1, scipy 1.15.2,
  xgboost 2.1.3, cupy-cuda12x 13.3.0, torch 2.7.0a0 from NGC 25.03, matplotlib 3.10.1, requests 2.32.3, h5py)
- SCoRE reference code: clone https://github.com/Tian-Bai/SCoRE into `external/SCoRE` and check out commit `401b7ca`.
  `code/score_sdr_gpu.py` reimplements its e-values on the GPU and reuses its boosting and e-BH steps unchanged.

## Reproducing the paper

Run the scripts from the repository root in this order. The experiment log gives the exact commands and settings.

1. Data audit and cleaning: `audit_data_v1.py`, `clean_v1.py` (E01 to E03)
2. Brazilian external set: `amazon_step1.py`, `amazon_match_qc.py`, `amazon_tz_check.py`, `amazon_tz_check2.py`,
   `gauge_qc_brazil.py`, `pair_brazil.py`, `brazil_nodes.py` (E13)
3. SWORD attributes and support: `sword_support.py`, `sword_domain_v1.py`, `sword_domain_v2.py` (E11)
4. Frozen main pipeline: `bash code/final_chain.sh` (`tune_score.py`, `final_main.py`, `final_extras.py`,
   `final_mondrian.py`, `final_condense.py`; E14 to E16)
5. Robustness and shift: `run_robustness_v6.py`, `run_shift_v7.py`, `real_shift_v8.py`, `real_shift_v9.py`,
   `external_test_v2.py`, `external_diverse.py` (E09 to E13)
6. Prospective and audit analyses: `prospective_test.py`, `audit_response.py`, `group_sdr_convention.py`
7. Single-site sensitivity: `final_single_site.py` (E19)
8. Tables and figures: `table1_counts.py`, `fig1_study_area.py`, `results_digest.py`, `fig3_fig4.py`, `fig5_support_map.py`

## Where each result comes from

| Manuscript item | Script | Output |
| --- | --- | --- |
| Supplementary Table S3 | `table1_counts.py` | `results/manuscript/table1_counts.txt` |
| Table 1, Fig. 3 | `final_main.py`, `fig3_fig4.py` | `results/final/main_summary.csv`, `results/manuscript/fig3_data.csv` |
| Table 2 | `run_robustness_v6.py`, `final_extras.py`, `prospective_test.py`, `audit_response.py`, `final_single_site.py` | `results/robustness_v6/`, `results/final/extras_temporal.csv`, `results/audit_response/`, `results/sensitivity_single_site/` |
| Table 3, Fig. 4 | `external_test_v2.py`, `real_shift_v9.py`, `fig3_fig4.py` | `results/external_v2*/`, `results/real_shift_v9/`, `results/manuscript/fig4_data.csv` |
| Supplementary Table S1 | `final_mondrian.py`, `group_sdr_convention.py` | `results/final/mondrian_*`, `results/audit_response/F_*` |
| Fig. 1 | `fig1_study_area.py` | `results/manuscript/fig1_study_area.pdf` |
| Fig. 2 | TikZ | `figures_source/fig2_pipeline_source.tex` |
| Fig. 5 | `audit_response.py`, `fig5_support_map.py` | `results/audit_response/E_support_classes.csv`, `results/manuscript/fig5_support_map.pdf` |
| Configuration counts and checks | `audit_checks.py`, `final_reconcile.py` | printed reports |

## Not included

- Raw and processed third-party data (see Data above)
- `results/audit_v1/cd_matched_overpasses.csv`, which contains raw SWOT observation rows
- Superseded pre-final results and internal working notes

## License

No license is granted. All rights are reserved by the authors. The repository is shared so that the results of the
manuscript can be inspected and reproduced. For any other use, please contact the corresponding author.

## Contact

Rashmi Rathi Upadhyay (corresponding author), SRM University-AP, rashmirathi.u@srmap.edu.in
Ravi Sharma (code), Graphic Era (Deemed to be University), ravisharma.cse@geu.ac.in
