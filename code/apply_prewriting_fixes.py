"""Apply the pre-writing documentation fixes (14 Oct 2026) and freeze outputs before manuscript drafting.
Run on the DGX:  cd /workspace/ravi/swot && python3 code/apply_prewriting_fixes.py

Every replacement is checked: if the old text is not found exactly, the fix is appended as an erratum instead, so the
documents are always corrected. A summary of what was replaced or appended is printed and saved.
Freeze: sha256 manifest of code, frozen results, processed data and docs (docs/FREEZE_MANIFEST_2026-10-14.tsv) and a
code snapshot (frozen/code_freeze_2026-10-14.tar.gz). Finally the experiment log is regenerated.
"""
import hashlib, pathlib, subprocess, sys, tarfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
REPORT = []


def patch(fname, old, new, erratum_title):
    p = DOCS / fname
    text = p.read_text(encoding="utf-8")
    if old in text:
        p.write_text(text.replace(old, new, 1), encoding="utf-8")
        REPORT.append(f"{fname}: replaced '{old[:60]}...'")
    else:
        p.write_text(text.rstrip() + f"\n\n### Erratum (14 Oct 2026): {erratum_title}\n{new}\n", encoding="utf-8")
        REPORT.append(f"{fname}: old text not found, appended erratum '{erratum_title}'")


def append(fname, block):
    p = DOCS / fname
    p.write_text(p.read_text(encoding="utf-8").rstrip() + "\n" + block, encoding="utf-8")
    REPORT.append(f"{fname}: appended section")


# ---------------------------------------------------------------- AUDIT_RESPONSE.md
patch("AUDIT_RESPONSE.md",
"""## What the paper must now say
- Primary guarantee: E[SDR] <= alpha for retrospective, same-period bias-adjusted error (level-change accuracy).
- Prospective use with an offset fixed from earlier data overshoots by 2-4 points at tau 0.25 m (offset drift,
  possibly including the 2025 switch from reprocessed to forward products). Mitigation (drift-aware calibration with
  labels defined against earlier-window offsets) is untested; state as limitation or test before submission.
- Transfer results: empirically valid in the evaluated configurations, with weight diagnostics reported.
- Group-conditional calibration (width x north): per-group validity confirmed on Versions D and C under the SDR
  convention; exploratory in design; in the far north it mostly abstains.

## Remaining before submission
- Look up the exact SCoRE theorem numbers (SDR e-BH guarantee; weighted version; robustness to estimated weights).
- Optional: drift-aware calibration test (about 10 GPU minutes) if prospective deployment is to be claimed.
- Literature re-check (stop rule 3).""",
"""## What the paper must now say
- Primary guarantee: E[SDR] <= alpha for retrospective screening with the same-period, within-site bias-adjusted
  WSE error (label B), under assumptions A1-A4.
- Prospective deployment: a drift-aware prospective protocol was evaluated using causal features, 2024 calibration
  labels constructed with lagged offsets, and held-out 2025 observations. It was empirically valid in all four tested
  reach/node x alpha settings, but is not theorem-backed because the calibration and deployment periods are not
  exactly exchangeable (addendum H). Without drift-aware calibration, offsets fixed from earlier data give realised
  SDR up to 0.238 at alpha 0.20 (E17).
- Transfer: estimated physical-covariate weighting yielded empirical SDR control in all 72 evaluated regional and
  continental configurations; no exact finite-sample guarantee with estimated weights; weight diagnostics reported.
- Group-conditional calibration (width x north): per-group SDR control in Version D under the SDR convention; the
  pattern was reproduced on Version C, which is supportive rather than independent confirmation because Version C
  labels were used for hyperparameter tuning; exploratory in design; in the far north it works largely through
  abstention (low certification and coverage).

## Remaining before submission
- Look up the exact SCoRE theorem numbers (SDR e-BH guarantee; weighted version; robustness to estimated weights).
- Literature re-check (stop rule 3).""",
"prospective protocol completed; transfer empirical; Version C supportive")
patch("AUDIT_RESPONSE.md", "exploratory; confirmed on Version C",
      "exploratory; pattern reproduced on Version C (supportive only: Version C labels used for tuning)",
      "Version C is supportive, not independent")

# ---------------------------------------------------------------- GUARANTEE_AND_DEFINITIONS.md
patch("GUARANTEE_AND_DEFINITIONS.md",
"""- E16 group definitions (width x north) were chosen after E15: exploratory mitigation; confirmed on Version C (see
  results/final/mondrian_C).""",
"""- E16 group definitions (width x north) were chosen after E15: exploratory mitigation; the pattern was reproduced on
  Version C (results/final/mondrian_C), which is supportive rather than independent because Version C labels were
  used for hyperparameter tuning.""",
"E16 on Version C is supportive")

append("GUARANTEE_AND_DEFINITIONS.md", """
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
""")

# ---------------------------------------------------------------- EXPERIMENT_LOG_part1.md
patch("EXPERIMENT_LOG_part1.md", "distance to outlet): valid.",
      "distance to outlet): empirical SDR control in all evaluated settings (estimated weights).", "E12 wording")
patch("EXPERIMENT_LOG_part1.md", "weights 16/16 valid;", "weights 16/16 empirical SDR control;", "E14 wording (regional)")
patch("EXPERIMENT_LOG_part1.md", "physical weights 56/56 valid;",
      "physical weights 56/56 empirical SDR control (estimated weights, not an exact guarantee);", "E14 wording (Brazil)")
patch("EXPERIMENT_LOG_part1.md", "forward-in-time overshoot of about one point at alpha 0.20;",
      "temporal holdout 0.207-0.208 at alpha 0.20 (E15); prospective offsets up to 0.238 without drift-aware "
      "calibration (E17) and below alpha with it (E18);", "temporal analyses separated")
patch("EXPERIMENT_LOG_part1.md", "E16 confirmed on Version C;",
      "E16 pattern reproduced on Version C (supportive only: Version C labels were used for tuning);",
      "Version C supportive")
append("EXPERIMENT_LOG_part1.md", """
## 10. Pre-writing corrections (14 Oct 2026; code/apply_prewriting_fixes.py)
Prompted by SWOT_PreWriting_Guide.pdf. Wording only, no new experiments: prospective protocol recorded as completed;
transfer results stated as empirical SDR control with estimated weights; temporal analyses E15 / E17 / E18 reported
separately; Version C results supportive (Version C used for tuning); canonical term "within-site bias-adjusted WSE
error"; support-map ratio and thresholds defined (GUARANTEE_AND_DEFINITIONS.md Sections 7-11). Outputs frozen
(docs/FREEZE_MANIFEST_2026-10-14.tsv; frozen/code_freeze_2026-10-14.tar.gz). Experimentation closed.
""")

# ---------------------------------------------------------------- design_note_v0.6.md
append("design_note_v0.6.md", """
## 15. Errata (14 Oct 2026, pre-writing)
- Section 12 "physical weights were valid in 72 of 72" and "56 of 56": read as "estimated physical-covariate weighting
  yielded empirical SDR control in all evaluated configurations"; no exact guarantee with estimated weights.
- Section 13 "forward-in-time overshoot of about one point": temporal results are separated in
  GUARANTEE_AND_DEFINITIONS.md Section 10 (E15 0.207-0.208; E17 up to 0.238; E18 below alpha).
- Section 14 and all Version C subgroup results: supportive, not independent (Version C used for tuning).
- Terminology: "within-site bias-adjusted WSE error" replaces "level-change accuracy".
""")

# ---------------------------------------------------------------- freeze
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


groups = [ROOT / "code", ROOT / "data" / "processed", ROOT / "docs"] + [
    ROOT / "results" / d for d in ("final", "audit_response", "external_v2", "external_v2_tau0.5", "external_v2_tau0.1",
                                    "external_v2_node", "external_diverse", "real_shift_v9", "real_shift_v8",
                                    "robustness_v6", "shift_v7", "sword_domain_v2", "sword_domain_v1", "sword_support_v1")]
manifest = DOCS / "FREEZE_MANIFEST_2026-10-14.tsv"
lines = ["path\tbytes\tsha256"]
for g in groups:
    if g.exists():
        for f in sorted(x for x in g.rglob("*") if x.is_file() and x != manifest):
            lines.append(f"{f.relative_to(ROOT)}\t{f.stat().st_size}\t{sha(f)}")
manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
frozen = ROOT / "frozen"; frozen.mkdir(exist_ok=True)
tgz = frozen / "code_freeze_2026-10-14.tar.gz"
with tarfile.open(tgz, "w:gz") as t:
    t.add(ROOT / "code", arcname="code")
REPORT.append(f"freeze: {len(lines) - 1} files in {manifest.name}; code snapshot {tgz.name} sha256 {sha(tgz)}")

# ---------------------------------------------------------------- regenerate the log
r = subprocess.run([sys.executable, str(ROOT / "code" / "make_log_appendix.py")], capture_output=True, text=True)
REPORT.append("log: " + (r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr.strip()[-200:]))
(DOCS / "PREWRITING_FIXES_REPORT.txt").write_text("\n".join(REPORT) + "\n", encoding="utf-8")
print("\n".join(REPORT))
