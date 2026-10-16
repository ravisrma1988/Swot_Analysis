"""Read-only reconciliation for the final audit: configuration counts, flag-grid precision, Eq. (8) versus Fig. 5 data."""
import pathlib, subprocess
import numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]; R = ROOT / "results"
def sh(c): return subprocess.run(c, shell=True, capture_output=True, text=True, cwd=ROOT).stdout.strip()
m = pd.read_csv(R / "final" / "main_summary.csv")
s6 = m[m.method.str.startswith("S6")]
print(f"B3 main: S6 configurations {len(s6)} (types {s6.type.nunique()} x versions {s6.v.nunique()} x taus {s6.tau.nunique()} x alphas {s6.alpha.nunique()} x variants {s6.method.nunique()}); flagged {int((s6.excess_over_2se > 0).sum())}")
rb = pd.read_csv(R / "robustness_v6" / "summary.csv"); r6 = rb[rb.method.str.startswith("S6")].copy()
print("  robustness columns:", list(rb.columns))
if "excess_over_2se" not in r6.columns:
    sdc = next(c for c in ("sdr_sd", "sd", "site_fdp_kept_sd") if c in r6.columns); nc = next(c for c in ("n", "splits") if c in r6.columns)
    r6["excess_over_2se"] = (r6.sdr - r6.alpha) - 2 * r6[sdc] / np.sqrt(r6[nc])
print(f"B3 robustness: S6 configurations {len(r6)}; flagged {int((r6.excess_over_2se > 0).sum())}; flagged designs {sorted(r6[r6.excess_over_2se > 0].design.unique())}")
f4 = pd.read_csv(R / "manuscript" / "fig4_data.csv")
print(f"B3 transfer: configurations {len(f4)} (regional {int((f4.experiment == 'regional').sum())}, Brazil {int((f4.experiment == 'Brazil').sum())}); "
      f"unweighted flagged {int((f4.excess_over_2se > 0).sum())}, weighted flagged {int((f4.excess_over_2se_w > 0).sum())}; "
      f"weighted point estimates above alpha {int((f4.sdr_w > f4.alpha).sum())}")
ps = pd.read_csv(R / "final" / "main_per_split.csv")
g = ps[(ps.type == "reach") & (ps.v == "D") & (ps.tau == 0.25) & (ps.alpha == 0.10) & ps.method.str.startswith("M1")]
print(f"B4 flag grid reach a=0.10: splits {len(g)}, mean retention {g.retention.mean():.5f}, splits with retention > 0 {int((g.retention > 0).sum())}, "
      f"mean realized SDR {g.site_fdp_kept.mean():.4f}, splits with SDR > 0 {int((g.site_fdp_kept > 0).sum())}")
u = pd.read_csv(R / "audit_response" / "E_support_classes.csv")
den = (u.w / u.ratio).replace([np.inf, -np.inf], np.nan).dropna()
print(f"P3 Eq.(8): implied denominator w/ratio min {den.min():.6g} max {den.max():.6g} (constant => ratio = w / common sum)")
print(sh("grep -rn -E 'ratio *=|w_gauged|w\\[g|sum\\(' code/audit_response*.py code/support_map*.py 2>/dev/null | grep -i -E 'ratio|sum' | head -6"))
