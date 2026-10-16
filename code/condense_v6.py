"""Condense results/robustness_v6/summary.csv into compact tables (retention / site coverage / realised SDR)."""
import pathlib
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
s = pd.read_csv(ROOT / "results" / "robustness_v6" / "summary.csv")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20); pd.set_option("display.max_rows", 200)
lines = []


def cell(r):
    return f"{r.retention:.2f}/{r.site_coverage:.2f}/{r.sdr:.3f}" + (f" n{r.share_none:.2f}" if r.share_none > 0 else "")


for a in (0.2, 0.1):
    sub = s[(s.method == "S6 SCoRE-SDR (homo)") & (s.alpha == a) & ((s.m == 3) | s.m.isna())].copy()
    sub["tv"] = sub.type + " " + sub.v
    sub["cell"] = sub.apply(cell, axis=1)
    lines.append(f"\n== S6 homo, m=3, alpha={a}: retention / site coverage / realised SDR (nX = share of splits certifying nothing)")
    lines.append(sub.pivot(index="design", columns="tv", values="cell").to_string())
m3 = s[(s.design == "unit/ext") & (s.method == "M3 pooled")].copy()
m3["tv"] = m3.type + " " + m3.v
m3["cell"] = m3.apply(cell, axis=1)
lines.append("\n== M3 pooled, unit/ext (realised site-level SDR for comparison)")
lines.append(m3.pivot(index="alpha", columns="tv", values="cell").to_string())
ms = s[(s.design == "unit/ext") & (s.method == "S6 SCoRE-SDR (homo)")].copy()
ms["tv"] = ms.type + " " + ms.v
ms["cell"] = ms.apply(cell, axis=1)
lines.append("\n== m sensitivity, unit/ext, S6 homo")
lines.append(ms.pivot_table(index=["alpha", "m"], columns="tv", values="cell", aggfunc="first").to_string())
nn = s[(s.design == "unit/ext") & (s.method == "S6 SCoRE-SDR (None)") & (s.m == 3)].copy()
nn["tv"] = nn.type + " " + nn.v
nn["cell"] = nn.apply(cell, axis=1)
lines.append("\n== S6 without pruning (None), unit/ext, m=3")
lines.append(nn.pivot(index="alpha", columns="tv", values="cell").to_string())
sr = pd.read_csv(ROOT / "results" / "robustness_v6" / "stoprule1.csv")
lines.append(f"\n== Stop rule 1: {int(sr.violation.sum())} of {len(sr)} S6 configurations flagged (excess > 2 SE)")
if sr.violation.any():
    lines.append(sr[sr.violation].to_string(index=False))
text = "\n".join(lines)
print(text)
(ROOT / "results" / "robustness_v6" / "condensed.txt").write_text(text, encoding="utf-8")
