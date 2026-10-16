"""Condense all final results into results/final/condensed_all.txt (14 Oct 2026)."""
import pathlib
import pandas as pd

R = pathlib.Path(__file__).resolve().parents[1] / "results"
L = []
f = lambda r: f"{r.retention:.2f}/{r.site_coverage:.2f}/{r.sdr:.3f}" + (f" n{r.share_none:.2f}" if r.get("share_none", 0) > 0 else "")

s = pd.read_csv(R / "final" / "main_summary.csv")
v = s[(s.method == "S6 SCoRE-SDR (homo)") & (s.tau == 0.25)]
L.append("== validity curve, S6 homo, tau 0.25 (realised SDR by alpha)")
L.append(v.pivot_table(index="alpha", columns=["type", "v"], values="sdr").round(3).to_string())
m = s[(s.method == "S6 SCoRE-SDR (homo)") & (s.v == "D") & (s.alpha.isin([0.1, 0.2]))].copy(); m["c"] = m.apply(f, axis=1)
L.append("\n== S6 homo, Version D, tau frontier (kept/coverage/SDR)")
L.append(m.pivot_table(index=["tau", "alpha"], columns="type", values="c", aggfunc="first").to_string())
a = pd.read_csv(R / "final" / "labelA_summary.csv"); a["c"] = a.apply(f, axis=1)
L.append("\n== label A (absolute), Version D")
L.append(a.pivot_table(index=["tau", "alpha"], columns=["type", "method"], values="c", aggfunc="first").to_string())

sr = pd.read_csv(R / "robustness_v6" / "stoprule1.csv")
L.append("\n== robustness stop rule 1 flagged:\n" + sr[sr.violation].to_string(index=False))
rb = pd.read_csv(R / "robustness_v6" / "summary.csv")
rb = rb[(rb.method == "S6 SCoRE-SDR (homo)") & ((rb.m == 3) | rb.m.isna())].copy(); rb["c"] = rb.apply(f, axis=1)
L.append("\n== robustness, S6 homo m=3")
L.append(rb.pivot_table(index=["design"], columns=["type", "v", "alpha"], values="c", aggfunc="first").to_string())

for name, keys in (("shift_v7", ["type", "shift", "b", "alpha", "method"]), ("real_shift_v9", ["type", "design", "alpha", "method"])):
    x = pd.read_csv(R / name / "summary.csv"); x["c"] = x.apply(f, axis=1)
    L.append(f"\n== {name} (kept/coverage/SDR; excess_over_2se > 0 = violation)")
    L.append(x[keys + ["c", "excess_over_2se"]].round(3).to_string(index=False))

for d in ("external_v2", "external_v2_tau0.5", "external_v2_tau0.1", "external_v2_node"):
    p = R / d / "summary.csv"
    if not p.exists():
        L.append(f"\n== {d}: missing"); continue
    x = pd.read_csv(p); x["c"] = x.apply(f, axis=1)
    L.append(f"\n== {d}")
    L.append(x.pivot_table(index=["stratum", "alpha"], columns="method", values="c", aggfunc="first").to_string())
    L.append("violations: " + str(x[x.excess_over_2se > 0][["stratum", "alpha", "method", "sdr"]].round(3).values.tolist()))
x = pd.read_csv(R / "external_diverse" / "summary.csv"); x["c"] = x.apply(f, axis=1)
L.append("\n== diverse training")
L.append(x.pivot_table(index=["stratum", "alpha"], columns=["score", "calibration"], values="c", aggfunc="first").to_string())
(R / "final" / "condensed_all.txt").write_text("\n".join(L), encoding="utf-8")
print("written", len("\n".join(L)), "chars")
