"""Recompute group-level realised SDR under the SDR convention (0 when a group certifies nothing in a split), alongside the
conditional average (only splits where the group certified something) and the certification frequency.
Inputs: results/final/mondrian_groups_per_split.csv (Version D), results/final/mondrian_C/mondrian_groups_per_split.csv,
results/final/extras_conditional.csv (E15). Output: results/audit_response/F_group_sdr_convention.csv and .md"""
import pathlib
import numpy as np
import pandas as pd

R = pathlib.Path(__file__).resolve().parents[1] / "results"
frames = []
for v, p in (("D", R / "final" / "mondrian_groups_per_split.csv"), ("C", R / "final" / "mondrian_C" / "mondrian_groups_per_split.csv")):
    x = pd.read_csv(p); x["version"] = v; frames.append(x)
e = pd.read_csv(R / "final" / "extras_conditional.csv"); e = e[e.level == "site"].copy()
e["version"], e["scheme"] = "D", "global (E15)"; frames.append(e)
g = pd.concat(frames, ignore_index=True)
g["certified_any"] = g.coverage > 0
g["sdr_conv"] = np.where(g.certified_any, g.sdr, 0.0)
keys = ["version", "type", "alpha", "attribute", "group", "scheme"]
s = g.groupby(keys).agg(n_splits=("sdr_conv", "size"), cert_freq=("certified_any", "mean"), coverage=("coverage", "mean"),
                         sdr_convention=("sdr_conv", "mean"), sdr_sd=("sdr_conv", "std"), sdr_conditional=("sdr", "mean"))
s["excess_over_2se"] = s.sdr_convention - s.index.get_level_values("alpha") - 2 * s.sdr_sd / np.sqrt(s.n_splits)
out = R / "audit_response"; s.to_csv(out / "F_group_sdr_convention.csv")
focus = s.reset_index()
focus = focus[((focus.attribute == "width") & (focus.group == "<100 m")) | ((focus.attribute == "latitude") & (focus.group == ">=55N"))]
pd.set_option("display.width", 220)
txt = focus.round(3).to_string(index=False)
(out / "F_group_sdr_convention.md").write_text(txt, encoding="utf-8")
print(txt)
print("\nany group/scheme with excess_over_2se > 0 (SDR convention):")
print(s[s.excess_over_2se > 0].round(3).to_string() if (s.excess_over_2se > 0).any() else "none")
