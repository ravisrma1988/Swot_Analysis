"""Figs. 3 and 4 of the manuscript from the frozen plot tables written by results_digest.py.
Fig. 3: realized SDR and retention over the nominal level (Version D, tau 0.25 m), reach and node.
Fig. 4: unweighted versus physical-covariate weighted calibration in all 72 transfer configurations.
Outputs: results/manuscript/fig3_validity_frontier.pdf, results/manuscript/fig4_transfer.pdf"""
import pathlib
import pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
M = pathlib.Path(__file__).resolve().parents[1] / "results" / "manuscript"
plt.rcParams.update({"font.family": "serif", "font.size": 8})

d = pd.read_csv(M / "fig3_data.csv")
d["m"] = d.method.map({"S6 SCoRE-SDR (homo)": "P", "S1 site CRC (abstain = 0)": "S", "M3 pooled conformal": "M"})
col = {"P": "tab:blue", "S": "tab:green", "M": "tab:red"}
name = {"P": "Proposed", "S": "Site risk control", "M": "Pooled conformal"}
fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.75))
for m in ("P", "S", "M"):
    for t, ls, mk in (("reach", "-", "o"), ("node", "--", "s")):
        x = d[(d.m == m) & (d.type == t)].sort_values("alpha")
        for a, y in ((ax[0], "sdr"), (ax[1], "retention")):
            a.plot(x.alpha, x[y], ls=ls, marker=mk, ms=3, lw=1, color=col[m], label=f"{name[m]}, {t}")
ax[0].plot([0.03, 0.32], [0.03, 0.32], color="0.5", lw=0.8, ls=":", label="Realized SDR equal to $\\alpha$")
for a, yl, tt in ((ax[0], "Realized SDR", "(a)"), (ax[1], "Retention", "(b)")):
    a.set_xlabel(r"Nominal level $\alpha$"); a.set_ylabel(yl); a.set_title(tt, loc="left", fontsize=9)
    a.set_xticks([0.05, 0.1, 0.15, 0.2, 0.25, 0.3]); a.grid(alpha=0.3, lw=0.5)
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, -0.13), ncol=4, fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig(M / "fig3_validity_frontier.pdf", bbox_inches="tight", pad_inches=0.02)

t = pd.read_csv(M / "fig4_data.csv"); t["ex_u"] = t.sdr - t.alpha; t["ex_w"] = t.sdr_w - t.alpha
grp = [("regional", "reach", "Regional, reach", "tab:blue", "^"), ("regional", "node", "Regional, node", "tab:blue", "v"),
       ("Brazil", "reach", "Brazil, reach", "tab:green", "o"), ("Brazil", "node", "Brazil, node", "tab:orange", "s")]
fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.85))
for e, ty, lab, c, mk in grp:
    x = t[(t.experiment == e) & (t.type == ty)]
    for a, xx, yy in ((ax[0], "ex_u", "ex_w"), (ax[1], "retention", "retention_w")):
        a.scatter(x[xx], x[yy], s=14, marker=mk, facecolors="none" if ty == "node" else c, edgecolors=c, lw=0.8, label=lab)
ax[0].axhline(0, color="0.4", lw=0.8); ax[0].axvline(0, color="0.4", lw=0.8)
ax[0].set_xlabel("Unweighted SDR minus nominal level"); ax[0].set_ylabel("Weighted SDR minus nominal level")
ax[1].plot([0, 1], [0, 1], color="0.5", lw=0.8, ls=":"); ax[1].set_xlabel("Unweighted retention"); ax[1].set_ylabel("Weighted retention")
ax[0].set_title("(a)", loc="left", fontsize=9); ax[1].set_title("(b)", loc="left", fontsize=9)
for a in ax: a.grid(alpha=0.3, lw=0.5)
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, -0.07), ncol=4, fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig(M / "fig4_transfer.pdf", bbox_inches="tight", pad_inches=0.02)
print("Figs. 3 and 4 written to", M)
