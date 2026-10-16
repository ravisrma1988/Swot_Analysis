"""Results digest for Section VI (read-only on frozen outputs) and Fig. 5 support map.
Prints compact tables; writes plot data to results/manuscript/{fig3_data.csv, fig4_data.csv}; renders fig5_support_map."""
import pathlib
import numpy as np
import pandas as pd
R = pathlib.Path(__file__).resolve().parents[1] / "results"; M = R / "manuscript"; M.mkdir(exist_ok=True)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)
P = lambda title, df: print(f"\n== {title}\n" + df.to_string(index=False))

m = pd.read_csv(R / "final" / "main_summary.csv")
d = m[(m.v == "D") & (m.tau == 0.25) & (m.alpha.isin([0.1, 0.2]) | m.alpha.isna())]
P("main D tau0.25", d[["type", "alpha", "method", "retention", "site_coverage", "pooled_fdp", "sdr", "sdr_sd"]].round(3))
f = pd.read_csv(R / "final" / "extras_filters.csv").groupby("method")[["retention", "site_coverage", "fdp", "site_fdp_kept"]].mean().round(3)
print("\n== named filters (reach D)\n" + f.to_string())
print("\n== paired differences\n" + (R / "audit_response" / "G_main_table.md").read_text().split("Paired")[1][:900])
v = m[(m.v == "D") & (m.tau == 0.25) & m.method.isin(["S6 SCoRE-SDR (homo)", "S1 site CRC (abstain = 0)", "M3 pooled conformal"])]
v[["type", "alpha", "method", "retention", "sdr", "sdr_sd"]].to_csv(M / "fig3_data.csv", index=False)
fr = m[(m.v == "D") & (m.method == "S6 SCoRE-SDR (homo)") & m.alpha.isin([0.1, 0.2])]
P("tau frontier S6 homo D", fr[["type", "tau", "alpha", "retention", "site_coverage", "sdr"]].round(3))
P("Version C S6 homo tau0.25", m[(m.v == "C") & (m.tau == 0.25) & (m.method == "S6 SCoRE-SDR (homo)") & m.alpha.isin([0.1, 0.2])][["type", "alpha", "retention", "sdr"]].round(3))
rb = pd.read_csv(R / "robustness_v6" / "summary.csv")
rb = rb[(rb.method == "S6 SCoRE-SDR (homo)")]
cols = [c for c in ("design", "type", "v", "alpha", "m", "retention", "sdr") if c in rb.columns]
P("robustness S6 homo", rb[cols].round(3))
t = pd.read_csv(R / "final" / "extras_temporal.csv")
P("temporal holdout", t.groupby(["type", "alpha", "method"]).agg(ret=("retention", "mean"), sdr=("site_fdp_kept", "mean"), sd=("site_fdp_kept", "std")).reset_index().round(3))
P("prospective offsets (E17)", pd.read_csv(R / "audit_response" / "B_prospective_offset.csv").round(3))
P("prospective protocol (E18)", pd.read_csv(R / "audit_response" / "H_prospective_summary.csv").round(3))
ss = pd.read_csv(R / "shift_v7" / "summary.csv")
P("semi-synthetic", ss[["type", "shift", "b", "alpha", "method", "retention", "sdr", "excess_over_2se"]].round(3))
rows = []
for name, path, tau, kind in [("regional", R / "real_shift_v9" / "summary.csv", 0.25, "reach/node")] + \
        [("Brazil", R / dd / "summary.csv", tt, ty) for dd, tt, ty in (("external_v2", 0.25, "reach"), ("external_v2_tau0.5", 0.5, "reach"),
                                                                      ("external_v2_tau0.1", 0.1, "reach"), ("external_v2_node", 0.25, "node"))]:
    x = pd.read_csv(path)
    key = "design" if "design" in x.columns else "stratum"
    x["experiment"], x["tau"] = name, tau
    if "type" not in x.columns:
        x["type"] = kind
    x["setting"] = x[key]
    u = x[x.method.str.contains("unweighted")].set_index(["type", "setting", "alpha"])
    w = x[x.method.str.contains("physical")].set_index(["type", "setting", "alpha"])
    j = u[["experiment", "tau", "retention", "sdr", "excess_over_2se"]].join(w[["retention", "sdr", "excess_over_2se"]], rsuffix="_w")
    rows.append(j.reset_index())
tr = pd.concat(rows); tr.to_csv(M / "fig4_data.csv", index=False)
print(f"\n== transfer configs {len(tr)}; unweighted flagged {int((tr.excess_over_2se > 0).sum())}; weighted flagged {int((tr.excess_over_2se_w > 0).sum())}")
b = pd.read_csv(R / "external_v2" / "summary.csv")
P("Brazil reach tau0.25", b[["stratum", "alpha", "method", "retention", "sdr", "excess_over_2se"]].round(3))
P("weight diagnostics", pd.read_csv(R / "audit_response" / "C_weight_diagnostics.csv")[["experiment", "type", "setting", "auc_cal_vs_eval", "ess_C", "n_C", "dominance_gt_0p1"]].round(3))
g = pd.read_csv(R / "audit_response" / "F_group_sdr_convention.csv")
g = g[(g.version == "D") & (((g.attribute == "width") & (g.group == "<100 m")) | ((g.attribute == "latitude") & (g.group == ">=55N")))]
P("subgroups D (SDR convention)", g[["type", "alpha", "attribute", "group", "scheme", "cert_freq", "coverage", "sdr_convention"]].round(3))
P("group-conditional overall D", pd.read_csv(R / "final" / "mondrian_overall.csv").round(3))
P("seed stability", pd.read_csv(R / "final" / "extras_seeds.csv").groupby(["type", "alpha"])[["always", "never", "sometimes", "mean_jaccard", "retention_sd"]].mean().reset_index().round(3))

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
u = pd.read_csv(R / "audit_response" / "E_support_classes.csv")
fig, ax = plt.subplots(figsize=(7.16, 3.6))
for k, c in (("within", "tab:blue"), ("weak", "tab:orange"), ("outside", "tab:red")):
    s = u[u.support.str.startswith(k)]
    lab = {"within": "within support", "weak": "weak support", "outside": "outside support"}[k]
    ax.scatter(s.x, s.y, s=0.25, c=c, linewidths=0, rasterized=True, label=f"{lab} ({len(s)}, {len(s) / len(u):.1%})")
ax.set_xlim(-170, -50); ax.set_ylim(5, 84); ax.set_aspect("equal")
ax.set_xlabel("Longitude (deg)", fontsize=8); ax.set_ylabel("Latitude (deg)", fontsize=8); ax.tick_params(labelsize=7)
ax.legend(loc="lower left", fontsize=7, markerscale=10, frameon=True)
fig.tight_layout(); fig.savefig(M / "fig5_support_map.pdf", dpi=300); fig.savefig(M / "fig5_support_map.png", dpi=200)
print("\nfig5 written")
