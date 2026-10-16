"""Fig. 5 support map, print-readable version: full extent to -15 deg longitude; marker size grows with class
(within < weak < outside) so sparse classes stay visible after reduction; outside class in black for grayscale and
colour-blind distinction; legend wording 'estimated support' (diagnostic, not certification). Legend below the map."""
import pathlib
import pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = pathlib.Path(__file__).resolve().parents[1] / "results"; M = R / "manuscript"
u = pd.read_csv(R / "audit_response" / "E_support_classes.csv")
fig, ax = plt.subplots(figsize=(7.16, 3.75))
for k, c, s, lab in (("within", "tab:blue", 0.35, "within estimated support"), ("weak", "tab:orange", 1.1, "weak estimated support"),
                     ("outside", "black", 2.2, "outside estimated support")):
    d = u[u.support.str.startswith(k)]
    ax.scatter(d.x, d.y, s=s, c=c, linewidths=0, rasterized=True, label=f"{lab} ({len(d):,}, {len(d) / len(u):.1%})")
ax.set_xlim(-170, -15); ax.set_ylim(5, 84); ax.set_aspect("equal")
ax.set_xlabel("Longitude (deg)", fontsize=8); ax.set_ylabel("Latitude (deg)", fontsize=8); ax.tick_params(labelsize=7)
leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), fontsize=7, frameon=False, ncol=3, columnspacing=1.2)
for h in leg.legend_handles:
    h.set_sizes([18])
fig.savefig(M / "fig5_support_map.pdf", dpi=300, bbox_inches="tight", pad_inches=0.02)
fig.savefig(M / "fig5_support_map.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
print("fig5 rewritten")
