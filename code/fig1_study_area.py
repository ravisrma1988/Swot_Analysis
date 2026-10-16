"""Fig. 1 (study area) and Table I counts for Section III. Basic colours, no decoration.
Backdrop: all SWORD v17b river reaches (light grey) for North America and South America.
(a) North American gauged reaches (Version D, science phase, non-tidal, >= 5 pairs) by agency group.
(b) Brazilian ANA gauges of the external test (Brasilia time, no dam/reservoir, no tidal, >= 5 pairs), main stratum.
Outputs: results/manuscript/fig1_study_area.pdf/.png, results/manuscript/table1_counts.txt
"""
import pathlib, sys
import numpy as np
import pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code")); sys.path.insert(0, str(ROOT / "external" / "pylib"))
import h5py
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from swotio import SITE
from run_robustness_v6 import subset

OUT = ROOT / "results" / "manuscript"; OUT.mkdir(parents=True, exist_ok=True)
raw = ROOT / "data" / "raw" / "sword_v17b"


def reaches(nc):
    with h5py.File(nc, "r") as f:
        g = f["reaches"]
        r = pd.DataFrame({k: np.asarray(g[k][...]) for k in ("reach_id", "x", "y", "lakeflag")})
    r["reach_id"] = r.reach_id.astype("int64")
    return r


na_all, sa_all = reaches(raw / "na_sword_v17b.nc"), reaches(raw / "sa_sword_v17b.nc")
riv = lambda r: r[(r.reach_id % 10 == 1) & (r.lakeflag == 0)]
na, sa = riv(na_all), riv(sa_all)
df = pd.read_parquet(ROOT / "data" / "processed" / "paired_clean_v1.parquet")
d = subset(df, "reach", "D")
s = d.groupby(SITE).agg(fid=("fid", "first"), agency=("gage_agency", "first")).reset_index(drop=True)
s = s.merge(na_all[["reach_id", "x", "y"]], left_on="fid", right_on="reach_id", how="left")
b = pd.read_parquet(ROOT / "data" / "processed" / "brazil_pairs_v1.parquet")
b = b[(b.tz_mode == "brasilia") & ~b.dam_or_reservoir.astype(bool) & ~b.tidal_candidate.astype(bool)]
b["n"] = b.groupby(SITE).wse.transform("size"); b = b[b.n >= 5]
main = b[(b.name_status != "mismatch") & (b.dist_km <= 0.5)]
st = pd.read_csv(ROOT / "results" / "amazon_v1" / "matches_qc.csv")[["code", "lat", "lon"]].drop_duplicates("code")
bg = st[st.code.isin(main.code.unique())]

fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.75), gridspec_kw={"width_ratios": [2.44, 1]})
for a, bgd in ((ax[0], na), (ax[1], sa)):
    a.scatter(bgd.x, bgd.y, s=0.15, c="0.72", linewidths=0, rasterized=True)
us, ca = s[s.agency == "USGS"], s[s.agency != "USGS"]
ax[0].scatter(us.x, us.y, s=3, c="tab:blue", linewidths=0, label=f"United States gauges ({len(us)})", rasterized=True)
ax[0].scatter(ca.x, ca.y, s=3, c="tab:red", linewidths=0, label=f"Canadian gauges ({len(ca)})", rasterized=True)
ax[1].scatter(bg.lon, bg.lat, s=3, c="tab:green", linewidths=0, label=f"Brazilian gauges ({len(bg)})", rasterized=True)
ax[0].set_xlim(-170, -50); ax[0].set_ylim(24, 72); ax[1].set_xlim(-75, -34); ax[1].set_ylim(-34, 6)
for a, t in zip(ax, ("(a)", "(b)")):
    a.set_xlabel("Longitude (deg)", fontsize=8); a.set_ylabel("Latitude (deg)", fontsize=8)
    a.tick_params(labelsize=7); a.set_aspect("equal")
    a.set_title(t, fontsize=9, loc="left")
h0, l0 = ax[0].get_legend_handles_labels(); h1, l1 = ax[1].get_legend_handles_labels()
fig.legend(h0 + h1, l0 + l1, loc="lower center", bbox_to_anchor=(0.5, -0.07), ncol=3, fontsize=7, markerscale=3, frameon=False)
fig.tight_layout()
fig.savefig(OUT / "fig1_study_area.pdf", dpi=300, bbox_inches="tight", pad_inches=0.02); fig.savefig(OUT / "fig1_study_area.png", dpi=300, bbox_inches="tight", pad_inches=0.02)

lines = [f"NA Version D reach sites with SWORD position: {s.x.notna().sum()} of {len(s)}; USGS {len(us)}, Canada {len(ca)}",
         f"Brazil main-stratum gauges on map: {len(bg)}"]
bn = pd.read_parquet(ROOT / "data" / "processed" / "brazil_pairs_nodes_v1.parquet")
bn = bn[(bn.tz_mode == "brasilia") & ~bn.dam_or_reservoir.astype(bool) & ~bn.tidal_candidate.astype(bool)].copy()
bn["n"] = bn.groupby(SITE).wse.transform("size"); bn = bn[bn.n >= 5].copy()
bn["eB"] = (bn.wse - bn.gage_stage_m) - (bn.wse - bn.gage_stage_m).groupby([bn[c] for c in SITE]).transform("median")
bm = bn[(bn.name_status != "mismatch") & (bn.dist_km <= 0.5)]
lines.append(f"Brazil node main stratum: {len(bm)} pairs, {bm.gauge.nunique()} gauges, {bm.groupby(SITE).ngroup().max() + 1} sites, "
             f"bad rate {np.mean(np.abs(bm.eB) > 0.25):.3f}")
lines.append(f"Brazil reach main stratum: {len(main)} pairs, {main.gauge.nunique()} gauges")
(OUT / "table1_counts.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
