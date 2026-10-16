"""Certifiable domain v2 (13 Oct 2026): support of ungauged North American river reaches with PHYSICAL covariates
(as real_shift_v9: no latitude/longitude). Compares with the location-based analysis (sword_support_v1, sword_domain_v1).

1. Cross-fitted (5-fold, XGBoost CUDA) density ratio, gauged vs ungauged river reaches: AUC, weight quantiles, ESS.
2. Domains at q in {0.90, 0.95, 0.99} of gauged weights, refit weights, ESS (as sword_domain_v1).
3. Transferability ratio per ungauged reach: w_j / sum of gauged weights, with all gauged reaches as calibration.
   Heuristic classes from real_shift_v9 diagnostics (median ratio ~0.01 certified well, >= 0.2 mostly abstained):
   <= 0.01 likely certifiable, 0.01-0.1 partial, > 0.1 abstention likely. These classes are descriptive, not a guarantee.
Outputs: results/sword_domain_v2/{report.txt, reach_support_physical.csv, domain_map_physical.png}
"""
import io, pathlib, sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from sklearn.metrics import roc_auc_score
import sword_support as ss
from sword_domain_v1 import cross_fit
from real_shift_v9 import physical_table

OUT = ss.ROOT / "results" / "sword_domain_v2"; OUT.mkdir(parents=True, exist_ok=True)
buf = io.StringIO()


def out(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); buf.write(s + "\n")


def main():
    sw, COV = physical_table()
    base = ss.load_na()[["reach_id", "lakeflag"]]
    base["reach_id"] = base.reach_id.astype("int64")
    r = sw.merge(base, on="reach_id")
    rivers = r[(r.reach_id % 10 == 1) & (r.lakeflag == 0)].copy()
    clean = pd.read_parquet(ss.ROOT / "data" / "processed" / "paired_clean_v1.parquet", columns=["type", "v", "fid", "phase", "tidal_gauge"])
    gauged = set(clean[(clean.type == "reach") & (clean.v == "D") & (clean.phase == "science") & (~clean.tidal_gauge)].fid.astype("int64"))
    g = rivers.reach_id.isin(gauged).to_numpy()
    Z = rivers[COV].to_numpy(np.float32)
    y = (~g).astype(np.int32)
    p = cross_fit(Z, y)
    w = p / (1 - p) * g.sum() / (~g).sum()
    wg, wt = w[g], w[~g]
    out(f"river reaches {len(rivers)}, gauged {g.sum()}; covariates {COV}")
    out(f"physical covariates: AUC {roc_auc_score(y, p):.3f} (location-based v1: 0.803); "
        f"ESS {wg.sum() ** 2 / (wg ** 2).sum():.0f} of {g.sum()} (location-based v1: 132)")
    out("weight quantiles gauged  :", np.round(np.quantile(wg, [.05, .25, .5, .75, .95, .99]), 3).tolist())
    out("weight quantiles ungauged:", np.round(np.quantile(wt, [.05, .25, .5, .75, .95, .99]), 3).tolist())
    for q in (0.90, 0.95, 0.99):
        thr = np.quantile(wg, q)
        dom = (~g) & (w <= thr)
        keep = g | dom
        yy = (~g[keep]).astype(np.int32)
        pp = cross_fit(Z[keep], yy)
        wn = pp / (1 - pp) * g.sum() / dom.sum()
        wgn = wn[yy == 0]
        out(f"q={q}: domain {dom.sum()} = {dom.sum() / (~g).sum():.3f} of ungauged (location-based v1: "
            f"{ {0.90: 0.480, 0.95: 0.578, 0.99: 0.755}[q] }); AUC {roc_auc_score(yy, pp):.3f}; ESS {wgn.sum() ** 2 / (wgn ** 2).sum():.0f}")
    ratio = wt / wg.sum()
    cls = np.select([ratio <= 0.01, ratio <= 0.1], ["likely certifiable", "partial"], "abstention likely")
    out("transferability classes (ungauged reaches):", pd.Series(cls).value_counts(normalize=True).round(3).to_dict())
    u = rivers.loc[~g, ["reach_id", "x", "y"]].assign(w=wt, ratio=ratio, cls=cls)
    u.to_csv(OUT / "reach_support_physical.csv", index=False)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 7))
    colours = {"likely certifiable": "#1b9e77", "partial": "#e6ab02", "abstention likely": "#d95f02"}
    for k, c in colours.items():
        s = u[u.cls == k]
        ax.scatter(s.x, s.y, s=0.3, c=c, label=f"{k} ({len(s)})")
    k = rivers[g]
    ax.scatter(k.x, k.y, s=2, c="black", label=f"gauged ({len(k)})")
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.legend(loc="lower left", markerscale=8)
    ax.set_title("Transferability of site-level guarantees (physical covariates; heuristic classes)")
    fig.savefig(OUT / "domain_map_physical.png", dpi=150, bbox_inches="tight")
    (OUT / "report.txt").write_text(buf.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
