"""E19 sensitivity (16 Oct 2026): main design restricted to river units that hold exactly one site.

Purpose: in the main design the random partition is drawn over river units, and ~10% of sites share a unit, so site-level
exchangeability (the SCoRE condition) is approximate. Keeping only single-site units makes unit and site exchangeability
coincide, so the theorem's conditions hold exactly for the evaluated population.
Pre-specified report (whatever the outcome): retention and realized SDR, reach and node, Version D, tau 0.25 m,
alpha 0.10 and 0.20, S6 'homo' (proposed) and S6 None, with the same 2-SE descriptive flag as the main run.
Reuses the frozen functions of final_main.py unchanged; same seed formulas; 4 repeats x 5 folds over units.
Outputs (new folder, nothing frozen is touched): results/sensitivity_single_site/{per_split,summary}.csv, report.txt
"""
import io, sys, time, warnings
import numpy as np, pandas as pd, cupy as cp
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from swotio import ROOT, SITE
from features_v2 import features_v2, twin_map
from check_failure_mode import grid_masks
from run_robustness_v6 import subset
from final_main import run_split, REPEATS, FOLDS        # frozen functions and constants, unchanged

warnings.filterwarnings("ignore")
OUT = ROOT / "results" / "sensitivity_single_site"; OUT.mkdir(parents=True, exist_ok=True)
TAU, ALPHAS = 0.25, (0.10, 0.20)
buf = io.StringIO()
def out(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); buf.write(s + "\n")

t0 = time.time()
df = pd.read_parquet(ROOT / "data" / "processed" / "paired_clean_v1.parquet")
twin = twin_map(df, pd.read_parquet(ROOT / "data" / "processed" / "cd_crosswalk_v1.parquet"))
rows = []
for t in ("reach", "node"):
    v = "D"
    full = subset(df, t, v)
    sites_per_unit = full.groupby(SITE).unit.first().value_counts()
    single = sites_per_unit.index[sites_per_unit == 1]
    d = full[full.unit.isin(single)]
    out(f"{t} {v}: units {len(sites_per_unit)} -> single-site units {len(single)}; sites {full.groupby(SITE).ngroups} -> "
        f"{d.groupby(SITE).ngroups}; pairs {len(full)} -> {len(d)}")
    X = features_v2(d, t, twin.loc[d.index]).replace([np.inf, -np.inf], np.nan)
    Xg = cp.asarray(X.to_numpy(np.float32)); F = grid_masks(d, X, t)
    unit, site, qflag = d.unit.to_numpy(), d.groupby(SITE).ngroup().to_numpy(), d.qflag.to_numpy()
    assert (pd.Series(site).groupby(unit).nunique() == 1).all(), "a unit holds more than one site"
    units = np.unique(unit)
    bad = (d.eB.abs() > TAU).to_numpy(); y = cp.asarray(bad.astype(np.int32))
    for q in range(REPEATS):
        folds = np.array_split(np.random.default_rng(5000 * q + 17 * (t == "node") + 3 * (v == "C")).permutation(units), FOLDS)
        for f in range(FOLDS):
            rows += run_split(Xg, y, bad, unit, site, qflag, F, folds[f], np.random.default_rng(100 * q + f), 1000 * q + 10 * f,
                              dict(type=t, v=v, tau=TAU, rep=q, fold=f), ALPHAS, baselines=False, m3=False)
    out(f"{t}: bad rate {bad.mean():.3f}; elapsed {time.time() - t0:.0f} s")
res = pd.DataFrame(rows); res.to_csv(OUT / "per_split.csv", index=False)
g = res.groupby(["type", "alpha", "method"])
s = g.agg(retention=("retention", "mean"), site_coverage=("site_coverage", "mean"), pooled_fdp=("fdp", "mean"),
          sdr=("site_fdp_kept", "mean"), sdr_sd=("site_fdp_kept", "std"), n=("site_fdp_kept", "size"))
s["excess_over_2se"] = (s.sdr - s.index.get_level_values("alpha")) - 2 * s.sdr_sd / np.sqrt(s.n)
s.to_csv(OUT / "summary.csv")
out(s.round(3).to_string())
out(f"flagged (excess > 2 SE): {int((s.excess_over_2se > 0).sum())} of {len(s)}; total {time.time() - t0:.0f} s")
(OUT / "report.txt").write_text(buf.getvalue(), encoding="utf-8")
