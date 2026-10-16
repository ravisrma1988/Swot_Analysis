"""SWORD v17b support analysis (13 Oct 2026): how far are ungauged North American river reaches from the gauged ones?

Source: our Version D gauged reaches (SWORD v17b ids). Target: all other North American SWORD v17b river reaches
(type digit 1, lakeflag 0). Covariates from SWORD only (same definition for both): log width, prior WSE, lat, lon,
channels, obstruction flag, log reach length, log flow accumulation, slope.
Density ratio w = P(target|z) / P(source|z) * n_source / n_target from a 5-fold cross-fitted XGBoost (CUDA) classifier.
Outputs: results/sword_support_v1/{report.txt, reach_weights.csv, weight_map.png}
"""
import io, pathlib, shutil, sys, zipfile
import numpy as np
import pandas as pd
import cupy as cp
import xgboost as xgb
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external" / "pylib"))
import h5py

RAW = ROOT / "data" / "raw" / "sword_v17b"
OUT = ROOT / "results" / "sword_support_v1"; OUT.mkdir(parents=True, exist_ok=True)
WANT = ["reach_id", "x", "y", "width", "wse", "slope", "facc", "reach_length", "n_chan_mod", "obstr_type", "lakeflag"]
buf = io.StringIO()


def out(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); buf.write(s + "\n")


def load_na():
    z = zipfile.ZipFile(RAW / "SWORD_v17b_netcdf.zip")
    names = [n for n in z.namelist() if n.lower().endswith(".nc") and "na_sword" in n.lower()]
    out("zip members (first 12):", z.namelist()[:12]); out("North America file:", names)
    path = RAW / pathlib.Path(names[0]).name
    if not path.exists():
        with z.open(names[0]) as src, open(path, "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 24)
    with h5py.File(path, "r") as f:
        g = f["reaches"]
        out("reaches variables:", sorted(g.keys()))
        missing = [v for v in WANT if v not in g]
        if missing:
            sys.exit(f"missing variables {missing}; adapt WANT to the names listed above")
        return pd.DataFrame({v: np.asarray(g[v][:]).ravel() for v in WANT})


def covariates(r):
    return pd.DataFrame({"log_width": np.log(np.clip(r.width, 1, None)), "wse": r.wse, "lat": r.y, "lon": r.x,
                         "nch": r.n_chan_mod, "obstr": (r.obstr_type > 0).astype(float),
                         "log_len": np.log(np.clip(r.reach_length, 1, None)), "log_facc": np.log(np.clip(r.facc, 1, None)),
                         "slope": r.slope}).replace([np.inf, -np.inf], np.nan)


def main():
    r = load_na()
    r["type"] = r.reach_id.astype("int64") % 10
    rivers = r[(r.type == 1) & (r.lakeflag == 0)].copy()
    clean = pd.read_parquet(ROOT / "data" / "processed" / "paired_clean_v1.parquet", columns=["type", "v", "fid", "phase", "tidal_gauge"])
    gauged = set(clean[(clean.type == "reach") & (clean.v == "D") & (clean.phase == "science") & (~clean.tidal_gauge)].fid.astype("int64"))
    rivers["gauged"] = rivers.reach_id.astype("int64").isin(gauged)
    out(f"NA reaches {len(r)}; river reaches (type 1, lakeflag 0) {len(rivers)}; gauged among them {int(rivers.gauged.sum())} "
        f"of {len(gauged)} gauged reach ids")
    Z = covariates(rivers).to_numpy(np.float32); yv = (~rivers.gauged).to_numpy().astype(np.int32)
    p = np.zeros(len(rivers)); rng = np.random.default_rng(0); fold = rng.integers(0, 5, len(rivers))
    for k in range(5):
        tr, te = fold != k, fold == k
        clf = xgb.XGBClassifier(device="cuda", tree_method="hist", n_estimators=300, learning_rate=0.05, max_depth=5,
                                subsample=0.8, colsample_bytree=0.8, random_state=k)
        clf.fit(cp.asarray(Z[tr]), cp.asarray(yv[tr]))
        pr = clf.predict_proba(cp.asarray(Z[te]))[:, 1]
        p[te] = cp.asnumpy(pr) if isinstance(pr, cp.ndarray) else np.asarray(pr)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    n_s, n_t = int(rivers.gauged.sum()), int((~rivers.gauged).sum())
    rivers["w"] = p / (1 - p) * n_s / n_t
    from sklearn.metrics import roc_auc_score
    out(f"out-of-fold AUC (gauged vs ungauged): {roc_auc_score(yv, p):.3f}")
    wg, wt = rivers.w[rivers.gauged].to_numpy(), rivers.w[~rivers.gauged].to_numpy()
    out("weight quantiles, gauged reaches :", np.round(np.quantile(wg, [.05, .25, .5, .75, .95, .99]), 3).tolist())
    out("weight quantiles, ungauged reaches:", np.round(np.quantile(wt, [.05, .25, .5, .75, .95, .99]), 3).tolist())
    out(f"ESS of gauged weights: {wg.sum() ** 2 / (wg ** 2).sum():.0f} of {len(wg)}")
    for thr in (np.quantile(wg, 0.99), wg.max()):
        out(f"share of ungauged reaches with w <= {thr:.2f} (gauged-support proxy): {(wt <= thr).mean():.3f}")
    rivers[["reach_id", "x", "y", "width", "facc", "gauged", "w"]].to_csv(OUT / "reach_weights.csv", index=False)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 7))
    u = rivers[~rivers.gauged]
    sc = ax.scatter(u.x, u.y, c=np.log10(u.w), s=0.3, cmap="viridis", vmin=-1, vmax=3)
    g = rivers[rivers.gauged]
    ax.scatter(g.x, g.y, s=3, c="red", label="gauged reaches")
    fig.colorbar(sc, label="log10 density ratio w (ungauged / gauged)")
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.legend(loc="lower left")
    fig.savefig(OUT / "weight_map.png", dpi=150, bbox_inches="tight")
    (OUT / "report.txt").write_text(buf.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
