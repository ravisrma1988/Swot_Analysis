"""Certifiable domain v1 (13 Oct 2026): restrict the ungauged target to reaches inside gauged support and recompute
weights and effective sample size. Thresholds: q-quantiles of the gauged reaches' weights from sword_support_v1.
For each q: domain = ungauged river reaches with w <= threshold (label-free); weights are refit (5-fold XGBoost on CUDA,
gauged vs domain) and the ESS of the gauged weights is reported. Outputs: results/sword_domain_v1/{report.txt,
domain_q99.csv, domain_map.png}
"""
import io, pathlib, sys
import numpy as np
import pandas as pd
import cupy as cp
import xgboost as xgb
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import sword_support as ss

OUT = ss.ROOT / "results" / "sword_domain_v1"; OUT.mkdir(parents=True, exist_ok=True)
buf = io.StringIO()


def out(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); buf.write(s + "\n")


def cross_fit(Z, y, seed=0):
    p = np.zeros(len(y)); fold = np.random.default_rng(seed).integers(0, 5, len(y))
    for k in range(5):
        tr, te = fold != k, fold == k
        clf = xgb.XGBClassifier(device="cuda", tree_method="hist", n_estimators=300, learning_rate=0.05, max_depth=5,
                                subsample=0.8, colsample_bytree=0.8, random_state=k).fit(cp.asarray(Z[tr]), cp.asarray(y[tr]))
        pr = clf.predict_proba(cp.asarray(Z[te]))[:, 1]
        p[te] = cp.asnumpy(pr) if isinstance(pr, cp.ndarray) else np.asarray(pr)
    return np.clip(p, 1e-6, 1 - 1e-6)


def main():
    w0 = pd.read_csv(ss.ROOT / "results" / "sword_support_v1" / "reach_weights.csv")
    r = ss.load_na()
    r["type"] = r.reach_id.astype("int64") % 10
    rivers = r[(r.type == 1) & (r.lakeflag == 0)].merge(w0[["reach_id", "gauged", "w"]], on="reach_id")
    Z = ss.covariates(rivers).to_numpy(np.float32)
    g = rivers.gauged.to_numpy()
    wg = rivers.w[g].to_numpy()
    out(f"river reaches {len(rivers)}, gauged {g.sum()}; full-target ESS (from sword_support_v1) "
        f"{wg.sum() ** 2 / (wg ** 2).sum():.0f}")
    for q in (0.90, 0.95, 0.99):
        thr = np.quantile(wg, q)
        dom = (~g) & (rivers.w.to_numpy() <= thr)
        keep = g | dom
        y = (~g[keep]).astype(np.int32)
        p = cross_fit(Z[keep], y)
        w = p / (1 - p) * g.sum() / dom.sum()
        wgn = w[y == 0]
        out(f"q={q}: threshold {thr:.2f}; domain {dom.sum()} reaches = {dom.sum() / (~g).sum():.3f} of ungauged; "
            f"AUC {roc_auc_score(y, p):.3f}; ESS {wgn.sum() ** 2 / (wgn ** 2).sum():.0f} of {g.sum()}; "
            f"gauged weight q95/q99 {np.quantile(wgn, .95):.2f}/{np.quantile(wgn, .99):.2f}")
        if q == 0.99:
            rivers.assign(in_domain=dom | g)[["reach_id", "x", "y", "gauged", "w", "in_domain"]].to_csv(OUT / "domain_q99.csv", index=False)
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 7))
            o = rivers[~g & ~dom]; i = rivers[dom]; k = rivers[g]
            ax.scatter(o.x, o.y, s=0.3, c="#d95f02", label=f"outside support ({len(o)})")
            ax.scatter(i.x, i.y, s=0.3, c="#1b9e77", label=f"inside support ({len(i)})")
            ax.scatter(k.x, k.y, s=2, c="black", label=f"gauged ({len(k)})")
            ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.legend(loc="lower left", markerscale=8)
            ax.set_title("Certifiable domain (q = 0.99 of gauged weights), SWORD v17b North American river reaches")
            fig.savefig(OUT / "domain_map.png", dpi=150, bbox_inches="tight")
    (OUT / "report.txt").write_text(buf.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
