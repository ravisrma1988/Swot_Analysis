"""Score-model tuning (14 Oct 2026): grouped 5-fold CV over river units on Version C (reach and node), so tuning data are
disjoint from the Version D headline evaluation. 24-config grid + the previous default; selection by mean log-loss
over both types. Then a monotone-constraint test for the selected config (P(bad) non-decreasing in flags, bits,
uncertainty, dark fraction and consistency residuals; non-increasing in area ratio).
Outputs: results/final/{tuning.csv, monotone.csv, model_config.json}
"""
import itertools, json, pathlib, sys, time, warnings
import numpy as np
import pandas as pd
import cupy as cp
import xgboost as xgb
from sklearn.metrics import log_loss, roc_auc_score
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from swotio import ROOT
from run_robustness_v6 import subset
from features_v2 import features_v2, twin_map

warnings.filterwarnings("ignore")
OUT = ROOT / "results" / "final"; OUT.mkdir(parents=True, exist_ok=True)
BASE = dict(device="cuda", tree_method="hist", objective="binary:logistic", subsample=0.8, colsample_bytree=0.8)
GRID = [dict(max_depth=d, learning_rate=lr, n_estimators=n, min_child_weight=w)
        for d, lr, n, w in itertools.product((4, 6, 8), (0.03, 0.06), (300, 700), (1, 10))]
GRID.append(dict(max_depth=6, learning_rate=0.05, n_estimators=400, min_child_weight=1))   # previous default
UP = ("qflag", "dark_frac", "wse_u", "wse_r_u", "pdiff", "xovr_cal_q", "robust_z", "dev_med5", "jump_prev", "jump_next",
      "hampel_r", "hampel_flag", "width_wse_resid", "site_frac_q2plus", "site_mad_wse", "ice")


def monotone(cols):
    return "(" + ",".join("1" if (c in UP or c.startswith("bit")) else ("-1" if c == "area_ratio" else "0") for c in cols) + ")"


def cv(X, y, unit, params, seed=0):
    folds = np.array_split(np.random.default_rng(seed).permutation(np.unique(unit)), 5)
    p = np.zeros(len(y)); Xg = cp.asarray(X.to_numpy(np.float32)); yg = cp.asarray(y.astype(np.int32))
    for k, fu in enumerate(folds):
        te = np.isin(unit, fu); tr = ~te
        clf = xgb.XGBClassifier(random_state=k, **params).fit(Xg[cp.asarray(np.nonzero(tr)[0])], yg[cp.asarray(np.nonzero(tr)[0])])
        pr = clf.predict_proba(Xg[cp.asarray(np.nonzero(te)[0])])[:, 1]
        p[te] = cp.asnumpy(pr) if isinstance(pr, cp.ndarray) else np.asarray(pr)
    return log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)), roc_auc_score(y, p)


def main():
    t0 = time.time()
    df = pd.read_parquet(ROOT / "data" / "processed" / "paired_clean_v1.parquet")
    twin = twin_map(df, pd.read_parquet(ROOT / "data" / "processed" / "cd_crosswalk_v1.parquet"))
    data = {}
    for t in ("reach", "node"):
        d = subset(df, t, "C")
        X = features_v2(d, t, twin.loc[d.index]).replace([np.inf, -np.inf], np.nan)
        data[t] = (X, (d.eB.abs() > 0.25).to_numpy(), d.unit.to_numpy())
    rows = []
    for i, g in enumerate(GRID):
        for t, (X, y, u) in data.items():
            ll, auc = cv(X, y, u, {**BASE, **g})
            rows.append(dict(cfg=i, type=t, logloss=ll, auc=auc, **g))
        print(f"cfg {i + 1}/{len(GRID)} at {time.time() - t0:.0f} s", flush=True)
    r = pd.DataFrame(rows); r.to_csv(OUT / "tuning.csv", index=False)
    mean = r.groupby("cfg")[["logloss", "auc"]].mean()
    best = int(mean.logloss.idxmin())
    print(mean.sort_values("logloss").head(5).round(4).to_string())
    print(f"previous default (cfg {len(GRID) - 1}): {mean.loc[len(GRID) - 1].round(4).to_dict()}; selected cfg {best}: {GRID[best]}")
    mono = []
    for t, (X, y, u) in data.items():
        for flag in (False, True):
            p = {**BASE, **GRID[best], **({"monotone_constraints": monotone(X.columns)} if flag else {})}
            ll, auc = cv(X, y, u, p)
            mono.append(dict(type=t, monotone=flag, logloss=ll, auc=auc))
    mono = pd.DataFrame(mono); mono.to_csv(OUT / "monotone.csv", index=False); print(mono.round(4).to_string(index=False))
    gain = mono[~mono.monotone].logloss.mean() - mono[mono.monotone].logloss.mean()
    cfg = {**GRID[best], "subsample": 0.8, "colsample_bytree": 0.8, "monotone_gain_logloss": round(float(gain), 5),
           "tuned_on": "Version C reach+node, grouped 5-fold by unit, tau 0.25"}
    (OUT / "model_config.json").write_text(json.dumps(cfg, indent=2))
    print("model_config.json written:", cfg)


if __name__ == "__main__":
    main()
