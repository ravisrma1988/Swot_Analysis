"""Diverse-training test (14 Oct 2026), GPU-bound: does adding Brazilian reaches to score training help on held-out
Brazilian reaches? Reach level, tau 0.25, alpha {0.10, 0.20}, Brasilia time, strata 'main' and 'strict_qc', 20 repeats.
Brazil units split in thirds: B_train / B_cal / B_eval. NA units 50/50 into T / C.
Scores: A = trained on NA T only; B = trained on NA T + B_train. Lambda from out-of-fold predictions on each score's own
training set. Calibration: NA C or B_cal. Evaluation: S6 SCoRE-SDR ('homo') on B_eval.
Outputs: results/external_diverse/{per_split.csv, summary.csv, report.txt}
"""
import io, pathlib, sys, time, warnings
import numpy as np
import pandas as pd
import cupy as cp
import xgboost as xgb
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from swotio import ROOT, SITE
from external_test_v1 import notwin, run_s6
from external_test_v2 import prep
from prototype_site_level import evaluate
from prototype_score_sdr import site_scores
from run_frontier_v5 import XGB
from run_robustness_v6 import subset, oof, choose_lam, fact

warnings.filterwarnings("ignore")
OUT = ROOT / "results" / "external_diverse"; OUT.mkdir(parents=True, exist_ok=True)
TAU, ALPHAS, M, REPEATS = 0.25, (0.10, 0.20), 3, 20
buf = io.StringIO()


def out(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); buf.write(s + "\n")


def fit_predict(Xg, yg, tr, preds, seed):
    clf = xgb.XGBClassifier(random_state=seed, **XGB).fit(Xg[cp.asarray(tr)], yg[cp.asarray(tr)])
    res = []
    for p in preds:
        pr = clf.predict_proba(Xg[cp.asarray(p)])[:, 1]
        res.append(1 - (cp.asnumpy(pr) if isinstance(pr, cp.ndarray) else np.asarray(pr)))
    return res


def main():
    t0 = time.time()
    d = subset(pd.read_parquet(ROOT / "data" / "processed" / "paired_clean_v1.parquet"), "reach", "D")
    Xna = notwin(d); cols = Xna.columns
    b, masks = prep()
    Xb = notwin(b).reindex(columns=cols); bits = [c for c in cols if c.startswith("bit")]; Xb[bits] = Xb[bits].fillna(0)
    n_na = len(d)
    Xg = cp.asarray(np.vstack([Xna.to_numpy(np.float32), Xb.to_numpy(np.float32)]))
    bad = np.r_[(d.eB.abs() > TAU).to_numpy(), (b.eB.abs() > TAU).to_numpy()]; yg = cp.asarray(bad.astype(np.int32))
    site_na = d.groupby(SITE).ngroup().to_numpy()
    site = np.r_[site_na, b.groupby(SITE).ngroup().to_numpy() + site_na.max() + 1]
    unit = np.r_[d.unit.to_numpy(), b.unit.to_numpy()]
    rows = []
    for stratum in ("main", "strict_qc"):
        sel = np.r_[np.zeros(n_na, bool), masks[stratum]]
        for r in range(REPEATS):
            rng = np.random.default_rng(41000 + r); seed = 700 + r
            nu = rng.permutation(np.unique(unit[:n_na])); T_u, C_u = nu[: len(nu) // 2], nu[len(nu) // 2:]
            bu = rng.permutation(np.unique(unit[sel])); k = len(bu) // 3
            Btr, Bcal, Bev = bu[:k], bu[k: 2 * k], bu[2 * k:]
            iT = np.nonzero(np.isin(unit[:n_na], T_u))[0]; iC = np.nonzero(np.isin(unit[:n_na], C_u))[0]
            iBtr, iBcal, iBev = (np.nonzero(sel & np.isin(unit, x))[0] for x in (Btr, Bcal, Bev))
            sE = fact(site[iBev])
            for sname, tr in (("A: NA-only score", iT), ("B: NA + Brazil score", np.r_[iT, iBtr])):
                mu_C, mu_Bcal, mu_E = fit_predict(Xg, yg, tr, [iC, iBcal, iBev], seed)
                mu_T = oof(Xg, yg, tr, unit[tr], seed, rng)
                sT = fact(site[tr]); lams = np.unique(np.quantile(mu_T, np.linspace(0.02, 0.98, 30)))
                for a in ALPHAS:
                    lam = choose_lam(mu_T, bad[tr], sT, sT.max() + 1, lams, a, M)
                    s_E, _, _ = site_scores(mu_E, None, sE, sE.max() + 1, lam, M)
                    for cname, mu_cal, ical in (("NA calib", mu_C, iC), ("Brazil calib", mu_Bcal, iBcal)):
                        sc = fact(site[ical])
                        s_c, L_c, _ = site_scores(mu_cal, bad[ical], sc, sc.max() + 1, lam, M)
                        cert = run_s6(L_c, s_c, s_E, a, seed)
                        keep = np.zeros(len(iBev), bool) if cert is None else (mu_E >= lam) & cert[sE]
                        rows.append(dict(stratum=stratum, rep=r, alpha=a, score=sname, calibration=cname,
                                         **evaluate(keep, bad[iBev], sE, sE.max() + 1)))
            out(f"{stratum} repeat {r + 1}/{REPEATS} at {time.time() - t0:.0f} s")
    res = pd.DataFrame(rows); res["none"] = res.retention == 0
    res.to_csv(OUT / "per_split.csv", index=False)
    s = res.groupby(["stratum", "alpha", "score", "calibration"]).agg(
        retention=("retention", "mean"), site_coverage=("site_coverage", "mean"), sdr=("site_fdp_kept", "mean"),
        sdr_sd=("site_fdp_kept", "std"), share_none=("none", "mean"))
    s["excess_over_2se"] = (s.sdr - s.index.get_level_values("alpha")) - 2 * s.sdr_sd / np.sqrt(REPEATS)
    s.to_csv(OUT / "summary.csv"); out(s.round(3).to_string())
    (OUT / "report.txt").write_text(buf.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
