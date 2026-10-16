"""Prototype v3b: site-level SDR e-BH with a SINGLE threshold (v3's uniform mixture diluted the e-values; diagnosed 13 Oct).

With one fixed threshold t, every held-out site with s_j <= t gets E = (n_B + 1) / (1 + sum_B L_i 1{s_i <= t}) and e-BH
certifies all of them iff (1 + sum_B L 1{s <= t}) / (n_B + 1) <= alpha * K_U / m_U, otherwise none (valid: t, lam from A).
lam and t chosen on A as the most permissive pair whose plug-in selective risk on A is <= rho * alpha, rho in {1.0, 0.8}.
Reference rows are read from results/site_level_proto_v3/per_split.csv (same splits and seeds).
"""
import io, sys, time, warnings
import numpy as np
import pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from swotio import ROOT, SITE, label_B
from check_failure_mode import features, p_good
from prototype_site_level import evaluate
from prototype_score_sdr import site_scores, plugin_curve

warnings.filterwarnings("ignore")
pd.set_option("display.width", 230); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 200)
OUT = ROOT / "results" / "site_level_proto_v3"
TAU, ALPHAS, REPEATS, M = 0.25, (0.10, 0.20), 20, 3
buf = io.StringIO()


def out(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); buf.write(s + "\n")


def sdr_single(mu_A, bad_A, sA, nsA, mu_B, bad_B, sB, nsB, mu_U, sU, nsU, lams, alpha, rho):
    best = (0.0, None, None)
    for lam in lams:
        s, L, kept = site_scores(mu_A, bad_A, sA, nsA, lam, M)
        so, fin, risk, ret = plugin_curve(s, L, kept, len(mu_A))
        feas = np.nonzero(fin & (risk <= rho * alpha))[0]
        if len(feas) and ret[feas.max()] > best[0]:
            best = (ret[feas.max()], lam, so[feas.max()])
    _, lam, t = best
    if lam is None:
        return np.zeros(len(mu_U), bool), np.nan
    s_B, L_B, _ = site_scores(mu_B, bad_B, sB, nsB, lam, M)
    s_U, _, _ = site_scores(mu_U, None, sU, nsU, lam, M)
    K_U = int(np.sum(s_U <= t))
    lhs = (1 + np.sum(L_B * (s_B <= t))) / (nsB + 1)
    ok = K_U > 0 and lhs <= alpha * K_U / nsU
    return ((mu_U >= lam) & (s_U <= t)[sU]) if ok else np.zeros(len(mu_U), bool), lhs * nsU / max(K_U, 1)


def main():
    t0 = time.time()
    df = pd.read_parquet(ROOT / "data" / "processed" / "paired_clean_v1.parquet")
    rows = []
    for t in ("reach", "node"):
        for v in ("D", "C"):
            d = df[(df.type == t) & (df.v == v) & (df.phase == "science") & (~df.tidal_gauge)].copy()
            d["n"] = d.groupby(SITE).wse.transform("size")
            d = d[d.n >= 5].copy()
            d["eB"], _ = label_B(d)
            bad = (d.eB.abs() > TAU).to_numpy()
            unit = (d.fid.astype("int64") // (10 if t == "reach" else 10000)).to_numpy()
            site = d.groupby(SITE).ngroup().to_numpy()
            Xn = features(d, t).to_numpy(float)
            units = np.unique(unit)
            for r in range(REPEATS):
                rng = np.random.default_rng(10007 * r + (t == "node") * 7 + (v == "C") * 3)
                gu = rng.permutation(units); g_units = gu[: len(gu) // 2]
                U = ~np.isin(unit, g_units)
                half = len(g_units) // 2
                tr, ca = np.isin(unit, g_units[:half]), np.isin(unit, g_units[half:])
                mu_ca, mu_U = p_good(Xn[tr], bad[tr], [Xn[ca], Xn[U]], r)
                bad_ca, bad_U = bad[ca], bad[U]
                u_ca = pd.factorize(unit[ca])[0]
                s_U = pd.factorize(site[U])[0]; ns = s_U.max() + 1
                inA = np.isin(u_ca, rng.permutation(np.unique(u_ca))[: int(0.4 * (u_ca.max() + 1))])
                sA, sB = pd.factorize(site[ca][inA])[0], pd.factorize(site[ca][~inA])[0]
                lams = np.unique(np.quantile(mu_ca, np.linspace(0.02, 0.98, 30)))
                for a in ALPHAS:
                    for rho in (1.0, 0.8):
                        keep, est = sdr_single(mu_ca[inA], bad_ca[inA], sA, sA.max() + 1, mu_ca[~inA], bad_ca[~inA], sB, sB.max() + 1,
                                               mu_U, s_U, ns, lams, a, rho)
                        rows.append(dict(type=t, v=v, rep=r, alpha=a, method=f"S5b site SDR e-BH single t (rho={rho})",
                                         est_sdr=est, **evaluate(keep, bad_U, s_U, ns)))
            out(f"   done {t} {v} at {time.time() - t0:.0f} s")
    res = pd.concat([pd.read_csv(OUT / "per_split.csv"), pd.DataFrame(rows)], ignore_index=True)
    res = res[~res.method.str.startswith("S5 site SDR e-BH (m=1)")]
    res["sel_exceed"] = res.site_fdp_kept > res.alpha
    res["none"] = res.retention == 0
    res.to_csv(OUT / "per_split_with_v3b.csv", index=False)
    s = res.groupby(["type", "v", "alpha", "method"]).agg(
        retention=("retention", "mean"), site_coverage=("site_coverage", "mean"), pooled_fdp=("fdp", "mean"),
        site_fdp_where_kept=("site_fdp_kept", "mean"), share_splits_gt_alpha=("sel_exceed", "mean"),
        share_splits_none=("none", "mean"))
    s.to_csv(OUT / "summary_with_v3b.csv")
    out(s.round(3).to_string())
    (OUT / "report_v3b.txt").write_text(buf.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
