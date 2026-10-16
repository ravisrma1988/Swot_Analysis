"""Prototype v4 (13 Oct 2026): site-level SDR with the authors' SCoRE implementation.

SCoRE: Bai & Jin (2026) arXiv 2603.24704, code github.com/Tian-Bai/SCoRE (MIT), commit 401b7ca, cloned to external/SCoRE.
We call SCoRE.SCoRE.SCoRE_SDR unchanged, with sites as units:
  eligible site  : at least M observations with mu >= lam (label-free); ineligible sites abstain deterministically and are
                   excluded from calibration and test, so abstention cannot dilute the risk
  site score     : mean (1 - mu) over its kept observations (smaller = safer, as SCoRE expects)
  site loss L    : share of kept observations with |eB| > TAU (bias-relative labels)
  lam            : chosen on calibration part A (40% of calibration units) by plug-in retention at level alpha;
                   SCoRE runs on part B (calibration) and the held-out sites U. gamma = alpha (authors' recommendation).
Variants: prune=None and prune='homo' (authors' recommended boosting; randomised, seeded per split).
Reference rows (M3, S1, S4, S5b) from results/site_level_proto_v3/per_split_with_v3b.csv, same splits and seeds.
"""
import io, sys, time, warnings
import numpy as np
import pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "external" / "SCoRE"))
from SCoRE.SCoRE import SCoRE_SDR
from swotio import ROOT, SITE, label_B
from check_failure_mode import features, p_good
from prototype_site_level import evaluate
from prototype_score_sdr import site_scores, plugin_curve

warnings.filterwarnings("ignore")
pd.set_option("display.width", 230); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 200)
OUT = ROOT / "results" / "site_level_proto_v4"; OUT.mkdir(parents=True, exist_ok=True)
TAU, ALPHAS, REPEATS, M = 0.25, (0.10, 0.20), 20, 3
buf = io.StringIO()


def out(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); buf.write(s + "\n")


def choose_lam(mu_A, bad_A, sA, nsA, lams, alpha):
    best = (0.0, lams[len(lams) // 2])
    for lam in lams:
        s, L, kept = site_scores(mu_A, bad_A, sA, nsA, lam, M)
        so, fin, risk, ret = plugin_curve(s, L, kept, len(mu_A))
        feas = np.nonzero(fin & (risk <= alpha))[0]
        if len(feas) and ret[feas.max()] > best[0]:
            best = (ret[feas.max()], lam)
    return best[1]


def score_sdr(mu_B, bad_B, sB, nsB, mu_U, sU, nsU, lam, alpha, prune, seed):
    s_B, L_B, _ = site_scores(mu_B, bad_B, sB, nsB, lam, M)
    s_U, _, _ = site_scores(mu_U, None, sU, nsU, lam, M)
    eB, eU = np.isfinite(s_B), np.isfinite(s_U)
    if eB.sum() == 0 or eU.sum() == 0:
        return np.zeros(len(mu_U), bool)
    sel = SCoRE_SDR((L_B[eB], s_B[eB]), s_U[eU], alpha, alpha, prune=prune, random_state=seed)
    cert = np.zeros(nsU, bool)
    cert[np.nonzero(eU)[0][np.asarray(sel, dtype=int)]] = True
    return (mu_U >= lam) & cert[sU]


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
                rng = np.random.default_rng(10007 * r + (t == "node") * 7 + (v == "C") * 3)   # same splits as v1-v3
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
                    lam = choose_lam(mu_ca[inA], bad_ca[inA], sA, sA.max() + 1, lams, a)
                    for prune in (None, "homo"):
                        keep = score_sdr(mu_ca[~inA], bad_ca[~inA], sB, sB.max() + 1, mu_U, s_U, ns, lam, a, prune, 1000 + r)
                        rows.append(dict(type=t, v=v, rep=r, alpha=a, method=f"S6 SCoRE-SDR sites (prune={prune})",
                                         **evaluate(keep, bad_U, s_U, ns)))
            out(f"   done {t} {v} at {time.time() - t0:.0f} s")
    ref = pd.read_csv(ROOT / "results" / "site_level_proto_v3" / "per_split_with_v3b.csv")
    ref = ref[ref.method.str.match(r"M3|S1|S4|S5b site SDR e-BH single t \(rho=0.8\)")]
    res = pd.concat([ref, pd.DataFrame(rows)], ignore_index=True)
    res["sel_exceed"] = res.site_fdp_kept > res.alpha
    res["none"] = res.retention == 0
    res.to_csv(OUT / "per_split.csv", index=False)
    s = res.groupby(["type", "v", "alpha", "method"]).agg(
        retention=("retention", "mean"), site_coverage=("site_coverage", "mean"), pooled_fdp=("fdp", "mean"),
        site_fdp_where_kept=("site_fdp_kept", "mean"), share_splits_gt_alpha=("sel_exceed", "mean"),
        share_splits_none=("none", "mean"))
    s.to_csv(OUT / "summary.csv")
    out("\nsite_fdp_where_kept: mean over splits of the average bad share at held-out sites that keep data (0 if a split keeps nothing)")
    out(s.round(3).to_string())
    (OUT / "report.txt").write_text(buf.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
