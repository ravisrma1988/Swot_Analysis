"""Real-shift experiment v9 (13 Oct 2026): same designs, splits and seeds as v8, but density-ratio weights from PHYSICAL
river attributes only (no latitude/longitude): log width, prior WSE, slope, log flow accumulation, log reach length,
channels, obstruction, ice fraction (share of the year with SWORD iceflag >= 1), full-ice fraction (iceflag == 2),
stream order, log distance to outlet. Question: does weighting on physical drivers restore validity across regions
with non-zero retention, where location-based weights (v8) had no overlap?
Diagnostic: test-site weight relative to the calibration weight total (w_j / W); large values block certification.
Outputs: results/real_shift_v9/{per_split.csv, summary.csv, report.txt}
"""
import io, sys, time, warnings
import numpy as np
import pandas as pd
import cupy as cp
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "external" / "pylib"))
import h5py
from swotio import ROOT, SITE
from features_v2 import features_v2, twin_map
from prototype_site_level import evaluate
from prototype_score_sdr import site_scores
from score_sdr_gpu import score_sdr_gpu, score_sdr_w_gpu
from run_frontier_v5 import mu_good
from run_robustness_v6 import subset, oof, choose_lam, fact
from run_shift_v7 import ratio_weights, ess
from real_shift_v8 import site_table, DESIGNS

warnings.filterwarnings("ignore")
pd.set_option("display.width", 240); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 300)
OUT = ROOT / "results" / "real_shift_v9"; OUT.mkdir(parents=True, exist_ok=True)
TAU, ALPHAS, REPEATS, M = 0.25, (0.10, 0.20), 20, 3
buf = io.StringIO()


def out(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); buf.write(s + "\n")


def physical_table():
    with h5py.File(ROOT / "data" / "raw" / "sword_v17b" / "na_sword_v17b.nc", "r") as f:
        g = f["reaches"]
        get = lambda v: np.asarray(g[v][...]).astype(float)
        r = pd.DataFrame({v: get(v) for v in ["reach_id", "x", "y", "width", "wse", "slope", "facc", "reach_length",
                                               "n_chan_mod", "obstr_type", "stream_order", "dist_out"]})
        ice = np.asarray(g["iceflag"][...]).astype(float)
    r = r.replace(-9999.0, np.nan)
    ice[ice < 0] = np.nan
    r["reach_id"] = r.reach_id.astype("int64")
    valid = ~np.isnan(ice)
    nvalid = valid.sum(axis=0).astype(float)
    nvalid[nvalid == 0] = np.nan
    r["ice_frac"] = ((ice >= 1) & valid).sum(axis=0) / nvalid          # share of valid days with partial or full ice
    r["ice_full_frac"] = ((ice == 2) & valid).sum(axis=0) / nvalid
    clipped_log = lambda s: np.log(np.clip(s, 1, None))
    phys = pd.DataFrame({"log_width": clipped_log(r.width), "wse": r.wse, "slope": r.slope, "log_facc": clipped_log(r.facc),
                         "log_len": clipped_log(r.reach_length), "nch": r.n_chan_mod, "obstr": (r.obstr_type > 0).astype(float),
                         "ice_frac": r.ice_frac, "ice_full_frac": r.ice_full_frac, "stream_order": r.stream_order,
                         "log_dist_out": clipped_log(r.dist_out)})
    return pd.concat([r[["reach_id", "x", "y"]], phys], axis=1), list(phys.columns)


def main():
    t0 = time.time()
    sw, COV = physical_table()
    out(f"physical covariates: {COV}; ice_frac available for {sw.ice_frac.notna().mean():.3f} of NA reaches")
    df = pd.read_parquet(ROOT / "data" / "processed" / "paired_clean_v1.parquet")
    twin = twin_map(df, pd.read_parquet(ROOT / "data" / "processed" / "cd_crosswalk_v1.parquet"))
    rows = []
    for t in ("reach", "node"):
        d = subset(df, t, "D")
        X = features_v2(d, t, twin.loc[d.index]).replace([np.inf, -np.inf], np.nan)
        Xg = cp.asarray(X.to_numpy(np.float32))
        bad = (d.eB.abs() > TAU).to_numpy(); y = cp.asarray(bad.astype(np.int32))
        site = d.groupby(SITE).ngroup().to_numpy(); unit = d.unit.to_numpy()
        st = site_table(d, t, sw)
        Zs = st[COV].to_numpy(float)
        ut = st.groupby("unit").agg(agency=("agency", "first"), x=("x", "first"), y=("y", "first"))
        for di, (name, (fsrc, ftgt)) in enumerate(DESIGNS.items()):
            src_u, tgt_u = ut.index[fsrc(ut)].to_numpy(), ut.index[ftgt(ut)].to_numpy()
            t1 = time.time()
            for rep in range(REPEATS):
                rng = np.random.default_rng(11000 * rep + 101 * di + 7 * (t == "node"))      # identical to v8
                su, tu = rng.permutation(src_u), rng.permutation(tgt_u)
                T_u, C_u, F_u, E_u = su[: len(su) // 2], su[len(su) // 2:], tu[: len(tu) // 2], tu[len(tu) // 2:]
                iT, iC, iU = (np.nonzero(np.isin(unit, x))[0] for x in (T_u, C_u, E_u))
                seed = 100 * rep + 13 * di + 1
                mu_C, mu_U = mu_good(Xg, y, iT, [iC, iU], seed)
                mu_T = oof(Xg, y, iT, unit[iT], seed, rng)
                uC, uU = pd.unique(site[iC]), pd.unique(site[iU])
                sC, sU = fact(site[iC]), fact(site[iU])
                w_all = ratio_weights(Zs[np.unique(site[iT])], Zs[np.unique(site[np.isin(unit, F_u)])], Zs[np.r_[uC, uU]], seed)
                wC, wU = w_all[: len(uC)], w_all[len(uC):]
                lams = np.unique(np.quantile(mu_T, np.linspace(0.02, 0.98, 30)))
                for a in ALPHAS:
                    sT = fact(site[iT])
                    lam = choose_lam(mu_T, bad[iT], sT, sT.max() + 1, lams, a, M)
                    s_C, L_C, _ = site_scores(mu_C, bad[iC], sC, sC.max() + 1, lam, M)
                    s_U, _, _ = site_scores(mu_U, None, sU, sU.max() + 1, lam, M)
                    eC, eU = np.isfinite(s_C), np.isfinite(s_U)
                    ratio = wU[eU] / max(wC[eC].sum(), 1e-12) if eC.any() else np.array([np.nan])
                    for meth, W in (("S6 unweighted", None), ("S6w physical weights", (wC, wU))):
                        keep = np.zeros(len(iU), bool)
                        if eC.any() and eU.any():
                            if W is None:
                                sel = score_sdr_gpu(L_C[eC], s_C[eC], s_U[eU], a, a, prune="homo", random_state=seed)
                            else:
                                sel = score_sdr_w_gpu(L_C[eC], s_C[eC], W[0][eC], s_U[eU], W[1][eU], a, a, prune="homo", random_state=seed)
                            cert = np.zeros(sU.max() + 1, bool); cert[np.nonzero(eU)[0][sel]] = True
                            keep = (mu_U >= lam) & cert[sU]
                        rows.append(dict(design=name, type=t, rep=rep, alpha=a, method=meth,
                                         ess_C=ess(W[0][eC]) if W is not None else np.nan,
                                         wratio_med=float(np.median(ratio)) if W is not None else np.nan,
                                         wratio_gt_alpha=float(np.mean(ratio > a)) if W is not None else np.nan,
                                         n_cal_sites=int(eC.sum()), bad_C=bad[iC].mean(), bad_target=bad[iU].mean(),
                                         **evaluate(keep, bad[iU], sU, sU.max() + 1)))
            out(f"{t} {name}: {time.time() - t1:.0f} s (total {time.time() - t0:.0f} s)")
            pd.DataFrame(rows).to_csv(OUT / "per_split.csv", index=False)
    res = pd.DataFrame(rows)
    res["none"] = res.retention == 0
    res["exceed"] = res.site_fdp_kept > res.alpha
    res.to_csv(OUT / "per_split.csv", index=False)
    keys = ["type", "design", "alpha", "method"]
    s = res.groupby(keys).agg(retention=("retention", "mean"), site_coverage=("site_coverage", "mean"),
                              sdr=("site_fdp_kept", "mean"), sdr_sd=("site_fdp_kept", "std"), share_gt_alpha=("exceed", "mean"),
                              share_none=("none", "mean"), ess_C=("ess_C", "mean"), wratio_med=("wratio_med", "mean"),
                              wratio_gt_alpha=("wratio_gt_alpha", "mean"), bad_C=("bad_C", "mean"), bad_target=("bad_target", "mean"))
    s["excess_over_2se"] = (s.sdr - s.index.get_level_values("alpha")) - 2 * s.sdr_sd / np.sqrt(REPEATS)
    s.to_csv(OUT / "summary.csv")
    out("\nwratio = test-site weight / calibration weight total (eligible sites); excess_over_2se > 0 flags a violation")
    out(s.round(3).to_string())
    (OUT / "report.txt").write_text(buf.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
