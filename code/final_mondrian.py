"""Group-conditional (Mondrian-type) calibration (14 Oct 2026), GPU-bound, tuned model, same splits/seeds as final_main.
Version D, tau 0.25, alpha {0.10, 0.20}, reach and node.
Groups use label-free site attributes available at ungauged reaches: prior width class (<100, 100-300, >=300 m) and
north (latitude >= 55 N). Schemes: 'global' (one calibration), 'mondrian width', 'mondrian width x north'.
Within each group: lambda from out-of-fold predictions on the group's training sites (global lambda if < 20 sites) and
SCoRE-SDR ('homo') calibrated on the group's calibration sites only (group abstains if < 20 eligible calibration sites).
Per-group guarantees hold within each pre-specified group; overall validity is not implied in general (the overall
rate weights groups by random certified counts) and is checked empirically. (Corrected 14 Oct 2026 after audit.)
Outputs: results/final/mondrian_{per_split,overall,groups}.csv, mondrian_report.txt
"""
import io, sys, time, warnings
import numpy as np
import pandas as pd
import cupy as cp
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from swotio import ROOT, SITE
from features_v2 import features_v2, twin_map
from prototype_site_level import evaluate
from prototype_score_sdr import site_scores
from run_frontier_v5 import mu_good
from run_robustness_v6 import subset, oof, choose_lam, fact
from final_extras import certify, group_rows

warnings.filterwarnings("ignore")
VERSION = __import__("os").environ.get("VERSION", "D")
OUT = ROOT / "results" / "final" if VERSION == "D" else ROOT / "results" / "final" / f"mondrian_{VERSION}"
OUT.mkdir(parents=True, exist_ok=True)
TAU, ALPHAS, REPEATS, FOLDS, M, MIN_SITES = 0.25, (0.10, 0.20), 4, 5, 3, 20
buf = io.StringIO()


def out(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); buf.write(s + "\n")


def mondrian(gl, site, iT, iC, iU, mu_T, mu_C, mu_U, bad, lam_global, a, seed):
    keep = np.zeros(len(iU), bool); abstained = []
    for g in np.unique(gl[site[iU]]):
        mT, mC, mU = gl[site[iT]] == g, gl[site[iC]] == g, gl[site[iU]] == g
        sTg, sCg, sUg = fact(site[iT][mT]), fact(site[iC][mC]), fact(site[iU][mU])
        if mC.sum() == 0 or sCg.max() + 1 < MIN_SITES:
            abstained.append(str(g)); continue
        lam = lam_global
        if mT.any() and sTg.max() + 1 >= MIN_SITES:
            lams = np.unique(np.quantile(mu_T[mT], np.linspace(0.02, 0.98, 30)))
            lam = choose_lam(mu_T[mT], bad[iT][mT], sTg, sTg.max() + 1, lams, a, M)
        _, k = certify(mu_C[mC], bad[iC][mC], sCg, mu_U[mU], sUg, lam, a, "homo", seed)
        keep[np.nonzero(mU)[0]] = k
    return keep, abstained


def main():
    t0 = time.time()
    df = pd.read_parquet(ROOT / "data" / "processed" / "paired_clean_v1.parquet")
    twin = twin_map(df, pd.read_parquet(ROOT / "data" / "processed" / "cd_crosswalk_v1.parquet"))
    rows, grows = [], []
    for t in ("reach", "node"):
        d = subset(df, t, VERSION)
        X = features_v2(d, t, twin.loc[d.index]).replace([np.inf, -np.inf], np.nan)
        Xg = cp.asarray(X.to_numpy(np.float32))
        bad = (d.eB.abs() > TAU).to_numpy(); y = cp.asarray(bad.astype(np.int32))
        unit, site = d.unit.to_numpy(), d.groupby(SITE).ngroup().to_numpy(); units = np.unique(unit)
        sa = d.groupby(SITE).agg(w=("p_width", "first"), lat=("gage_latitude", "first")).reset_index(drop=True)
        w3 = pd.cut(sa.w, [0, 100, 300, np.inf], labels=["<100 m", "100-300 m", ">=300 m"]).astype(str).to_numpy()
        north = np.where(sa.lat >= 55, ">=55N", "<55N")
        cell = np.char.add(np.char.add(w3.astype(str), " | "), north.astype(str))
        lat4 = pd.cut(sa.lat, [-np.inf, 35, 45, 55, np.inf], labels=["<35N", "35-45N", "45-55N", ">=55N"]).astype(str).to_numpy()
        out(f"{t}: sites per width class {pd.Series(w3).value_counts().to_dict()}; north sites {int((north == '>=55N').sum())}")
        for q in range(REPEATS):
            folds = np.array_split(np.random.default_rng(5000 * q + 17 * (t == "node")).permutation(units), FOLDS)
            for f in range(FOLDS):
                rng = np.random.default_rng(100 * q + f); seed = 1000 * q + 10 * f
                U = np.isin(unit, folds[f]); gg = rng.permutation(np.unique(unit[~U])); half = len(gg) // 2
                iT, iC, iU = np.nonzero(np.isin(unit, gg[:half]))[0], np.nonzero(np.isin(unit, gg[half:]))[0], np.nonzero(U)[0]
                mu_C, mu_U = mu_good(Xg, y, iT, [iC, iU], seed)
                mu_T = oof(Xg, y, iT, unit[iT], seed, rng)
                sT, sC, sU = fact(site[iT]), fact(site[iC]), fact(site[iU])
                lams = np.unique(np.quantile(mu_T, np.linspace(0.02, 0.98, 30)))
                uU = pd.unique(site[iU])
                groups = {"width": w3[uU], "latitude": lat4[uU], "width x north": cell[uU]}
                for a in ALPHAS:
                    lam = choose_lam(mu_T, bad[iT], sT, sT.max() + 1, lams, a, M)
                    _, keep_g = certify(mu_C, bad[iC], sC, mu_U, sU, lam, a, "homo", seed)
                    variants = {"global": (keep_g, [])}
                    variants["mondrian width"] = mondrian(w3, site, iT, iC, iU, mu_T, mu_C, mu_U, bad, lam, a, seed)
                    variants["mondrian width x north"] = mondrian(cell, site, iT, iC, iU, mu_T, mu_C, mu_U, bad, lam, a, seed)
                    for scheme, (keep, abst) in variants.items():
                        tag = dict(type=t, rep=q, fold=f, alpha=a, scheme=scheme)
                        rows.append(dict(tag, abstained_groups=";".join(abst), **evaluate(keep, bad[iU], sU, sU.max() + 1)))
                        grows += [r for r in group_rows(tag, keep, bad[iU], sU, groups, np.array(["all"] * len(iU))) if r["level"] == "site"]
            out(f"   {t} repeat {q + 1}/{REPEATS} at {time.time() - t0:.0f} s")
    r = pd.DataFrame(rows); g = pd.DataFrame(grows)
    r.to_csv(OUT / "mondrian_per_split.csv", index=False); g.to_csv(OUT / "mondrian_groups_per_split.csv", index=False)
    ov = r.groupby(["type", "alpha", "scheme"]).agg(retention=("retention", "mean"), coverage=("site_coverage", "mean"),
                                                    sdr=("site_fdp_kept", "mean"), sd=("site_fdp_kept", "std"))
    ov["excess_over_2se"] = ov.sdr - ov.index.get_level_values("alpha") - 2 * ov.sd / np.sqrt(REPEATS * FOLDS)
    ov.to_csv(OUT / "mondrian_overall.csv")
    gs = g.groupby(["type", "alpha", "attribute", "group", "scheme"]).agg(coverage=("coverage", "mean"), sdr=("sdr", "mean"),
                                                                          sd=("sdr", "std"), n=("sdr", "count"))
    gs["excess_over_2se"] = gs.sdr - gs.index.get_level_values("alpha") - 2 * gs.sd / np.sqrt(gs.n.clip(lower=1))
    gs.to_csv(OUT / "mondrian_groups.csv")
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 300)
    out("\n== overall\n" + ov.round(3).to_string())
    piv = gs.reset_index().pivot_table(index=["type", "alpha", "attribute", "group"], columns="scheme", values="sdr").round(3)
    out("\n== realised site-level SDR by group and scheme\n" + piv.to_string())
    cov = gs.reset_index().pivot_table(index=["type", "alpha", "attribute", "group"], columns="scheme", values="coverage").round(3)
    out("\n== coverage by group and scheme\n" + cov.to_string())
    out(f"\ndone in {time.time() - t0:.0f} s")
    (OUT / "mondrian_report.txt").write_text(buf.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
