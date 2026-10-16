"""GPU (PyTorch) port of SCoRE_SDR e-values (Bai & Jin 2026; github.com/Tian-Bai/SCoRE, MIT, commit 401b7ca).

The reference loops over test points, each O(n + m). Here all test points are processed at once as an (m x (n+m))
tensor problem on the GPU, replicating the reference arithmetic operation by operation (float64), so e-values agree
exactly. Sorting, prefix sums and tie correction run in NumPy in the reference order (sequential cumsum). Pruning and
e-BH call the authors' own functions, so random draws and the selection rule are identical.
Run this file directly to execute the equivalence test against the reference implementation.
"""
import pathlib, sys, time
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "external" / "SCoRE"))
from SCoRE.SCoRE import SCoRE_SDR as SCoRE_SDR_ref, SCoRE_SDR_w as SCoRE_SDR_w_ref, _uniform_random
from SCoRE.utility import eBH

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _full(x, v):
    """Tensor of value v shaped like x, so divisions are tensor/tensor (true division, as in NumPy)."""
    return torch.full_like(x, float(v))


def _prefix(Lcal, Scal, Stest):
    n, m = len(Scal), len(Stest)
    score = np.concatenate([Scal, Stest]).astype(float)
    lval = np.concatenate([Lcal, np.zeros(m)]).astype(float)
    typ = np.concatenate([np.zeros(n), np.ones(m)])                       # 'calib' < 'test' as in the reference sort
    o = np.lexsort((typ, lval, score))
    M = score[o]
    NUMER = np.cumsum(np.where(typ[o] == 0, lval[o], 0.0))
    DENOM = 1.0 + np.cumsum(typ[o] == 1)
    last = np.r_[np.nonzero(M[1:] != M[:-1])[0], len(M) - 1]              # last index of each tie group
    grp = np.searchsorted(last, np.arange(len(M)))
    return M, NUMER[last][grp], DENOM[last][grp].astype(float)


def sdr_evalues(Lcal, Scal, Stest, gamma, chunk=4096):
    Lcal, Scal, Stest = (np.asarray(x, float) for x in (Lcal, Scal, Stest))
    n, m = len(Scal), len(Stest)
    M, NUMER, DENOM = _prefix(Lcal, Scal, Stest)
    Mt, Nt, Dt = (torch.as_tensor(x, dtype=torch.float64, device=DEV) for x in (M, NUMER, DENOM))
    N = len(M)
    idx = torch.arange(N, device=DEV)
    c = (n + 1) * gamma / m
    out = []
    for a in range(0, m, chunk):
        S = torch.as_tensor(Stest[a:a + chunk], dtype=torch.float64, device=DEV)[:, None]
        ind = (S <= Mt[None, :]).to(torch.float64)
        Dj = Dt[None, :] - ind
        FR0 = Nt[None, :] / Dj / _full(Dj, n + 1) * m
        FR1 = (Nt[None, :] + ind) / Dj / _full(Dj, n + 1) * m
        ELL = c * Dj - Nt[None, :]
        ok0, ok1 = FR0 <= gamma, FR1 <= gamma
        i0 = torch.where(ok0, idx, -1).max(dim=1).values
        i1 = torch.where(ok1, idx, -1).max(dim=1).values
        ninf = torch.tensor(-np.inf, dtype=torch.float64, device=DEV)
        t0 = torch.where(i0 >= 0, Mt[i0.clamp(min=0)], ninf)
        t1 = torch.where(i1 >= 0, Mt[i1.clamp(min=0)], ninf)
        Y = torch.where(ok0, ELL, ninf)
        suf = torch.flip(torch.cummax(torch.flip(Y, [1]), dim=1).values, [1])           # max over i' >= i
        max_ell = torch.cat([suf[:, 1:], torch.full((len(S), 1), -np.inf, dtype=torch.float64, device=DEV)], dim=1)
        lo = torch.maximum(S[:, 0], t1)[:, None]
        cond = (Mt[None, :] >= lo) & (Mt[None, :] <= t0[:, None]) & ok0 & (ELL > max_ell)
        val = torch.where(cond, torch.div(torch.full_like(ELL, float(n + 1)), ELL + Nt[None, :]),
                          torch.tensor(np.inf, dtype=torch.float64, device=DEV))
        e = val.min(dim=1).values
        same = t1 == t0
        Ni1 = Nt[i1.clamp(min=0)]
        e = torch.where(same, torch.div(torch.full_like(Ni1, float(n + 1)), 1 + Ni1), e)
        e = torch.where(S[:, 0] > t1, torch.zeros_like(e), e)
        out.append(e.cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def score_sdr_gpu(Lcal, Scal, Stest, alpha, gamma, prune=None, random_state=None, return_evals=False):
    ev = sdr_evalues(Lcal, Scal, Stest, gamma)
    if prune == "hete":
        ev = ev / _uniform_random(random_state, len(ev))
    elif prune == "homo":
        ev = ev / _uniform_random(random_state)
    sel = np.asarray(eBH(ev, alpha), dtype=int)
    return (sel, ev) if return_evals else sel


def sdr_evalues_w(Lcal, Scal, wcal, Stest, wtest, gamma, chunk=4096):
    """Weighted (covariate-shift) e-values, replicating SCoRE_SDR_w operation by operation."""
    Lcal, Scal, wcal, Stest, wtest = (np.asarray(x, float) for x in (Lcal, Scal, wcal, Stest, wtest))
    n, m = len(Scal), len(Stest)
    W = np.sum(wcal)
    score = np.concatenate([Scal, Stest]); lval = np.concatenate([Lcal, np.zeros(m)])
    wv = np.concatenate([wcal, wtest]); typ = np.concatenate([np.zeros(n), np.ones(m)])
    o = np.lexsort((typ, wv, lval, score))                                # reference sorts (score, L, w, type)
    M = score[o]
    NUMER = np.cumsum(np.where(typ[o] == 0, wv[o] * lval[o], 0.0))
    DENOM = 1.0 + np.cumsum(typ[o] == 1)
    last = np.r_[np.nonzero(M[1:] != M[:-1])[0], len(M) - 1]
    grp = np.searchsorted(last, np.arange(len(M)))
    NUMER, DENOM = NUMER[last][grp], DENOM[last][grp].astype(float)
    Mt, Nt, Dt = (torch.as_tensor(x, dtype=torch.float64, device=DEV) for x in (M, NUMER, DENOM))
    idx = torch.arange(len(M), device=DEV)
    ninf = torch.tensor(-np.inf, dtype=torch.float64, device=DEV)
    pinf = torch.tensor(np.inf, dtype=torch.float64, device=DEV)
    out = []
    for a in range(0, m, chunk):
        S = torch.as_tensor(Stest[a:a + chunk], dtype=torch.float64, device=DEV)[:, None]
        wj = torch.as_tensor(wtest[a:a + chunk], dtype=torch.float64, device=DEV)
        Wj = W + wj                                                       # calib_w_sum + wtest[j]
        ind = (S <= Mt[None, :]).to(torch.float64)
        Dj = Dt[None, :] - ind
        FR0 = Nt[None, :] / Dj / Wj[:, None] * m
        FR1 = (Nt[None, :] + wj[:, None] * ind) / Dj / Wj[:, None] * m
        cj = Wj / wj * gamma / _full(wj, m)
        ELL = cj[:, None] * Dj - Nt[None, :] / wj[:, None]
        ok0, ok1 = FR0 <= gamma, FR1 <= gamma
        i0 = torch.where(ok0, idx, -1).max(dim=1).values
        i1 = torch.where(ok1, idx, -1).max(dim=1).values
        t0 = torch.where(i0 >= 0, Mt[i0.clamp(min=0)], ninf)
        t1 = torch.where(i1 >= 0, Mt[i1.clamp(min=0)], ninf)
        Y = torch.where(ok0, ELL, ninf)
        suf = torch.flip(torch.cummax(torch.flip(Y, [1]), dim=1).values, [1])
        max_ell = torch.cat([suf[:, 1:], torch.full((len(S), 1), -np.inf, dtype=torch.float64, device=DEV)], dim=1)
        lo = torch.maximum(S[:, 0], t1)[:, None]
        cond = (Mt[None, :] >= lo) & (Mt[None, :] <= t0[:, None]) & ok0 & (ELL > max_ell)
        val = torch.where(cond, Wj[:, None] / (wj[:, None] * ELL + Nt[None, :]), pinf)
        e = val.min(dim=1).values
        e = torch.where(t1 == t0, Wj / (wj + Nt[i1.clamp(min=0)]), e)
        e = torch.where(S[:, 0] > t1, torch.zeros_like(e), e)
        out.append(e.cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def score_sdr_w_gpu(Lcal, Scal, wcal, Stest, wtest, alpha, gamma, prune=None, random_state=None, return_evals=False):
    ev = sdr_evalues_w(Lcal, Scal, wcal, Stest, wtest, gamma)
    if prune == "hete":
        ev = ev / _uniform_random(random_state, len(ev))
    elif prune == "homo":
        ev = ev / _uniform_random(random_state)
    sel = np.asarray(eBH(ev, alpha), dtype=int)
    return (sel, ev) if return_evals else sel


def _equivalence_test():
    rng = np.random.default_rng(0)
    n_cases = bad_e = bad_sel = 0
    n_cases_w = bad_ew = bad_selw = 0
    t_ref = t_gpu = 0.0
    worst = 0.0
    for n, m, reps in [(30, 20, 3), (120, 80, 3), (300, 500, 2), (700, 900, 1)]:
        for ties in (False, True):
            for binary in (False, True):
                for gamma in (0.1, 0.2):
                    for _ in range(reps):
                        S = rng.uniform(0, 1, n + m)
                        if ties:
                            S = np.round(S, 2)
                        noisy = np.clip(S[:n] + 0.2 * rng.normal(size=n), 0, 1)
                        Lc = (rng.uniform(size=n) < noisy).astype(float) if binary else noisy
                        Scal, Stest = S[:n], S[n:]
                        t = time.time(); sr, er = SCoRE_SDR_ref((Lc, Scal), Stest, gamma, gamma, return_evals=True); t_ref += time.time() - t
                        t = time.time(); sg, eg = score_sdr_gpu(Lc, Scal, Stest, gamma, gamma, return_evals=True); t_gpu += time.time() - t
                        n_cases += 1
                        if not np.array_equal(er, eg):
                            bad_e += 1
                            fin = np.isfinite(er) & np.isfinite(eg) & (er > 0)
                            if fin.any():
                                worst = max(worst, float(np.max(np.abs(er[fin] - eg[fin]) / er[fin])))
                        same_sel = set(np.asarray(sr).tolist()) == set(sg.tolist())
                        for prune in ("homo", "hete"):
                            a = SCoRE_SDR_ref((Lc, Scal), Stest, gamma, gamma, prune=prune, random_state=123)
                            b = score_sdr_gpu(Lc, Scal, Stest, gamma, gamma, prune=prune, random_state=123)
                            same_sel &= set(np.asarray(a).tolist()) == set(b.tolist())
                        bad_sel += not same_sel
                        wc, wt = rng.lognormal(0, 0.7, n), rng.lognormal(0, 0.7, m)
                        t = time.time(); srw, erw = SCoRE_SDR_w_ref((Lc, Scal), Stest, wc, wt, gamma, gamma, return_evals=True); t_ref += time.time() - t
                        t = time.time(); sgw, egw = score_sdr_w_gpu(Lc, Scal, wc, Stest, wt, gamma, gamma, return_evals=True); t_gpu += time.time() - t
                        n_cases_w += 1
                        bad_ew += not np.array_equal(erw, egw)
                        same_w = set(np.asarray(srw).tolist()) == set(sgw.tolist())
                        a = SCoRE_SDR_w_ref((Lc, Scal), Stest, wc, wt, gamma, gamma, prune="homo", random_state=7)
                        b = score_sdr_w_gpu(Lc, Scal, wc, Stest, wt, gamma, gamma, prune="homo", random_state=7)
                        bad_selw += not (same_w and set(np.asarray(a).tolist()) == set(b.tolist()))
    print(f"device {DEV}; unweighted cases {n_cases}: e-value mismatches {bad_e} (worst rel diff {worst:.2e}), "
          f"selection mismatches {bad_sel}; weighted cases {n_cases_w}: e-value mismatches {bad_ew}, "
          f"selection mismatches {bad_selw}; time reference {t_ref:.1f} s vs GPU port {t_gpu:.1f} s")
    return bad_e == 0 and bad_sel == 0 and bad_ew == 0 and bad_selw == 0


if __name__ == "__main__":
    sys.exit(0 if _equivalence_test() else 1)
