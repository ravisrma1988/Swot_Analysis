"""External set, full-record ANA download (13 Oct 2026). Replaces the overpass-window method after timing tests:
ANA's legacy service costs about 4-5 s per request regardless of size, and returns a full year (35k 15-min records)
in about 9 s. So each station is fetched as three yearly requests covering 2023-07-26 .. 2025-12-31.

Order: Amazon basin first (ANA Bacia code 1), then the rest; stations with >= 10 SWOT D passes.
Stored per station as Parquet with the raw DataHora strings (time zone applied only at pairing).
Polite: 4 workers, 0.5 s pause after each request, retries with backoff. Resumable via .done markers.
Raw data: data/raw/ana_brazil/ana_stage_full/; logs in UTC and IST.
"""
import concurrent.futures as cf, pathlib, sys, time
import pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from amazon_step1 import records, ANA
from amazon_download import http, log

ROOT = pathlib.Path(__file__).resolve().parents[1]
AGF = ROOT / "data" / "raw" / "ana_brazil" / "ana_stage_full"; AGF.mkdir(parents=True, exist_ok=True)
SPANS = [("26/07/2023", "31/12/2023"), ("01/01/2024", "31/12/2024"), ("01/01/2025", "31/12/2025")]


def get_full(code):
    done = AGF / f"{int(code):08d}.done"
    if done.exists():
        return "skip", 0
    rows, failed = [], 0
    for a, b in SPANS:
        status, body = http(f"{ANA}/DadosHidrometeorologicos?codEstacao={int(code):08d}&DataInicio={a}&DataFim={b}")
        time.sleep(0.5)
        if status != 200:
            failed += 1; continue
        try:
            rows += [{k: r.get(k, "") for k in ("DataHora", "Nivel", "Vazao")} for r in records(body)]
        except Exception:
            failed += 1
    if failed == len(SPANS):
        return "failed", 0
    pd.DataFrame(rows, columns=["DataHora", "Nivel", "Vazao"]).to_parquet(AGF / f"{int(code):08d}.parquet", index=False)
    done.write_text(f"{len(rows)} rows; failed spans {failed}")
    return "ok", len(rows)


def main():
    m = pd.read_csv(ROOT / "results" / "amazon_v1" / "matches_qc.csv")
    m = m[m.n_swot_D >= 10].drop_duplicates("code")
    m["amazon_basin"] = pd.to_numeric(m.Bacia, errors="coerce") == 1
    m = m.sort_values(["amazon_basin", "code"], ascending=[False, True])
    log(f"{len(m)} stations ({int(m.amazon_basin.sum())} with ANA Bacia code 1 first); 3 yearly requests each")
    t0, k, n_ok, n_rows, n_fail = time.time(), 0, 0, 0, 0
    with cf.ThreadPoolExecutor(4) as ex:
        futs = [ex.submit(get_full, c) for c in m.code]
        for f in cf.as_completed(futs):
            s, nr = f.result()
            k += 1; n_ok += s in ("ok", "skip"); n_fail += s == "failed"; n_rows += nr
            if k % 50 == 0 or k == len(futs):
                eta = (len(futs) - k) / max(k / max(time.time() - t0, 1), 1e-9) / 3600
                log(f"{k}/{len(futs)} stations ({n_ok} complete, {n_fail} failed), {n_rows} rows this run, ETA {eta:.1f} h")
    log("download finished")


if __name__ == "__main__":
    main()
