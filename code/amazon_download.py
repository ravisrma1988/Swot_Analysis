"""External set bulk download (13 Oct 2026). Usage: python3 amazon_download.py [amazon|brazil]

Stage A: Hydrocron SWOT_L2_HR_RiverSP_D reach series (2023-07-26 .. 2025-12-31) for every matched reach -> raw CSV text.
Stage B: ANA legacy telemetry 15-min stage only in windows around each valid SWOT overpass: local dates assumed
  UTC-3 for windowing, each window [d-1, d+1] covers either time-zone reading; overlapping windows merged, max 30 days.
  Raw DataHora strings are stored untouched; the time zone is applied only at pairing.
Polite: 4 workers, 0.3 s pause per request, retries with backoff. Resumable: finished stations have a .done marker.
Raw data: data/raw/ana_brazil/{swot_reach_D, ana_stage}; log times in UTC and IST.
"""
import concurrent.futures as cf, datetime, json, pathlib, sys, time, urllib.error, urllib.parse, urllib.request
import pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from amazon_step1 import records, ANA, HYDROCRON

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "ana_brazil"
SW, AG = RAW / "swot_reach_D", RAW / "ana_stage"
for p in (SW, AG):
    p.mkdir(parents=True, exist_ok=True)
FULL = ("reach_id,time_str,cycle_id,pass_id,wse,wse_u,wse_r_u,width,width_u,slope,slope_u,area_total,area_tot_u,"
        "area_detct,area_det_u,area_wse,reach_q,reach_q_b,dark_frac,ice_clim_f,partial_f,n_good_nod,obs_frac_n,"
        "xovr_cal_q,xtrk_dist,p_wse,p_width,p_length,p_n_nodes,p_dam_id,p_n_ch_mod,p_low_slp,crid")
CORE = "reach_id,time_str,cycle_id,pass_id,wse,wse_u,width,reach_q,dark_frac,xtrk_dist,p_wse,p_width"


def stamp():
    u = datetime.datetime.now(datetime.timezone.utc)
    return f"{u:%Y-%m-%d %H:%M} UTC / {u + datetime.timedelta(hours=5, minutes=30):%H:%M} IST"


def log(*a):
    print(f"[{stamp()}]", *a, flush=True)


def http(url, tries=5):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=180) as r:
                return 200, r.read()
        except urllib.error.HTTPError as e:
            if e.code in (400, 404):
                return e.code, e.read()
            time.sleep(5 * (k + 1))
        except Exception:
            time.sleep(5 * (k + 1))
    return -1, b""


def hydrocron_url(rid, fields):
    q = {"feature": "Reach", "feature_id": str(int(rid)), "start_time": "2023-07-26T00:00:00Z",
         "end_time": "2025-12-31T23:59:59Z", "output": "csv", "fields": fields, "collection_name": "SWOT_L2_HR_RiverSP_D"}
    return HYDROCRON + "?" + urllib.parse.urlencode(q)


def get_reach(rid, fields):
    path = SW / f"{int(rid)}.csv"
    if path.exists():
        return True
    code, body = http(hydrocron_url(rid, fields))
    time.sleep(0.3)
    if code != 200:
        return False
    path.write_text(json.loads(body)["results"]["csv"], encoding="utf-8")
    return True


def windows(times):
    days = sorted({(t - pd.Timedelta(hours=3)).normalize() for t in times})
    w = []
    for d in days:
        a, b = d - pd.Timedelta(days=1), d + pd.Timedelta(days=1)
        if w and a <= w[-1][1] + pd.Timedelta(days=1) and (b - w[-1][0]).days <= 30:
            w[-1][1] = max(w[-1][1], b)
        else:
            w.append([a, b])
    return w


def get_station(code, rid):
    done = AG / f"{int(code):08d}.done"
    if done.exists():
        return "skip", 0
    p = SW / f"{int(rid)}.csv"
    if not p.exists():
        return "no_swot", 0
    sw = pd.read_csv(p)
    if "time_str" not in sw or sw.empty:
        return "no_swot", 0
    sw = sw[(sw.time_str != "no_data") & (pd.to_numeric(sw.wse, errors="coerce") > -1e10)]
    t = pd.to_datetime(sw.time_str, utc=True, errors="coerce").dropna().dt.tz_localize(None)
    rows = []
    for a, b in windows(t):
        code_http, body = http(f"{ANA}/DadosHidrometeorologicos?codEstacao={int(code):08d}"
                               f"&DataInicio={a:%d/%m/%Y}&DataFim={b:%d/%m/%Y}")
        time.sleep(0.3)
        if code_http == 200:
            try:
                rows += [{k: r.get(k, "") for k in ("CodEstacao", "DataHora", "Nivel", "Vazao")} for r in records(body)]
            except Exception:
                pass
    pd.DataFrame(rows, columns=["CodEstacao", "DataHora", "Nivel", "Vazao"]).to_csv(AG / f"{int(code):08d}.csv", index=False)
    done.write_text(stamp())
    return "ok", len(rows)


def main():
    region = sys.argv[1] if len(sys.argv) > 1 else "amazon"
    m = pd.read_csv(ROOT / "results" / "amazon_v1" / "matches_qc.csv")
    if region == "amazon":
        m = m[m.amazon_box]
    m = m[m.n_swot_D >= 10]
    rids = m.reach_id.unique()
    log(f"region {region}: {len(m)} stations on {len(rids)} reaches")
    code, body = http(hydrocron_url(rids[0], FULL))
    fields = FULL if code == 200 else CORE
    log(f"Hydrocron field probe: HTTP {code}; using {'FULL' if fields == FULL else 'CORE'} list"
        + ("" if code == 200 else f"; server said: {body[:300]!r}"))
    t0 = time.time()
    with cf.ThreadPoolExecutor(4) as ex:
        res = list(ex.map(lambda r: get_reach(r, fields), rids))
    log(f"Stage A done: {sum(res)}/{len(rids)} reaches saved in {time.time() - t0:.0f} s")
    t0, n_ok, n_rows, k = time.time(), 0, 0, 0
    with cf.ThreadPoolExecutor(4) as ex:
        futs = {ex.submit(get_station, c, r): c for c, r in zip(m.code, m.reach_id)}
        for f in cf.as_completed(futs):
            status, nr = f.result()
            k += 1; n_ok += status in ("ok", "skip"); n_rows += nr
            if k % 25 == 0:
                rate = k / max(time.time() - t0, 1)
                eta = (len(futs) - k) / max(rate, 1e-9) / 3600
                log(f"Stage B {k}/{len(futs)} stations, {n_rows} stage rows this run, ETA {eta:.1f} h")
    log(f"Stage B done: {n_ok}/{len(m)} stations complete, {n_rows} rows this run")


if __name__ == "__main__":
    main()
