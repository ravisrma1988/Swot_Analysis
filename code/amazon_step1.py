"""Amazon / Brazil external set, step 1 (13 Oct 2026): ANA telemetric stations -> SWORD v17b river nodes -> SWOT D coverage.

1. SWORD v17b South America (from the verified zip): river reaches (type 1, lakeflag 0) and their nodes.
2. ANA legacy telemetry service, ListaEstacoesTelemetricas: station codes and coordinates (fields printed as returned).
3. Stations with 8-digit fluviometric-style codes (>= 10000000) matched to the nearest river node within 1 km
   (same buffer as the USGS pairing); distance, node width and names kept for manual checks.
4. Hydrocron (SWOT_L2_HR_RiverSP_D, open API): valid observations per matched reach, 2023-07-26 to 2025-12-31.
Outputs: results/amazon_v1/{ana_stations.csv, matches.csv, report.txt}
"""
import io, json, pathlib, shutil, sys, time, urllib.parse, urllib.request, zipfile
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external" / "pylib"))
import h5py

RAW = ROOT / "data" / "raw"
OUT = ROOT / "results" / "amazon_v1"; OUT.mkdir(parents=True, exist_ok=True)
ANA = "http://telemetriaws1.ana.gov.br/ServiceANA.asmx"
HYDROCRON = "https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries"
buf = io.StringIO()


def out(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); buf.write(s + "\n")


def sword_sa():
    nc = RAW / "sword_v17b" / "sa_sword_v17b.nc"
    if not nc.exists():
        with zipfile.ZipFile(RAW / "sword_v17b" / "SWORD_v17b_netcdf.zip") as z, z.open("netcdf/sa_sword_v17b.nc") as s, open(nc, "wb") as d:
            shutil.copyfileobj(s, d, 1 << 24)
    with h5py.File(nc, "r") as f:
        r, n = f["reaches"], f["nodes"]
        reaches = pd.DataFrame({k: np.asarray(r[k][...]) for k in ["reach_id", "x", "y", "width", "lakeflag"]})
        reaches["river_name"] = [v.decode() if isinstance(v, bytes) else str(v) for v in r["river_name"][...]]
        nodes = pd.DataFrame({k: np.asarray(n[k][...]) for k in ["node_id", "reach_id", "x", "y", "width"]})
    reaches = reaches[(reaches.reach_id % 10 == 1) & (reaches.lakeflag == 0)]
    nodes = nodes[nodes.reach_id.isin(reaches.reach_id)]
    return reaches, nodes


def records(xml_bytes):
    """Rows of a .NET DataTable response: children of the first element under <diffgram>."""
    root = ET.fromstring(xml_bytes)
    dg = next(el for el in root.iter() if el.tag.split('}')[-1] == "diffgram")
    container = list(dg)[0] if len(dg) else dg
    return [{c.tag.split('}')[-1]: (c.text or "").strip() for c in row} for row in container]


def ana_stations():
    with urllib.request.urlopen(f"{ANA}/ListaEstacoesTelemetricas?statusEstacoes=&origem=", timeout=180) as r:
        return pd.DataFrame(records(r.read()))


def xyz(lon, lat):
    lo, la = np.radians(np.asarray(lon, float)), np.radians(np.asarray(lat, float))
    return 6371.0 * np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]   # chord ~ arc at km scale


def swot_count(reach_id):
    q = {"feature": "Reach", "feature_id": str(reach_id), "start_time": "2023-07-26T00:00:00Z",
         "end_time": "2025-12-31T23:59:59Z", "output": "csv", "fields": "reach_id,time_str,wse",
         "collection_name": "SWOT_L2_HR_RiverSP_D"}
    for attempt in range(4):
        try:
            with urllib.request.urlopen(HYDROCRON + "?" + urllib.parse.urlencode(q), timeout=120) as r:
                csv = json.load(r)["results"]["csv"]
            rows = [l.split(",") for l in csv.strip().splitlines()[1:]]
            ok = [x for x in rows if len(x) >= 3 and x[1] != "no_data" and float(x[2]) > -1e10]
            return len(ok)
        except urllib.error.HTTPError as e:
            if e.code in (400, 404):
                return 0
            time.sleep(5 * (attempt + 1))
        except Exception:
            time.sleep(5 * (attempt + 1))
    return -1


def main():
    reaches, nodes = sword_sa()
    out(f"SWORD v17b South America: {len(reaches)} river reaches, {len(nodes)} river nodes")
    st = ana_stations()
    st.to_csv(OUT / "ana_stations.csv", index=False)
    out(f"ANA telemetric stations: {len(st)}; fields {list(st.columns)}")
    out(st.head(3).to_string(index=False))
    col = {c.lower(): c for c in st.columns}
    code_c = next((col[k] for k in ("codestacao", "codigo", "codigoestacao") if k in col), None)
    lat_c = next((col[k] for k in ("latitude", "lat") if k in col), None)
    lon_c = next((col[k] for k in ("longitude", "lon", "long") if k in col), None)
    if not (code_c and lat_c and lon_c):
        sys.exit("could not identify code/latitude/longitude columns; see fields above")
    st["code"] = pd.to_numeric(st[code_c], errors="coerce")
    st["lat"] = pd.to_numeric(st[lat_c].str.replace(",", "."), errors="coerce")
    st["lon"] = pd.to_numeric(st[lon_c].str.replace(",", "."), errors="coerce")
    fl = st[(st.code >= 10_000_000) & (st.code <= 99_999_999) & st.lat.notna() & st.lon.notna()].drop_duplicates("code").copy()
    out(f"fluviometric-style codes with coordinates: {len(fl)}")
    tree = cKDTree(xyz(nodes.x, nodes.y))
    dist, idx = tree.query(xyz(fl.lon, fl.lat))
    fl["dist_km"], fl["node_id"] = dist, nodes.node_id.to_numpy()[idx]
    fl["reach_id"], fl["node_width"] = nodes.reach_id.to_numpy()[idx], nodes.width.to_numpy()[idx]
    m = fl[fl.dist_km <= 1.0].merge(reaches[["reach_id", "river_name", "width"]].rename(columns={"width": "reach_width"}), on="reach_id")
    out(f"matched within 1 km of a SWORD river node: {len(m)} stations on {m.reach_id.nunique()} reaches; "
        f"distance quantiles km {np.round(np.quantile(m.dist_km, [.5, .9]), 3).tolist()}")
    counts = {}
    for k, rid in enumerate(m.reach_id.unique()):
        counts[rid] = swot_count(rid)
        time.sleep(0.3)
        if k % 50 == 0:
            out(f"   Hydrocron {k}/{m.reach_id.nunique()}")
    m["n_swot_D"] = m.reach_id.map(counts)
    m.to_csv(OUT / "matches.csv", index=False)
    amz = (m.lat.between(-20, 6)) & (m.lon.between(-80, -44))
    for name, sel in (("all matched", m.index == m.index), ("Amazon bounding box", amz)):
        s = m[sel]
        out(f"{name}: stations {len(s)}, reaches {s.reach_id.nunique()}, with >= 10 valid SWOT D obs: "
            f"{int((s.n_swot_D >= 10).sum())} stations; failed queries {int((s.n_swot_D < 0).sum())}")
    (OUT / "report.txt").write_text(buf.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
