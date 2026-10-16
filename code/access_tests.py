"""Access tests for the Amazon external set: (1) Hydrocron SWOT RiverSP Version D time series (open API),
(2) ANA's legacy public telemetry service (no login), for the Rio Negro at Manaus gauge (code 14990000)."""
import urllib.parse, urllib.request

import pandas as pd

reach = int(pd.read_parquet("data/processed/paired_clean_v1.parquet", columns=["type", "v", "fid"])
            .query("type == 'reach' and v == 'D'").fid.iloc[0])
q = {"feature": "Reach", "feature_id": str(reach), "start_time": "2024-01-01T00:00:00Z", "end_time": "2024-03-01T00:00:00Z",
     "output": "csv", "fields": "reach_id,time_str,wse,reach_q", "collection_name": "SWOT_L2_HR_RiverSP_D"}
url = "https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries?" + urllib.parse.urlencode(q)
print("Hydrocron test reach", reach)
try:
    with urllib.request.urlopen(url, timeout=60) as r:
        body = r.read().decode("utf-8", "replace")
    print("  HTTP", r.status, "| first 600 chars:\n", body[:600])
except Exception as e:
    print("  failed:", e.__class__.__name__, str(e)[:300])

ana = ("http://telemetriaws1.ana.gov.br/ServiceANA.asmx/DadosHidrometeorologicos?"
       "codEstacao=14990000&DataInicio=01/01/2024&DataFim=03/01/2024")
print("\nANA legacy telemetry service")
try:
    with urllib.request.urlopen(ana, timeout=60) as r:
        body = r.read().decode("utf-8", "replace")
    print("  HTTP", r.status, "| length", len(body), "| first 800 chars:\n", body[:800])
except Exception as e:
    print("  failed:", e.__class__.__name__, str(e)[:300])
