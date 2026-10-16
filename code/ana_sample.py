"""Parse a sample of ANA legacy telemetry (Rio Negro at Manaus, 14990000): fields, time step, stage values."""
import urllib.request
import xml.etree.ElementTree as ET
import pandas as pd

url = ("http://telemetriaws1.ana.gov.br/ServiceANA.asmx/DadosHidrometeorologicos?"
       "codEstacao=14990000&DataInicio=01/06/2024&DataFim=03/06/2024")
with urllib.request.urlopen(url, timeout=60) as r:
    root = ET.fromstring(r.read())
recs = [{c.tag.split('}')[-1]: (c.text or "").strip() for c in el}
        for el in root.iter() if el.tag.split('}')[-1] == "DadosHidrometereologicos" and len(el)]
df = pd.DataFrame(recs)
print("fields:", list(df.columns), "| records:", len(df))
print(df.head(8).to_string(index=False))
t = pd.to_datetime(df.DataHora, errors="coerce").sort_values()
print("time step (most common):", t.diff().value_counts().head(3).to_dict())
print("Nivel non-empty share:", (df.get("Nivel", pd.Series(dtype=str)) != "").mean())
