"""Time long ANA legacy telemetry windows (90 and 365 days) for one station; check the record span returned."""
import time, urllib.request
import xml.etree.ElementTree as ET

ANA = "http://telemetriaws1.ana.gov.br/ServiceANA.asmx/DadosHidrometeorologicos?codEstacao=14990000"
for a, b, label in [("01/01/2025", "31/03/2025", "90 days"), ("01/01/2024", "31/12/2024", "365 days")]:
    t = time.time()
    try:
        with urllib.request.urlopen(f"{ANA}&DataInicio={a}&DataFim={b}", timeout=600) as r:
            body = r.read()
        stamps = [el.text for el in ET.fromstring(body).iter() if el.tag.endswith("DataHora") and el.text]
        print(f"{label:9s}: {time.time() - t:6.1f} s, {len(body) / 1e6:.2f} MB, {len(stamps)} records, "
              f"span {min(stamps) if stamps else '-'} .. {max(stamps) if stamps else '-'}", flush=True)
    except Exception as e:
        print(f"{label:9s}: failed after {time.time() - t:.1f} s: {e.__class__.__name__} {str(e)[:200]}", flush=True)
