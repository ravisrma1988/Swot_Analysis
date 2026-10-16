"""Time ANA legacy telemetry requests of different window lengths for one station (Rio Negro at Manaus)."""
import time, urllib.request

ANA = "http://telemetriaws1.ana.gov.br/ServiceANA.asmx/DadosHidrometeorologicos?codEstacao=14990000"
for a, b, label in [("10/03/2025", "10/03/2025", "1 day"), ("09/03/2025", "11/03/2025", "3 days"),
                    ("01/03/2025", "30/03/2025", "30 days"), ("10/04/2025", "10/04/2025", "1 day (repeat)")]:
    t = time.time()
    with urllib.request.urlopen(f"{ANA}&DataInicio={a}&DataFim={b}", timeout=300) as r:
        n = len(r.read())
    print(f"{label:15s}: {time.time() - t:6.1f} s, {n / 1e6:.2f} MB", flush=True)
