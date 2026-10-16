"""Match QC for ANA station -> SWORD reach matches (13 Oct 2026).

Repairs mojibake in ANA names (latin-1 bytes read as UTF-8 or the reverse), normalises names (accents, case, generic
prefixes), and compares ANA river names with SWORD river names (fuzzy ratio, token containment).
Status: match / unknown (no usable name on either side) / mismatch. Writes a stratified inspection sample.
Outputs: results/amazon_v1/{matches_qc.csv, qc_sample.csv, qc_report.txt}
"""
import difflib, io, pathlib, unicodedata
import numpy as np
import pandas as pd

OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "amazon_v1"
buf = io.StringIO()
GENERIC = {"RIO", "IGARAPE", "PARANA", "FURO", "BRACO", "CANAL", "DO", "DA", "DE", "DOS", "DAS", "E"}


def out(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); buf.write(s + "\n")


def fix(s):
    s = "" if pd.isna(s) else str(s)
    for enc_from, enc_to in (("latin-1", "utf-8"), ("cp1252", "utf-8")):
        try:
            t = s.encode(enc_from).decode(enc_to)
            if t != s:
                return t
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return s.replace("\ufffd", "")


def norm(s):
    s = unicodedata.normalize("NFKD", fix(s)).encode("ascii", "ignore").decode().upper()
    toks = [t for t in "".join(c if c.isalnum() else " " for c in s).split() if t not in GENERIC]
    return " ".join(toks)


def compare(ana, sword):
    a = norm(ana)
    variants = [norm(v) for v in str(sword).replace(",", ";").split(";")]
    variants = [v for v in variants if v and v != "NODATA"]
    if not a or not variants:
        return "unknown", np.nan
    r = max(difflib.SequenceMatcher(None, a, v).ratio() for v in variants)
    contained = any(a in v or v in a for v in variants)
    return ("match" if (r >= 0.8 or contained) else "mismatch"), r


def main():
    m = pd.read_csv(OUT / "matches.csv")
    m["ana_name_fixed"] = m.NomeRio.map(fix)
    m["station_name_fixed"] = m.NomeEstacao.map(fix) if "NomeEstacao" in m else ""
    res = [compare(a, s) for a, s in zip(m.NomeRio, m.river_name)]
    m["name_status"], m["name_ratio"] = [r[0] for r in res], [r[1] for r in res]
    m["amazon_box"] = m.lat.between(-20, 6) & m.lon.between(-80, -44)
    m.to_csv(OUT / "matches_qc.csv", index=False)
    out("name status, all matched stations:", m.name_status.value_counts().to_dict())
    out("name status, Amazon box:", m[m.amazon_box].name_status.value_counts().to_dict())
    out("distance km by status (median):", m.groupby("name_status").dist_km.median().round(3).to_dict())
    cols = ["code", "station_name_fixed", "ana_name_fixed", "river_name", "name_status", "name_ratio", "dist_km", "node_width", "lat", "lon"]
    sample = (m[m.amazon_box].groupby("name_status", group_keys=False)
              .apply(lambda g: g.sample(min(len(g), 15), random_state=0))[cols])
    sample.to_csv(OUT / "qc_sample.csv", index=False)
    out("\ninspection sample (Amazon box, up to 15 per status):")
    out(sample.round(3).to_string(index=False))
    (OUT / "qc_report.txt").write_text(buf.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
