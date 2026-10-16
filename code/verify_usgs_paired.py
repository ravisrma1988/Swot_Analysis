"""Verify the USGS paired SWOT-gage release against the ScienceBase record.

Reads expected names, sizes and MD5s from docs/sciencebase_item_<id>.json,
checks every file in data/raw/usgs_paired, and writes
docs/usgs_paired_MANIFEST.tsv (size, MD5, SHA-256, status) and
docs/usgs_paired.md5 (md5sum -c format). Exit code 1 on any mismatch,
missing file or unexpected extra file. Re-run before analysis jobs.
"""
import datetime, hashlib, json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "usgs_paired"
DOCS = ROOT / "docs"
REC = DOCS / "sciencebase_item_688ce6bdd4be02086276c296.json"


def digests(path):
    md5, sha = hashlib.md5(), hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            md5.update(chunk)
            sha.update(chunk)
    return md5.hexdigest(), sha.hexdigest()


def main():
    rec = json.loads(REC.read_text(encoding="utf-8"))
    expected = {f["name"]: f for f in rec["files"]}
    present = {p.name for p in RAW.iterdir() if p.is_file()}
    rows, bad = [], 0
    for name, f in sorted(expected.items()):
        p = RAW / name
        if not p.exists():
            rows.append((name, "", f["md5"], "", "", "MISSING")); bad += 1; continue
        size = p.stat().st_size
        md5, sha = digests(p)
        ok = size == f["size"] and md5 == f["md5"]
        bad += not ok
        rows.append((name, str(size), f["md5"], md5, sha, "OK" if ok else "MISMATCH"))
    for extra in sorted(present - set(expected)):
        rows.append((extra, "", "", "", "", "UNEXPECTED")); bad += 1
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    head = ("# source: doi:10.5066/P1FE9W9E, ScienceBase item " + rec["id"]
            + ", record last_updated " + rec["last_updated"] + "; verified " + stamp + "\n"
            + "file\tbytes\tmd5_expected\tmd5_observed\tsha256\tstatus\n")
    (DOCS / "usgs_paired_MANIFEST.tsv").write_text(head + "".join("\t".join(r) + "\n" for r in rows))
    (DOCS / "usgs_paired.md5").write_text("".join(f["md5"] + "  " + n + "\n" for n, f in sorted(expected.items())))
    for r in rows:
        print(f"{r[5]:10} {r[1]:>11}  {r[0]}")
    print(f"{len(rows) - bad}/{len(rows)} OK")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
