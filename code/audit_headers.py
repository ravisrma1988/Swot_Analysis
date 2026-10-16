"""Print row count, column count and header of every CSV in the USGS release,
and flag documented-vs-actual column differences (metadata digest vs header)."""
import csv, pathlib, re

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "usgs_paired"
digest = (ROOT / "docs" / "metadata_digest.md").read_text(encoding="utf-8")

documented = {}
for block in digest.split("## Entity: ")[1:]:
    name = block.split(" ", 1)[0]
    documented[name] = re.findall(r"^- \*\*(.+?)\*\*", block, flags=re.M)

for p in sorted(RAW.glob("*.csv")):
    with open(p, newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
        rows = sum(1 for _ in f)
    doc = documented.get(p.name, [])
    print(f"== {p.name}: {rows} rows, {len(header)} cols")
    print("   " + ", ".join(header))
    print(f"   in file, not documented: {[c for c in header if c not in doc]}")
    print(f"   documented, not in file: {[c for c in doc if c not in header]}")
