"""List the files of the SWORD v17b Zenodo record (15299138) with sizes and checksums."""
import json, urllib.request

with urllib.request.urlopen("https://zenodo.org/api/records/15299138", timeout=60) as r:
    rec = json.load(r)
print(rec["metadata"]["title"], "| version", rec["metadata"].get("version"), "| doi", rec.get("doi"))
for f in rec["files"]:
    print(f"{f['key']:45s} {f['size'] / 1e9:7.2f} GB  {f['checksum']}  {f['links']['self']}")
