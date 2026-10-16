"""Resumable, verified parallel download of one Zenodo file (written 13 Oct 2026 after curl restarted from zero).

Each of N byte ranges is fetched into its own part file. A chunk is written only if the server answers 206 with a
Content-Range that starts exactly where the part file ends; anything else (200 full body, wrong offset, network error)
is discarded and retried with backoff. Parts are then joined and the MD5 from the Zenodo API is checked.
Usage: python3 download_range.py <zenodo_record_id> <file_key> <out_dir> [parts]
"""
import concurrent.futures as cf, hashlib, json, os, pathlib, sys, time
import requests

rec, key, out_dir = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
parts = int(sys.argv[4]) if len(sys.argv) > 4 else 6
meta = requests.get(f"https://zenodo.org/api/records/{rec}", timeout=60).json()
f = next(x for x in meta["files"] if x["key"] == key)
size, md5_expected, url = f["size"], f["checksum"].split(":")[1], f["links"]["self"]
out = out_dir / key
print(f"{key}: {size} bytes, md5 {md5_expected}, {parts} parts", flush=True)
bounds = [(i * size // parts, (i + 1) * size // parts - 1) for i in range(parts)]


def fetch(i):
    start, end = bounds[i]
    part = out_dir / f"{key}.part{i:02d}"
    tries = 0
    while True:
        have = part.stat().st_size if part.exists() else 0
        if have == end - start + 1:
            return i, have
        if have > end - start + 1:
            part.unlink(); continue
        pos = start + have
        try:
            r = requests.get(url, headers={"Range": f"bytes={pos}-{end}"}, stream=True, timeout=(30, 120))
            cr = r.headers.get("Content-Range", "")
            if r.status_code != 206 or not cr.startswith(f"bytes {pos}-"):
                raise IOError(f"unexpected response {r.status_code} {cr!r}")
            with open(part, "ab") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
            tries = 0
        except Exception as e:
            tries += 1
            if tries > 200:
                raise RuntimeError(f"part {i} failed: {e}")
            time.sleep(min(60, 5 * tries))
            print(f"part {i}: retry {tries} at byte {pos} ({e.__class__.__name__}: {str(e)[:80]})", flush=True)


t0 = time.time()
with cf.ThreadPoolExecutor(parts) as ex:
    for i, n in ex.map(fetch, range(parts)):
        print(f"part {i} complete ({n} bytes) at {time.time() - t0:.0f} s", flush=True)
h = hashlib.md5()
with open(out, "wb") as dst:
    for i in range(parts):
        p = out_dir / f"{key}.part{i:02d}"
        with open(p, "rb") as src:
            for chunk in iter(lambda: src.read(1 << 24), b""):
                h.update(chunk); dst.write(chunk)
ok = h.hexdigest() == md5_expected and out.stat().st_size == size
print(f"joined {out.stat().st_size} bytes; md5 {h.hexdigest()} {'OK' if ok else 'MISMATCH'}", flush=True)
if ok:
    for i in range(parts):
        (out_dir / f"{key}.part{i:02d}").unlink()
sys.exit(0 if ok else 1)
