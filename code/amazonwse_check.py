"""Inspect the AmazonWSE repository via the GitHub API: licence, README, top-level and data-related files."""
import base64, json, urllib.request

API = "https://api.github.com/repos/rubencart/AmazonWSE"


def get(url):
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "swot-check"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


repo = get(API)
print("repo:", repo.get("full_name"), "| licence:", (repo.get("license") or {}).get("spdx_id"), "| updated:", repo.get("pushed_at"),
      "| size KB:", repo.get("size"), "| default branch:", repo.get("default_branch"))
tree = get(f"{API}/git/trees/{repo['default_branch']}?recursive=1")
paths = [t["path"] for t in tree.get("tree", [])]
print(f"{len(paths)} paths; top level:", sorted({p.split('/')[0] for p in paths}))
keys = ("data", "gauge", "ana", "insitu", "in_situ", "download", "zenodo", "swot", "readme", "license")
print("data-related paths (first 60):")
for p in [p for p in paths if any(k in p.lower() for k in keys)][:60]:
    print("  ", p)
try:
    readme = get(f"{API}/readme")
    text = base64.b64decode(readme["content"]).decode("utf-8", "replace")
    print("\n--- README (first 4000 chars) ---\n" + text[:4000])
except Exception as e:
    print("README not available:", e)
