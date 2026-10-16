"""Rebuild the freeze manifest after the experiment log was regenerated (keeps every docs checksum current)."""
import hashlib, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


groups = [ROOT / "code", ROOT / "data" / "processed", ROOT / "docs", ROOT / "frozen"] + [
    ROOT / "results" / d for d in ("final", "audit_response", "external_v2", "external_v2_tau0.5", "external_v2_tau0.1",
                                    "external_v2_node", "external_diverse", "real_shift_v9", "real_shift_v8",
                                    "robustness_v6", "shift_v7", "sword_domain_v2", "sword_domain_v1", "sword_support_v1")]
manifest = DOCS / "FREEZE_MANIFEST_2026-10-14.tsv"
lines = ["path\tbytes\tsha256"]
for g in groups:
    if g.exists():
        for f in sorted(x for x in g.rglob("*") if x.is_file() and x != manifest):
            lines.append(f"{f.relative_to(ROOT)}\t{f.stat().st_size}\t{sha(f)}")
manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"manifest rebuilt: {len(lines) - 1} files; sha256 {sha(manifest)}")
