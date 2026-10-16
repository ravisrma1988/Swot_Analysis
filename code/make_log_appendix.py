"""Build the automatic appendix of the experiment log and assemble docs/EXPERIMENT_LOG.md (part 1 + appendix).
Appendix: environment versions, code inventory (lines, modified time, docstring summary), results and log inventories
(sizes, modified times), raw-data inventory, sha256 of key outputs."""
import ast, datetime, hashlib, importlib, pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external" / "pylib"))
IST = datetime.timedelta(hours=5, minutes=30)


def ts(p):
    u = datetime.datetime.fromtimestamp(p.stat().st_mtime, datetime.timezone.utc)
    return f"{u:%Y-%m-%d %H:%M} UTC ({u + IST:%H:%M} IST)"


def size(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


L = ["\n## Appendix A. Environment (generated " + datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC") + ")"]
for m in ("pandas", "numpy", "sklearn", "xgboost", "cupy", "torch", "pyarrow", "h5py", "scipy"):
    try:
        L.append(f"- {m} {importlib.import_module(m).__version__}")
    except Exception as e:
        L.append(f"- {m}: not importable ({e.__class__.__name__})")
smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout.splitlines()
L += ["- " + l.strip("| ").strip() for l in smi if "Driver Version" in l or "B200" in l][:2]

L.append("\n## Appendix B. Code inventory (code/)")
L.append("| File | Lines | Modified | Summary |\n|---|---|---|---|")
for p in sorted((ROOT / "code").glob("*")):
    if p.suffix not in (".py", ".sh"):
        continue
    text = p.read_text(encoding="utf-8", errors="replace")
    summ = ""
    if p.suffix == ".py":
        try:
            doc = ast.get_docstring(ast.parse(text)) or ""
            summ = doc.strip().splitlines()[0] if doc.strip() else ""
        except SyntaxError:
            summ = "(not parsable)"
    else:
        summ = next((l.lstrip("# ").strip() for l in text.splitlines()[1:4] if l.startswith("#")), "")
    L.append(f"| {p.name} | {len(text.splitlines())} | {ts(p)} | {summ.replace('|', '/')[:150]} |")

L.append("\n## Appendix C. Results inventory (results/)")
for d in sorted(p for p in (ROOT / "results").iterdir() if p.is_dir()):
    files = sorted(f for f in d.rglob("*") if f.is_file())
    L.append(f"\n**{d.name}/** ({len(files)} files, {size(sum(f.stat().st_size for f in files))})")
    for f in files[:40]:
        L.append(f"- {f.relative_to(d)}: {size(f.stat().st_size)}, {ts(f)}")
    if len(files) > 40:
        L.append(f"- ... {len(files) - 40} more files")

L.append("\n## Appendix D. Logs (logs/)")
L.append("| Log | Size | Modified |\n|---|---|---|")
for p in sorted((ROOT / "logs").glob("*.log")):
    L.append(f"| {p.name} | {size(p.stat().st_size)} | {ts(p)} |")

L.append("\n## Appendix E. Raw data inventory (data/raw/)")
for d in sorted(p for p in (ROOT / "data" / "raw").rglob("*") if p.is_dir()):
    files = [f for f in d.iterdir() if f.is_file()]
    if files:
        L.append(f"- {d.relative_to(ROOT)}: {len(files)} files, {size(sum(f.stat().st_size for f in files))}")

L.append("\n## Appendix F. sha256 of key outputs")
keys = list((ROOT / "data" / "processed").glob("*.parquet")) + list((ROOT / "results" / "final").glob("*summary*.csv")) \
    + [ROOT / "results" / "final" / "model_config.json", ROOT / "results" / "final" / "condensed_all.txt",
       ROOT / "docs" / "design_note_v0.6.md"]
for p in sorted(set(keys)):
    if p.exists():
        L.append(f"- {p.relative_to(ROOT)} ({size(p.stat().st_size)}): `{sha(p)}`")

app = "\n".join(L) + "\n"
(ROOT / "docs" / "EXPERIMENT_LOG_appendix.md").write_text(app, encoding="utf-8")
part1 = (ROOT / "docs" / "EXPERIMENT_LOG_part1.md").read_text(encoding="utf-8")
(ROOT / "docs" / "EXPERIMENT_LOG.md").write_text(part1 + app, encoding="utf-8")
print(f"EXPERIMENT_LOG.md written: {len((part1 + app).splitlines())} lines")
