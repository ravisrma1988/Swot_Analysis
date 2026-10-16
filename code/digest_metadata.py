"""Digest the FGDC metadata XML of the USGS paired SWOT-gage release.

Writes docs/metadata_digest.md: abstract/supplemental info, data-quality
and lineage text (process steps), then every entity (file) with each
attribute's label, definition, units and domain. Pure parsing, no edits.
"""
import pathlib
import xml.etree.ElementTree as ET

ROOT = pathlib.Path(__file__).resolve().parents[1]
XML = ROOT / "data" / "raw" / "usgs_paired" / "SWOT_insitu_comparison_metadata.xml"
OUT = ROOT / "docs" / "metadata_digest.md"


def txt(el, path):
    found = el.find(path)
    return " ".join(found.text.split()) if found is not None and found.text else ""


def domain(attr):
    parts = []
    for dv in attr.findall("attrdomv"):
        for e in dv.findall("edom"):
            parts.append(f"{txt(e, 'edomv')} = {txt(e, 'edomvd')}")
        for r in dv.findall("rdom"):
            u = txt(r, "attrunit")
            parts.append(f"range [{txt(r, 'rdommin')}, {txt(r, 'rdommax')}]" + (f" {u}" if u else ""))
        if dv.find("udom") is not None:
            parts.append("udom: " + txt(dv, "udom"))
        if dv.find("codesetd") is not None:
            parts.append("codeset: " + txt(dv, "codesetd/codesetn"))
    return "; ".join(parts)


def main():
    root = ET.parse(XML).getroot()
    L = ["# Metadata digest: " + XML.name, ""]
    for label, path in [("Abstract", "idinfo/descript/abstract"),
                        ("Purpose", "idinfo/descript/purpose"),
                        ("Supplemental", "idinfo/descript/supplinf"),
                        ("Attribute accuracy", "dataqual/attracc/attraccr"),
                        ("Logical consistency", "dataqual/logic"),
                        ("Completeness", "dataqual/complete"),
                        ("Vertical accuracy", "dataqual/posacc/vertacc/vertaccr")]:
        L += [f"## {label}", txt(root, path) or "(none)", ""]
    L.append("## Lineage process steps")
    for i, p in enumerate(root.findall("dataqual/lineage/procstep"), 1):
        L += [f"{i}. [{txt(p, 'procdate')}] {txt(p, 'procdesc')}", ""]
    for ov in root.findall("eainfo/overview"):
        L += ["## Entity overview", txt(ov, "eaover"), txt(ov, "eadetcit"), ""]
    for det in root.findall("eainfo/detailed"):
        attrs = det.findall("attr")
        L += [f"## Entity: {txt(det, 'enttyp/enttypl')} ({len(attrs)} attributes)",
              txt(det, "enttyp/enttypd"), ""]
        for a in attrs:
            d = domain(a)
            L.append(f"- **{txt(a, 'attrlabl')}**: {txt(a, 'attrdef')}" + (f" | {d}" if d else ""))
        L.append("")
    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes, {len(L)} lines)")
    for det in root.findall("eainfo/detailed"):
        print(f"  {txt(det, 'enttyp/enttypl')}: {len(det.findall('attr'))} attributes")


if __name__ == "__main__":
    main()
