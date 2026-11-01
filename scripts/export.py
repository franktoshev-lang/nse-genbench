#!/usr/bin/env python3
"""Export runs to an Excel workbook, to read and score them by hand.

  python scripts/export.py [--slice EA-60] [--model KEY] [--variant ID]
                           [--scored yes|no] [--out FILE]

The filters are the ones on the Results tab; with none, every run is
exported. The file lands in out/exports/ unless --out says otherwise.
The UI's "Export to Excel" button makes the same file.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from genbench.export import build_workbook, export_filename
from genbench.runs import current_settings, out_dir
from genbench.store import load_results, load_scores, summary_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--slice")
    parser.add_argument("--model", help="model key from config/models.yml, or mock:good etc.")
    parser.add_argument("--variant")
    parser.add_argument("--scored", choices=("yes", "no"))
    parser.add_argument("--out", help="where to write the .xlsx (default: out/exports/)")
    args = parser.parse_args(argv)

    settings = current_settings(ROOT)
    out = out_dir(settings, ROOT)
    results, scores = load_results(out), load_scores(out)
    rows = summary_rows(results, scores)
    filters = {"slice": args.slice, "model": args.model, "variant": args.variant, "scored": args.scored}
    for field in ("slice", "model", "variant"):
        if filters[field]:
            rows = [r for r in rows if r[field] == filters[field]]
    if args.scored:
        rows = [r for r in rows if r["scored"] == (args.scored == "yes")]
    wanted = {r["run_id"] for r in rows}
    records = [r for r in results if r["run_id"] in wanted]
    if not records:
        print("No runs match these filters; nothing exported.", file=sys.stderr)
        return 2
    records.sort(key=lambda r: (str(r.get("slice")), str(r.get("model_key")), str(r.get("variant")),
                                str(r.get("created_utc")), r.get("sample") or 0))
    body = build_workbook(records, scores, root=ROOT, settings=settings, out_dir=out, filters=filters)
    path = Path(args.out) if args.out else out / "exports" / export_filename(filters)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    print(f"Exported {len(records)} run(s) to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
