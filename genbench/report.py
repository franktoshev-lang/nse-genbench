"""Plain-text results table. No plotting dependency."""
from __future__ import annotations

from typing import Sequence


def table(rows: Sequence[dict], cols: Sequence[str] | None = None) -> str:
    if not rows:
        return "(no rows)"
    cols = list(cols or rows[0].keys())
    def fmt(v):
        if v is None:
            return "-"
        if isinstance(v, bool):
            return "yes" if v else "no"
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)
    widths = {c: max(len(c), *(len(fmt(r.get(c))) for r in rows)) for c in cols}
    head = "  ".join(c.ljust(widths[c]) for c in cols)
    rule = "  ".join("-" * widths[c] for c in cols)
    body = "\n".join("  ".join(fmt(r.get(c)).ljust(widths[c]) for c in cols) for r in rows)
    return f"{head}\n{rule}\n{body}"
