"""Stage 1 - DATA. Load responses, and optionally ground truth, from files.

Nothing is generated here and nothing is shipped: every path is supplied at
run time. The group project's export format is not known yet, so each file
format is parsed by exactly one function registered in _PARSERS. Adding a
format means adding one function and one entry; validation stays shared.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

# The evidence types the group prompt's _append_evidence knows. A record with
# any other type would be left out of the prompt without a warning, so the
# loader rejects it instead.
EVIDENCE_TYPES = frozenset({"definite", "free_text_retrieval", "subtheme_candidate", "ambiguous"})


class DataError(ValueError):
    """The file exists but its contents cannot be used."""


@dataclass(frozen=True)
class Response:
    evidence_id: str
    document: str
    evidence_type: str

    def as_dict(self) -> dict:
        return asdict(self)


def _require_file(path, what: str) -> Path:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"{what} file not found: {p}. Nothing is generated as a fallback; "
            f"supply the path to an existing file."
        )
    return p


# ── format parsers: path -> list of raw records (dicts) ────────────────────
def _parse_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataError(f"{path}:{line_no}: not valid JSON ({exc.msg})") from exc
            if not isinstance(record, dict):
                raise DataError(f"{path}:{line_no}: expected one JSON object per line")
            records.append(record)
    return records


_PARSERS = {".jsonl": _parse_jsonl}


# ── public API ──────────────────────────────────────────────────────────────
def load_responses(path) -> list[Response]:
    """Read responses in file order. Only evidence_id, document and
    evidence_type are kept; any other fields in the file are ignored."""
    p = _require_file(path, "Responses")
    parser = _PARSERS.get(p.suffix.lower())
    if parser is None:
        raise DataError(
            f"unsupported responses format {p.suffix or '(no extension)'!r} for {p}; "
            f"supported: {', '.join(sorted(_PARSERS))}"
        )
    records = parser(p)
    if not records:
        raise DataError(f"{p}: no responses found")

    responses: list[Response] = []
    seen: set[str] = set()
    for n, record in enumerate(records, start=1):
        missing = [k for k in ("evidence_id", "document", "evidence_type")
                   if not str(record.get(k) or "").strip()]
        if missing:
            raise DataError(f"{p}: record {n} is missing {', '.join(missing)}")
        evidence_id = str(record["evidence_id"]).strip()
        evidence_type = str(record["evidence_type"]).strip()
        if evidence_type not in EVIDENCE_TYPES:
            raise DataError(
                f"{p}: record {n} ({evidence_id}) has evidence_type {evidence_type!r}; "
                f"the group prompt only uses {', '.join(sorted(EVIDENCE_TYPES))}"
            )
        if evidence_id in seen:
            raise DataError(f"{p}: duplicate evidence_id {evidence_id!r} (record {n})")
        seen.add(evidence_id)
        responses.append(Response(evidence_id, str(record["document"]), evidence_type))
    return responses


def load_truth(path) -> dict | None:
    """Optional ground truth for one slice. None when no path is given.

    JSON object with:
      planted         {theme: count}   themes present in the responses
      minority_theme  str              must be one of planted
      absent_theme    str              must not be one of planted
      labels          {evidence_id: theme | null}   optional, per response;
                                       null for low-information responses
      theme, id       str              optional
    """
    if path is None:
        return None
    p = _require_file(path, "Ground-truth")
    try:
        truth = json.loads(p.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise DataError(f"{p}: not valid JSON ({exc.msg})") from exc
    if not isinstance(truth, dict):
        raise DataError(f"{p}: expected a JSON object")

    planted = truth.get("planted")
    if not isinstance(planted, dict) or not planted:
        raise DataError(f"{p}: 'planted' must be a non-empty object of theme -> count")
    if not all(isinstance(v, int) and v >= 0 for v in planted.values()):
        raise DataError(f"{p}: every 'planted' count must be a non-negative integer")
    for key in ("minority_theme", "absent_theme"):
        if not isinstance(truth.get(key), str) or not truth[key].strip():
            raise DataError(f"{p}: '{key}' must be a theme name")
    if truth["minority_theme"] not in planted:
        raise DataError(f"{p}: minority_theme {truth['minority_theme']!r} is not in 'planted'")
    if truth["absent_theme"] in planted:
        raise DataError(f"{p}: absent_theme {truth['absent_theme']!r} is also in 'planted'")
    labels = truth.get("labels")
    if labels is not None and not isinstance(labels, dict):
        raise DataError(f"{p}: 'labels' must be an object of evidence_id -> theme or null")
    return truth
