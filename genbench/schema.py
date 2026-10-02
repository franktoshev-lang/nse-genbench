"""The dashboard-card schema, mirrored from the NSE pipeline.

Key order here matches prompts.py DASHBOARD_JSON_SCHEMA exactly, including
`summary` first. Defect D-2: that ordering forces the model to state its
conclusion before writing any supporting analysis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

CARD_KEYS = (
    "summary",
    "positive_comments",
    "critical_comments",
    "student_suggestions",
    "subthemes",
    "subtheme_manifest",
)

LIST_KEYS = CARD_KEYS[1:]


@dataclass
class ValidationResult:
    ok: bool
    missing_keys: list[str] = field(default_factory=list)
    extra_keys: list[str] = field(default_factory=list)
    bad_types: list[str] = field(default_factory=list)
    bad_manifest: list[str] = field(default_factory=list)

    @property
    def problems(self) -> list[str]:
        out = []
        out += [f"missing:{k}" for k in self.missing_keys]
        out += [f"extra:{k}" for k in self.extra_keys]
        out += [f"type:{k}" for k in self.bad_types]
        out += [f"manifest:{m}" for m in self.bad_manifest]
        return out


def validate_card(card: Any) -> ValidationResult:
    """Structural validation only. Says nothing about content quality."""
    if not isinstance(card, dict):
        return ValidationResult(ok=False, bad_types=["<root not an object>"])

    missing = [k for k in CARD_KEYS if k not in card]
    extra = [k for k in card if k not in CARD_KEYS]
    bad_types: list[str] = []

    if "summary" in card and not isinstance(card["summary"], str):
        bad_types.append("summary")
    for k in LIST_KEYS:
        if k in card and not isinstance(card[k], list):
            bad_types.append(k)

    bad_manifest: list[str] = []
    for i, entry in enumerate(card.get("subtheme_manifest") or []):
        if not isinstance(entry, dict):
            bad_manifest.append(f"[{i}] not an object")
            continue
        for req in ("name", "description", "evidence_ids"):
            if req not in entry:
                bad_manifest.append(f"[{i}] missing {req}")
        ids = entry.get("evidence_ids")
        if ids is not None and not isinstance(ids, list):
            bad_manifest.append(f"[{i}] evidence_ids not a list")

    return ValidationResult(
        ok=not (missing or bad_types or bad_manifest),
        missing_keys=missing,
        extra_keys=extra,
        bad_types=bad_types,
        bad_manifest=bad_manifest,
    )


def card_text(card: dict, include_suggestions: bool = True) -> str:
    """All generated prose in one string, for keyword and language checks."""
    parts: list[str] = [str(card.get("summary") or "")]
    for k in ("positive_comments", "critical_comments", "subthemes"):
        parts += [str(x) for x in (card.get(k) or [])]
    if include_suggestions:
        parts += [str(x) for x in (card.get("student_suggestions") or [])]
    for entry in card.get("subtheme_manifest") or []:
        if isinstance(entry, dict):
            parts += [str(entry.get("name") or ""), str(entry.get("description") or "")]
    return "\n".join(p for p in parts if p)


def cited_ids(card: dict) -> list[str]:
    out: list[str] = []
    for entry in card.get("subtheme_manifest") or []:
        if isinstance(entry, dict):
            for i in entry.get("evidence_ids") or []:
                out.append(str(i))
    return out
