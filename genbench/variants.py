"""Prompt variants, layered on top of the verbatim baseline in prompts.py.

A variant is the group's prompt code with the text edits declared in
config/variants.yml applied to it, then loaded as its own module. So every
difference from the baseline is written down in one place, and the baseline
file itself is never touched. P0 has no edits: it is genbench.prompts.

Variants can be added and removed from the UI. A new variant is only saved
if every edit matches the baseline code exactly as often as declared, the
edited code still runs, and every edit shows up in a prompt the model sees.
"""
from __future__ import annotations

import difflib
import re
import types
from functools import lru_cache
from pathlib import Path

import yaml

from . import prompts as baseline

VARIANTS_CONFIG = Path(__file__).resolve().parents[1] / "config" / "variants.yml"
BASELINE_ID = "P0"
VARIANT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,39}$")
PROMPT_KINDS = ("single", "map", "reduce")

# Small fixed inputs, only used to show what a variant's prompts look like.
EXAMPLE_DOCS = [
    {"evidence_id": "E0001", "document": "De feedback komt pas na zes weken.", "evidence_type": "definite"},
    {"evidence_id": "E0002", "document": "Er is maar één herkansingsmoment.", "evidence_type": "definite"},
    {"evidence_id": "E0003", "document": "Het rooster wordt vaak aangepast.", "evidence_type": "ambiguous"},
]
EXAMPLE_SUMMARIES = [
    {"summary": "Students say feedback arrives late.", "positive_comments": [],
     "critical_comments": ["Feedback arrives weeks after submission"], "student_suggestions": [],
     "subthemes": ["Late feedback"], "subtheme_manifest": [
         {"name": "Late feedback", "description": "Feedback takes weeks", "evidence_ids": ["E0001"]}],
     "batch_number": 1, "source_document_count": 60},
]

HEADER = """\
# Prompt variants for the generation benchmark.
#
# P0 is the group pipeline's prompt code, verbatim (genbench/prompts.py): the
# baseline control. Every other variant is P0 with the text edits listed under
# `replace` applied to that code, and changes exactly one `dimension`.
#
# Each edit is [old, new, count]: `old` must occur exactly `count` times in the
# baseline code, so an edit can never touch more places than declared.
# tests/test_variants.py checks this, and that a variant's rendered prompts
# equal P0's with the declared edits and nothing else. Variants added in the
# UI are written to this file.

"""


class VariantError(ValueError):
    pass


def load_variants(path=VARIANTS_CONFIG) -> dict[str, dict]:
    variants = (yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}).get("variants") or {}
    if BASELINE_ID not in variants:
        raise VariantError(f"{path}: the baseline variant {BASELINE_ID!r} is missing")
    for variant_id, spec in variants.items():
        if not isinstance(spec.get("dimension"), str) or not spec["dimension"].strip():
            raise VariantError(f"{variant_id}: 'dimension' must be one name; a variant changes one thing")
        edits = spec.get("replace") or []
        if variant_id == BASELINE_ID and edits:
            raise VariantError(f"{BASELINE_ID} is the verbatim baseline and cannot have edits")
        for edit in edits:
            if (not isinstance(edit, list) or len(edit) != 3 or not isinstance(edit[0], str)
                    or not edit[0] or not isinstance(edit[1], str)
                    or not isinstance(edit[2], int) or edit[2] < 1):
                raise VariantError(f"{variant_id}: each edit must be [old, new, count], got {edit!r}")
    return variants


def edits_for(variant_id: str, path=VARIANTS_CONFIG) -> list[tuple[str, str, int]]:
    variants = load_variants(path)
    if variant_id not in variants:
        raise VariantError(f"unknown variant {variant_id!r}; choose from {', '.join(variants)}")
    return [tuple(e) for e in variants[variant_id].get("replace") or []]


def baseline_source() -> str:
    return Path(baseline.__file__).read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def _build(name: str, edits: tuple[tuple[str, str, int], ...]) -> types.ModuleType:
    source = baseline_source()
    for old, new, count in edits:
        found = source.count(old)
        if found != count:
            raise VariantError(f"{name}: {old!r} occurs {found} times in the baseline, declared {count}")
        source = source.replace(old, new)
    module = types.ModuleType(f"genbench.prompts[{name}]")
    module.__file__ = f"<prompt variant {name}>"
    try:
        exec(compile(source, module.__file__, "exec"), module.__dict__)
    except Exception as exc:
        raise VariantError(f"{name}: the edited prompt code no longer runs ({exc})") from exc
    return module


def prompts_for(variant_id: str, path=VARIANTS_CONFIG) -> types.ModuleType:
    """The prompt module for a variant. P0 is genbench.prompts itself."""
    edits = edits_for(variant_id, path)          # also validates the config
    if variant_id == BASELINE_ID:
        return baseline
    return _build(variant_id, tuple(edits))


def render(module, kind: str, theme: str) -> str:
    """One prompt of a given kind, built from the small example inputs."""
    if kind == "single":
        return module.build_prompt(theme, EXAMPLE_DOCS)
    if kind == "map":
        return module.build_batch_summary_prompt(theme, EXAMPLE_DOCS, batch_number=1, total_batches=4)
    if kind == "reduce":
        return module.build_reduce_prompt(theme, EXAMPLE_SUMMARIES, source_document_count=240)
    raise VariantError(f"kind must be one of {PROMPT_KINDS}")


def word_diff(before: str, after: str) -> list[dict]:
    """[{"op": "same" | "del" | "ins", "text": ...}] at word level."""
    a, b = re.split(r"(\s+)", before), re.split(r"(\s+)", after)
    out: list[dict] = []

    def add(op: str, text: str) -> None:
        if not text:
            return
        if out and out[-1]["op"] == op:
            out[-1]["text"] += text
        else:
            out.append({"op": op, "text": text})

    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            add("same", "".join(a[i1:i2]))
        else:
            add("del", "".join(a[i1:i2]))
            add("ins", "".join(b[j1:j2]))
    return out


def check_edits(edits: list[tuple[str, str]], theme: str = "Examination & Assessment"
                ) -> tuple[list[tuple[str, str, int]], list[str]]:
    """Count each edit in the baseline code and check it reaches a prompt.
    Returns (edits with counts, problems)."""
    source = baseline_source()
    rendered = [render(baseline, kind, theme) for kind in PROMPT_KINDS]
    counted, problems = [], []
    if not edits:
        problems.append("add at least one change")
    for old, new in edits:
        if not old:
            problems.append("a change has no text to find")
            continue
        if old == new:
            problems.append(f"{old!r} is replaced by the same text")
            continue
        found = source.count(old)
        if found == 0:
            problems.append(f"{old!r} is not in the baseline prompt; copy it exactly from the preview")
            continue
        if not any(old in text for text in rendered):
            problems.append(f"{old!r} is in the code but not in any prompt the model sees")
            continue
        counted.append((old, new, found))
    return counted, problems


def preview(edits: list[tuple[str, str]], kind: str = "single",
            theme: str = "Examination & Assessment") -> dict:
    """What an unsaved variant would send, compared with P0."""
    counted, problems = check_edits(edits, theme)
    before = render(baseline, kind, theme)
    if problems or not counted:
        return {"problems": problems, "baseline": before, "diff": word_diff(before, before)}
    try:
        after = render(_build("preview", tuple(counted)), kind, theme)
    except VariantError as exc:
        return {"problems": [str(exc)], "baseline": before, "diff": word_diff(before, before)}
    return {"problems": [], "baseline": before, "diff": word_diff(before, after),
            "edits": [list(e) for e in counted]}


def _write(variants: dict, path) -> None:
    body = yaml.safe_dump({"variants": variants}, sort_keys=False, allow_unicode=True,
                          default_flow_style=None, width=100)
    Path(path).write_text(HEADER + body, encoding="utf-8", newline="\n")


def save_variant(variant_id: str, dimension: str, description: str,
                 edits: list[tuple[str, str]], path=VARIANTS_CONFIG) -> dict:
    """Add a new variant. Existing variants are never edited in place, because
    past runs refer to them; delete and add under a new name instead."""
    variants = load_variants(path)
    if not VARIANT_ID.match(variant_id or ""):
        raise VariantError("name: letters, digits, '.', '_' or '-', up to 40 characters")
    if variant_id in variants:
        raise VariantError(f"{variant_id!r} already exists; pick a new name")
    if not dimension.strip() or "," in dimension:
        raise VariantError("dimension: the one thing this variant changes, as one name")
    counted, problems = check_edits(edits)
    if problems:
        raise VariantError("; ".join(problems))
    _build(variant_id, tuple(counted))           # must still run
    variants[variant_id] = {"dimension": dimension.strip(), "description": description.strip(),
                            "replace": [list(e) for e in counted]}
    _write(variants, path)
    return variants[variant_id]


def delete_variant(variant_id: str, path=VARIANTS_CONFIG) -> None:
    """Remove a variant from the list. Runs made with it keep a copy of its edits."""
    variants = load_variants(path)
    if variant_id == BASELINE_ID:
        raise VariantError("P0 is the baseline control and cannot be removed")
    if variant_id not in variants:
        raise VariantError(f"unknown variant {variant_id!r}")
    del variants[variant_id]
    _write(variants, path)
