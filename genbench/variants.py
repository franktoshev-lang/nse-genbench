"""Prompt variants, layered on top of the verbatim baseline in prompts.py.

A variant is the group's prompt code with the text edits declared in
config/variants.yml applied to it, then loaded as its own module. So every
difference from the baseline is written down in one place, and the baseline
file itself is never touched. P0 has no edits: it is genbench.prompts.
"""
from __future__ import annotations

import types
from functools import lru_cache
from pathlib import Path

import yaml

from . import prompts as baseline

VARIANTS_CONFIG = Path(__file__).resolve().parents[1] / "config" / "variants.yml"
BASELINE_ID = "P0"


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


@lru_cache(maxsize=None)
def _build(variant_id: str, path: str) -> types.ModuleType:
    source = Path(baseline.__file__).read_text(encoding="utf-8")
    for old, new, count in edits_for(variant_id, path):
        found = source.count(old)
        if found != count:
            raise VariantError(f"{variant_id}: {old!r} occurs {found} times in the baseline, "
                               f"declared {count}")
        source = source.replace(old, new)
    module = types.ModuleType(f"genbench.prompts[{variant_id}]")
    module.__file__ = f"<prompt variant {variant_id}>"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def prompts_for(variant_id: str, path=VARIANTS_CONFIG) -> types.ModuleType:
    """The prompt module for a variant. P0 is genbench.prompts itself."""
    if variant_id == BASELINE_ID:
        edits_for(variant_id, path)          # still validates the config
        return baseline
    return _build(variant_id, str(path))
