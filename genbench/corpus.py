"""Planted-theme synthetic corpus generator.

The corpus IS the ground truth. Because the theme distribution is chosen
rather than discovered, coverage / minority-preservation / hallucination
become arithmetic instead of opinion.

Deterministic: same seed -> byte-identical output. No LLM involved.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import yaml

SLOT = re.compile(r"\{(\w+)\}")


@dataclass
class Response:
    evidence_id: str
    document: str
    evidence_type: str          # "definite" | "ambiguous"
    planted_theme: str | None   # None for noise
    language: str               # "nl" | "en"
    is_noise: bool


@dataclass
class Slice:
    id: str
    label: str
    responses: list[Response]
    planted: dict[str, int]     # theme -> count
    minority_theme: str
    absent_theme: str
    seed: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "seed": self.seed,
            "planted": self.planted,
            "minority_theme": self.minority_theme,
            "absent_theme": self.absent_theme,
            "n_responses": len(self.responses),
            "responses": [asdict(r) for r in self.responses],
        }


def _fill(template: str, fillers: dict[str, list[str]], rng: random.Random) -> str:
    def sub(m: re.Match) -> str:
        key = m.group(1)
        opts = fillers.get(key)
        return rng.choice(opts) if opts else m.group(0)
    return SLOT.sub(sub, template)


def _draw(templates: list[str], n: int, fillers: dict, rng: random.Random) -> list[str]:
    """Draw n filled templates, cycling so n may exceed the template count."""
    pool: list[str] = []
    while len(pool) < n:
        shuffled = templates[:]
        rng.shuffle(shuffled)
        pool += shuffled
    return [_fill(t, fillers, rng) for t in pool[:n]]


def build_slice(spec: dict, themes_cfg: dict, seed: int | None = None) -> Slice:
    used_seed = seed if seed is not None else spec.get("seed", 42)
    rng = random.Random(used_seed)
    by_name = {t["name"]: t for t in themes_cfg["themes"]}
    fillers = themes_cfg.get("fillers", {})

    plan: list[tuple[str | None, str, bool]] = []   # (theme, text, is_noise)
    planted: dict[str, int] = {}

    def add_theme(theme: str, count: int) -> None:
        if theme not in by_name:
            raise KeyError(f"unknown theme {theme!r}")
        for text in _draw(by_name[theme]["templates"], count, fillers, rng):
            plan.append((theme, text, False))
        planted[theme] = planted.get(theme, 0) + count

    add_theme(spec["dominant"]["theme"], spec["dominant"]["count"])
    for mid in spec.get("mid", []):
        add_theme(mid["theme"], mid["count"])
    add_theme(spec["minority"]["theme"], spec["minority"]["count"])

    for text in _draw(themes_cfg["noise"], spec.get("noise", 0), fillers, rng):
        plan.append((None, text, True))

    for text in _draw(themes_cfg["english"], spec.get("english", 0), fillers, rng):
        plan.append((None, text, False))   # English items carry no planted theme

    absent = spec["absent_theme"]
    if absent in planted:
        raise ValueError(f"absent_theme {absent!r} is also planted")

    rng.shuffle(plan)

    responses: list[Response] = []
    for i, (theme, text, is_noise) in enumerate(plan, start=1):
        lang = "en" if text in themes_cfg["english"] else "nl"
        responses.append(Response(
            evidence_id=f"E{i:04d}",
            document=text,
            evidence_type="definite",
            planted_theme=theme,
            language=lang,
            is_noise=is_noise,
        ))

    return Slice(
        id=spec["id"],
        label=spec["label"],
        responses=responses,
        planted=planted,
        minority_theme=spec["minority"]["theme"],
        absent_theme=absent,
        seed=used_seed,
    )


def load_configs(themes_path="config/themes.yml", slices_path="config/slices.yml"):
    themes = yaml.safe_load(Path(themes_path).read_text(encoding="utf-8"))
    slices = yaml.safe_load(Path(slices_path).read_text(encoding="utf-8"))
    return themes, slices


def write_slice(sl: Slice, out_dir: str | Path) -> tuple[Path, Path]:
    """JSONL of responses (pipeline-compatible) + a manifest of the ground truth."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    jsonl = out / f"{sl.id}.responses.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for r in sl.responses:
            f.write(json.dumps({
                "evidence_id": r.evidence_id,
                "document": r.document,
                "evidence_type": r.evidence_type,
            }, ensure_ascii=False) + "\n")
    manifest = out / f"{sl.id}.truth.json"
    d = sl.as_dict()
    manifest.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    return jsonl, manifest
