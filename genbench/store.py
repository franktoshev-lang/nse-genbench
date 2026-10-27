"""Where runs and human scores are kept: two append-only JSONL files in out/.

  results.jsonl  one line per run: the condition, the card, the automatic
                 scores and the tokens used (written by runs.execute)
  scores.jsonl   one line per human score. The latest line for a run counts;
                 earlier ones stay as history.

Both files hold model output with verbatim student quotes, so they are data:
out/ is never committed.
"""
from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

RESULTS_FILE = "results.jsonl"
SCORES_FILE = "scores.jsonl"
RATING_DIMENSIONS = ("usefulness", "trustworthiness", "completeness")
RATING_MIN, RATING_MAX = 1, 5


class ScoreError(ValueError):
    pass


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def append_results(out_dir, records: list[dict]) -> Path:
    path = Path(out_dir) / RESULTS_FILE
    _append_jsonl(path, records)
    return path


def load_results(out_dir) -> list[dict]:
    return _read_jsonl(Path(out_dir) / RESULTS_FILE)


def load_score_history(out_dir) -> list[dict]:
    return _read_jsonl(Path(out_dir) / SCORES_FILE)


def load_scores(out_dir) -> dict[str, dict]:
    """The current (latest) human score per run_id."""
    latest: dict[str, dict] = {}
    for rec in load_score_history(out_dir):
        latest[rec["run_id"]] = rec
    return latest


def save_score(out_dir, run_id: str, *, ratings: dict | None = None, claims: dict | None = None,
               note: str = "", scorer: str = "", known_run_ids: set[str] | None = None) -> dict:
    """Validate and append one human score for a run.

    ratings: {dimension: 1-5 or None}, judged separately, never blended.
    claims:  {manifest entry index: True (supported) / False (not supported)},
             the yes/no claim check from the officer protocol.
    """
    if known_run_ids is not None and run_id not in known_run_ids:
        raise ScoreError(f"unknown run {run_id!r}")
    clean_ratings = {}
    for dim, value in (ratings or {}).items():
        if dim not in RATING_DIMENSIONS:
            raise ScoreError(f"unknown rating {dim!r}; use {', '.join(RATING_DIMENSIONS)}")
        if value is not None and (not isinstance(value, int) or isinstance(value, bool)
                                  or not RATING_MIN <= value <= RATING_MAX):
            raise ScoreError(f"{dim} must be {RATING_MIN}-{RATING_MAX} or empty, got {value!r}")
        clean_ratings[dim] = value
    clean_claims = {}
    for index, value in (claims or {}).items():
        if not str(index).isdigit() or not isinstance(value, bool):
            raise ScoreError(f"claims must map an entry number to true/false, got {index!r}: {value!r}")
        clean_claims[str(index)] = value
    if not any(v is not None for v in clean_ratings.values()) and not clean_claims and not note.strip():
        raise ScoreError("nothing to save: give at least one rating, claim check or note")
    record = {"run_id": run_id, "scored_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "scorer": scorer.strip(), "ratings": clean_ratings, "claims": clean_claims,
              "note": note.strip()}
    _append_jsonl(Path(out_dir) / SCORES_FILE, [record])
    return record


def human_summary(score: dict | None) -> dict:
    """Numbers from one human score: mean rating per dimension and share of claims supported."""
    if not score:
        return {"scored": False, **{d: None for d in RATING_DIMENSIONS}, "claims_supported": None}
    ratings = score.get("ratings") or {}
    claims = list((score.get("claims") or {}).values())
    return {"scored": True, **{d: ratings.get(d) for d in RATING_DIMENSIONS},
            "claims_supported": round(sum(claims) / len(claims), 4) if claims else None}


def summary_rows(results: list[dict], scores: dict[str, dict]) -> list[dict]:
    """One row per run, newest first, for the results table."""
    rows = []
    for rec in results:
        s = rec.get("scores") or {}
        rows.append({
            "run_id": rec["run_id"], "created_utc": rec.get("created_utc"), "slice": rec.get("slice"),
            "model": rec.get("model_key"), "variant": rec.get("variant"), "sample": rec.get("sample"),
            "mode": rec.get("mode"), "valid_json": s.get("valid_json"), "coverage": s.get("coverage"),
            "minority_preserved": s.get("minority_preserved"),
            "hallucinated_absent": s.get("hallucinated_absent"),
            "off_theme_cited": s.get("off_theme_cited"), "evidence_validity": s.get("evidence_validity"),
            "evidence_support": s.get("evidence_support"), "words": s.get("words"),
            "parse_error": bool(rec.get("parse_error")),
            **human_summary(scores.get(rec["run_id"])),
        })
    return list(reversed(rows))


# What the matrix can show. "higher" says which direction is better.
MATRIX_METRICS = {
    "coverage": {"label": "Coverage", "higher": True},
    "minority_preserved": {"label": "Minority kept", "higher": True},
    "hallucinated_absent": {"label": "Hallucinated absent subtheme", "higher": False},
    "off_theme_cited": {"label": "Off-theme answers cited", "higher": False},
    "evidence_validity": {"label": "Evidence-ID validity", "higher": True},
    "evidence_support": {"label": "Evidence support", "higher": True},
    "valid_json": {"label": "Valid JSON", "higher": True},
    "usefulness": {"label": "Your rating: usefulness", "higher": True},
    "trustworthiness": {"label": "Your rating: trustworthiness", "higher": True},
    "completeness": {"label": "Your rating: completeness", "higher": True},
    "claims_supported": {"label": "Your claim checks: supported", "higher": True},
}


def matrix(results: list[dict], scores: dict[str, dict], slice_id: str | None = None) -> dict:
    """Every prompt x model combination with the mean of each metric."""
    rows = [r for r in summary_rows(results, scores) if slice_id in (None, "", r["slice"])]
    variants = sorted({r["variant"] for r in rows}, key=lambda v: (v != "P0", v))
    models = sorted({r["model"] for r in rows})
    cells = {}
    for v in variants:
        for m in models:
            group = [r for r in rows if r["variant"] == v and r["model"] == m]
            if not group:
                continue
            cell = {"n": len(group), "n_scored": sum(r["scored"] for r in group)}
            for metric in MATRIX_METRICS:
                values = [float(r[metric]) for r in group if r[metric] is not None]
                cell[metric] = round(statistics.mean(values), 4) if values else None
                if metric == "coverage" and len(values) >= 2:
                    cell["coverage_sd"] = round(statistics.stdev(values), 4)
            cells[f"{v}|{m}"] = cell
    return {"variants": variants, "models": models, "cells": cells, "metrics": MATRIX_METRICS}


def spend(results: list[dict], models_config: dict) -> dict:
    """Tokens billed per model (calls served from the cache are not counted),
    and euros where the models file gives prices."""
    models = models_config.get("models") or {}
    per_model: dict[str, dict] = {}
    for rec in results:
        tokens = rec.get("tokens") or {}
        if rec.get("request") is None:          # mock runs cost nothing
            continue
        m = per_model.setdefault(rec.get("model_key"), {"runs": 0, "prompt_tokens": 0,
                                                         "completion_tokens": 0, "reasoning_tokens": 0})
        m["runs"] += 1
        for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens"):
            m[k] += int(tokens.get(k) or 0)
    total_eur, priced = 0.0, True
    for key, m in per_model.items():
        spec = models.get(key) or {}
        p_in, p_out = spec.get("price_in_per_million"), spec.get("price_out_per_million")
        if p_in is None or p_out is None:
            m["eur"] = None
            priced = False
        else:
            m["eur"] = round(m["prompt_tokens"] * p_in / 1e6 + m["completion_tokens"] * p_out / 1e6, 4)
            total_eur += m["eur"]
    return {"per_model": per_model, "total_eur": round(total_eur, 4) if per_model else 0.0,
            "all_priced": priced, "budget_eur": models_config.get("budget_eur")}
