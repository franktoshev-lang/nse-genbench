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

from .rubric import HEADINGS, RUBRIC_VERSION, checks_from_score, heading_scores, rank_key, valid_key

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


def save_score(out_dir, run_id: str, *, checks: dict | None = None, ratings: dict | None = None,
               claims: dict | None = None, note: str = "", scorer: str = "",
               known_run_ids: set[str] | None = None) -> dict:
    """Validate and append one human score for a run.

    checks:  {item key: True (yes) / False (no)}, the scoring checklist in
             genbench/rubric.py. This is what the UI saves.
    ratings: {dimension: 1-5 or None}. Older scores only; kept readable.
    claims:  {manifest entry index: True / False}. Older scores only; read as T1.
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
    clean_checks = {}
    for key, value in (checks or {}).items():
        if value is None:
            continue
        if not valid_key(key) or not isinstance(value, bool):
            raise ScoreError(f"checks must map a checklist item to true/false, got {key!r}: {value!r}")
        clean_checks[str(key)] = value
    clean_claims = {}
    for index, value in (claims or {}).items():
        if not str(index).isdigit() or not isinstance(value, bool):
            raise ScoreError(f"claims must map an entry number to true/false, got {index!r}: {value!r}")
        clean_claims[str(index)] = value
    if (not clean_checks and not any(v is not None for v in clean_ratings.values())
            and not clean_claims and not note.strip()):
        raise ScoreError("nothing to save: answer at least one check, or write a note")
    record = {"run_id": run_id, "scored_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "scorer": scorer.strip(), "rubric": RUBRIC_VERSION, "checks": clean_checks,
              "ratings": clean_ratings, "claims": clean_claims, "note": note.strip()}
    _append_jsonl(Path(out_dir) / SCORES_FILE, [record])
    return record


def human_summary(score: dict | None) -> dict:
    """Numbers from one human score: the checklist's share of "yes" per heading
    (trustworthy, complete, useful), and the share of subthemes whose cited
    answers support them (check T1)."""
    if not score:
        return {"scored": False, **{h: None for h in HEADINGS}, "checks_answered": 0,
                "claims_supported": None}
    checks = checks_from_score(score)
    t1 = [v for k, v in checks.items() if k.startswith("T1:")]
    return {"scored": True, **heading_scores(checks),
            "claims_supported": round(sum(t1) / len(t1), 4) if t1 else None}


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
    "trustworthy": {"label": "Your checklist: trustworthy", "higher": True},
    "complete": {"label": "Your checklist: complete", "higher": True},
    "useful": {"label": "Your checklist: useful", "higher": True},
    "claims_supported": {"label": "Your checklist: subthemes supported (T1)", "higher": True},
}
HUMAN_METRICS = ("trustworthy", "complete", "useful", "claims_supported")


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
    return {"variants": variants, "models": models, "cells": cells, "metrics": MATRIX_METRICS,
            "ranking": ranking(rows)}


def ranking(rows: list[dict]) -> list[dict]:
    """Every prompt x model combination with at least one scored run, ranked by
    the rule fixed before scoring: trustworthy, then complete, then useful
    (each the mean over the combination's scored runs)."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["variant"], r["model"]), []).append(r)
    out = []
    for (variant, model), group in groups.items():
        scored = [r for r in group if r["scored"]]
        if not scored:
            continue
        entry = {"variant": variant, "model": model, "n": len(group), "n_scored": len(scored)}
        for h in HEADINGS:
            values = [r[h] for r in scored if r[h] is not None]
            entry[h] = round(statistics.mean(values), 4) if values else None
        out.append(entry)
    out.sort(key=lambda e: (*rank_key(e), e["variant"], e["model"]))
    for i, entry in enumerate(out, start=1):
        entry["rank"] = i
    return out


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
