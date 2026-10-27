"""Single-pass and map-reduce orchestration, mirroring the group pipeline.

Source: github.com/r-huijts/NSE_Applied_GenAI @ 6a48c37
  src/config/settings.py
      HIERARCHICAL_RAG_BATCH_DOCUMENTS = 60
      HIERARCHICAL_RAG_MAX_DOCUMENTS   = 240
  src/pipeline/04_generation/service.py
      _select_theme_documents      keeps evidence[:240]; the rest never reaches the model
      _document_batches            consecutive batches of 60
      _generate_hierarchical_json  one batch -> build_prompt; more -> one
                                   build_batch_summary_prompt per batch, then
                                   build_reduce_prompt over the parsed batch JSON,
                                   each tagged with batch_number and
                                   source_document_count

In map-reduce the prompt, model and answer stages repeat once per batch and
once more for the reduce step, because the reduce prompt is built from the
model's batch answers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from . import prompts
from .data import Response

BATCH_DOCUMENTS = 60
MAX_DOCUMENTS = 240
MODES = ("single", "mapreduce")


class AnswerParseError(RuntimeError):
    """The group's parse_llm_json could not recover a JSON object."""

    def __init__(self, step: str, raw: str, cause: Exception | None = None,
                 calls: list[dict] | None = None) -> None:
        super().__init__(f"{step}: could not parse the model answer as a JSON object ({cause})")
        self.step = step
        self.raw = raw
        self.calls = calls or []    # the calls made so far, so their tokens still count


def group_default_mode(n_analyzed: int) -> str:
    """What the group pipeline does with this many responses after the cap."""
    return "single" if n_analyzed <= BATCH_DOCUMENTS else "mapreduce"


@dataclass
class Generation:
    card: dict
    mode: str
    n_supplied: int
    n_analyzed: int
    calls: list[dict] = field(default_factory=list)


def generate_card(runner, theme: str, responses: list[Response], *,
                  mode: str | None = None, prompt_module=None,
                  log: Callable[[str], None] = print) -> Generation:
    """Turn responses into one dashboard card the way the group pipeline does.

    mode=None follows the group's own choice. Forcing a mode that the group
    would not pick for this many responses is allowed but changes the
    condition, so the caller should report it. prompt_module is a variant
    from genbench.variants; the default is the verbatim baseline.
    """
    pm = prompt_module or prompts
    evidence = [r.as_dict() for r in responses]
    selected = evidence[:MAX_DOCUMENTS]
    mode = mode or group_default_mode(len(selected))
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    gen = Generation(card={}, mode=mode, n_supplied=len(evidence), n_analyzed=len(selected))

    def call(step: str, prompt: str, n_responses: int) -> dict:
        log(f"  [2 prompt] {step}: {n_responses} responses in, {len(prompt):,} chars")
        raw = runner.generate(prompt)
        cached = getattr(runner, "last_from_cache", False)
        log(f"  [3 model ] {runner.key}{' (from cache)' if cached else ''}: {len(raw):,} chars back")
        gen.calls.append({"step": step, "n_responses": n_responses, "prompt_chars": len(prompt),
                          "answer_chars": len(raw), "from_cache": cached,
                          "usage": getattr(runner, "last_usage", None)})
        try:
            parsed = pm.parse_llm_json(raw)
        except Exception as exc:
            raise AnswerParseError(step, raw, exc, gen.calls) from exc
        if not isinstance(parsed, dict):
            raise AnswerParseError(step, raw, TypeError(f"got {type(parsed).__name__}"), gen.calls)
        log(f"  [4 answer] {step}: JSON object with {len(parsed)} keys")
        return parsed

    if mode == "single":
        gen.card = call("single", pm.build_prompt(theme, selected), len(selected))
        return gen

    batches = [selected[i:i + BATCH_DOCUMENTS] for i in range(0, len(selected), BATCH_DOCUMENTS)]
    batch_summaries = []
    for i, batch in enumerate(batches, start=1):
        prompt = pm.build_batch_summary_prompt(
            theme, batch, batch_number=i, total_batches=len(batches))
        parsed = call(f"map {i}/{len(batches)}", prompt, len(batch))
        parsed["batch_number"] = i
        parsed["source_document_count"] = len(batch)
        batch_summaries.append(parsed)

    reduce_prompt = pm.build_reduce_prompt(
        theme, batch_summaries, source_document_count=len(selected))
    gen.card = call("reduce", reduce_prompt, len(selected))
    return gen
