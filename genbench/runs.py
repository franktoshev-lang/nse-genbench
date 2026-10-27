"""One benchmark run, stage by stage: data -> prompt -> model -> answer -> score.

The code behind scripts/run.py and the UI. execute() walks the four stages,
reports progress through `log`, appends every run to out/results.jsonl and
returns the records.
"""
from __future__ import annotations

import os
import statistics
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from . import prompts
from .data import load_responses, load_truth
from .metrics import load_probes, score, stability
from .pipeline import MAX_DOCUMENTS, AnswerParseError, generate_card, group_default_mode
from .report import table
from .runners import (MODELS_CONFIG, MOCK_CARDS, ChatCompletionsRunner, MockRunner, ModelConfig,
                      ResponseCache, load_model_settings, read_dotenv)
from .store import append_results
from .variants import BASELINE_ID, VARIANTS_CONFIG, edits_for, load_variants, prompts_for

ROOT = Path(__file__).resolve().parents[1]
GROUP_SOURCE = "r-huijts/NSE_Applied_GenAI@6a48c37"
COLS = ["model", "sample", "valid_json", "coverage", "minority_preserved",
        "hallucinated_absent", "off_theme_cited", "evidence_validity", "evidence_support",
        "max_leakage", "lang", "words"]


class UsageError(ValueError):
    pass


@dataclass
class RunRequest:
    responses: str
    model: str
    truth: str | None = None
    variant: str = BASELINE_ID
    repeats: int = 1
    mode: str | None = None
    theme: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None


def current_settings(root: Path = ROOT) -> dict:
    """.env values, overridden by the process environment."""
    return {**read_dotenv(root / ".env"), **os.environ}


def out_dir(settings: dict, root: Path = ROOT) -> Path:
    path = Path(settings.get("GENBENCH_OUT_DIR") or "out")
    return path if path.is_absolute() else root / path


def billed_tokens(calls: list[dict]) -> dict:
    """Tokens of the calls that actually went to the endpoint (not the cache)."""
    total = {"calls": len(calls), "from_cache": 0, "prompt_tokens": 0,
             "completion_tokens": 0, "reasoning_tokens": 0, "not_reported": 0}
    for call in calls:
        if call.get("from_cache"):
            total["from_cache"] += 1
            continue
        usage = call.get("usage")
        if not usage:
            total["not_reported"] += 1
            continue
        total["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        total["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        details = usage.get("completion_tokens_details") or {}
        total["reasoning_tokens"] += int(details.get("reasoning_tokens") or 0)
    return total


def _runner_factory(req: RunRequest, settings: dict, log, transport, root: Path, models_path):
    """Report the model setup once; return (make_runner(sample), request settings)."""
    if req.model.startswith("mock:"):
        variant = req.model.split(":", 1)[1]
        if variant not in MOCK_CARDS:
            raise UsageError(f"unknown mock {req.model!r}; use one of "
                             + ", ".join(f"mock:{v}" for v in MOCK_CARDS))
        log(f"  runner:  {req.model} (fixed card, no model, no network)")
        return (lambda sample: MockRunner(variant)), None

    spec = load_model_settings(models_path).get(req.model)
    overrides = {"max_tokens": req.max_tokens, "temperature": req.temperature}
    if spec is None:
        overrides["model"] = req.model
        label = req.model
    else:
        model_id = str(spec.get("model_id") or "").strip()
        if not model_id:
            raise UsageError(f"{req.model}: model_id is empty in config/models.yml; "
                             f"fill in the id the endpoint expects")
        if req.temperature is not None and not spec.get("temperature", True):
            raise UsageError(f"{req.model} does not accept a temperature (config/models.yml)")
        overrides.update(model=model_id, send_temperature=spec.get("temperature", True),
                         token_limit_param=spec.get("token_limit_param", "max_tokens"))
        label = f"{spec.get('name', req.model)} ({model_id})"

    config = ModelConfig.from_env(root / ".env", environ=settings, **overrides)
    cache = ResponseCache(out_dir(settings, root) / "cache")
    temperature = config.temperature if config.send_temperature else None
    log(f"  runner:  {label} at {config.base_url}")
    log(f"  request: {config.token_limit_param}={config.max_tokens}  "
        f"temperature={'not sent' if temperature is None else temperature}  seed={config.seed}  "
        f"reasoning=provider default  tools/web search: never sent")
    log(f"  cache:   {cache.root}")
    request = {"token_limit_param": config.token_limit_param, "max_tokens": config.max_tokens,
               "temperature": temperature, "seed": config.seed, "reasoning": "provider default"}
    return (lambda sample: ChatCompletionsRunner(config, cache=cache, sample=sample,
                                                 transport=transport)), request


def stage(log, number: int, name: str) -> None:
    title = f"STAGE {number}  {name}"
    log(f"\n{title}\n{'-' * len(title)}")


def execute(req: RunRequest, *, log: Callable[[str], None] = print, transport=None,
            settings: dict | None = None, root: Path = ROOT, models_path=MODELS_CONFIG,
            variants_path=VARIANTS_CONFIG) -> list[dict]:
    """Run one condition req.repeats times. Raises on bad input or a failed request."""
    settings = current_settings(root) if settings is None else settings
    if req.repeats < 1:
        raise UsageError("--repeats must be at least 1")

    stage(log, 1, "DATA")
    responses = load_responses(req.responses)
    truth = load_truth(req.truth)
    slice_id = (truth or {}).get("id") or Path(req.responses).stem
    n_ambiguous = sum(r.evidence_type == "ambiguous" for r in responses)
    log(f"  responses: {req.responses}")
    log(f"             {len(responses)} records ({n_ambiguous} ambiguous)")
    if truth is None:
        log("  truth:     none -> coverage, minority, hallucination and support show '-'")
    else:
        log(f"  truth:     {req.truth}")

    stage(log, 2, "PROMPT")
    theme = req.theme or (truth or {}).get("theme")
    if not theme:
        raise UsageError("no theme given: pass --theme NAME, one of: "
                         + ", ".join(t for t in prompts.THEMES_LIST if t != prompts.LOW_INFORMATION_THEME))
    if theme == prompts.LOW_INFORMATION_THEME:
        raise UsageError(f"the group pipeline excludes {theme!r} from generation")
    prompt_module = prompts_for(req.variant, variants_path)
    variant_spec = load_variants(variants_path)[req.variant]
    if req.variant == BASELINE_ID:
        log(f"  prompts: group pipeline, {GROUP_SOURCE}, variant {BASELINE_ID} (verbatim)")
    else:
        log(f"  prompts: group pipeline, {GROUP_SOURCE}, variant {req.variant} "
            f"(changes: {variant_spec['dimension']})")
    known = theme in prompts.THEME_LLM_DEFINITIONS
    log(f"  theme:   {theme}" + ("" if known else "  (not a group theme: the scope line is just the name)"))
    n_analyzed = min(len(responses), MAX_DOCUMENTS)
    if len(responses) > MAX_DOCUMENTS:
        log(f"  cap:     first {MAX_DOCUMENTS} of {len(responses)} responses, as in the group "
            f"pipeline; the other {len(responses) - MAX_DOCUMENTS} never reach the model")
    default_mode = group_default_mode(n_analyzed)
    mode = req.mode or default_mode
    note = ("what the group pipeline does" if mode == default_mode else
            f"forced; the group pipeline would use {default_mode} for {n_analyzed} responses")
    log(f"  mode:    {mode} ({note})")

    stage(log, 3, "MODEL")
    make_runner, request = _runner_factory(req, settings, log, transport, root, models_path)
    log("  calls:   each step builds a prompt [2], calls the model [3], parses the answer [4]")
    runs = []
    for sample in range(req.repeats):
        if req.repeats > 1:
            log(f"  -- repeat {sample + 1} of {req.repeats}")
        runner = make_runner(sample)
        try:
            gen = generate_card(runner, theme, responses, mode=mode, prompt_module=prompt_module, log=log)
            runs.append({"sample": sample, "model": runner.key, "card": gen.card,
                         "calls": gen.calls, "parse_error": None})
        except AnswerParseError as exc:
            log(f"  [4 answer] {exc}; scoring an empty card")
            runs.append({"sample": sample, "model": runner.key, "card": {},
                         "calls": exc.calls, "parse_error": str(exc)})

    stage(log, 4, "ANSWER + SCORE")
    probes = load_probes(root / "config" / "probes.yml")
    rows, records = [], []
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for run in runs:
        sc = score(run["card"], responses, truth, probes, slice_id=slice_id)
        rows.append({"model": run["model"], "sample": run["sample"], **sc.flat()})
        records.append({
            "run_id": uuid.uuid4().hex[:12], "created_utc": created, "slice": slice_id,
            "responses_file": str(req.responses), "truth_file": req.truth, "theme": theme,
            "variant": req.variant, "variant_dimension": variant_spec["dimension"],
            "variant_edits": [list(e) for e in edits_for(req.variant, variants_path)], "mode": mode,
            "model_key": req.model, "model": run["model"], "sample": run["sample"],
            "request": request, "tokens": billed_tokens(run["calls"]), "calls": run["calls"],
            "parse_error": run["parse_error"], "scores": sc.flat(), "scorecard": sc.to_dict(),
            "card": run["card"],
        })
    log(f"  slice:   {slice_id}   variant: {req.variant}")
    log(table(rows, COLS))

    if len(runs) > 1:
        valid = sum(bool(r["valid_json"]) for r in rows)
        line = f"\n  repeats: {len(rows)}   valid JSON {valid}/{len(rows)}"
        coverages = [r["coverage"] for r in rows if r["coverage"] is not None]
        if len(coverages) >= 2:
            line += (f"   coverage mean {statistics.mean(coverages):.3f}"
                     f" sd {statistics.stdev(coverages):.3f}")
        jaccard = stability([run["card"] for run in runs])["mean_jaccard"]
        if jaccard is not None:
            line += f"   subtheme stability {jaccard:.3f}"
        log(line)

    tokens = [rec["tokens"] for rec in records]
    n_calls = sum(t["calls"] for t in tokens)
    cached = sum(t["from_cache"] for t in tokens)
    if request is None:
        log(f"  tokens:  none (mock), {n_calls} calls")
    else:
        log(f"  tokens:  {sum(t['prompt_tokens'] for t in tokens):,} in, "
            f"{sum(t['completion_tokens'] for t in tokens):,} out "
            f"({sum(t['reasoning_tokens'] for t in tokens):,} of them reasoning) over "
            f"{n_calls - cached} new calls; {cached} came from the cache and cost nothing")

    path = append_results(out_dir(settings, root), records)
    log(f"  saved:   {len(records)} run(s) appended to {path}")
    return records
