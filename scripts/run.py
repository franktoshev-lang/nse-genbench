#!/usr/bin/env python3
"""Run the benchmark stage by stage: data -> prompt -> model -> answer -> score.

  python scripts/run.py --responses PATH [--truth PATH] --model MODEL
                        [--variant P0] [--repeats N] [--mode single|mapreduce]
                        [--theme NAME] [--temperature T] [--max-tokens N]

--model is mock:good | mock:bad | mock:leaky (no model at all), a key from
config/models.yml (gemma-4-12b, gemma-4-31b, claude-opus-5-5, gpt-6-astra),
or any other model id for the endpoint in .env (MODEL_BASE_URL, MODEL_API_KEY).

--variant picks a prompt variant from config/variants.yml (default P0, the
group's prompt verbatim). --repeats runs the same condition N times; each
repeat is a separate request, not a cached copy of the first.

Every run is scored and appended to out/results.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from genbench import prompts
from genbench.data import DataError, load_responses, load_truth
from genbench.metrics import load_probes, score, stability
from genbench.pipeline import (MAX_DOCUMENTS, MODES, AnswerParseError, generate_card,
                               group_default_mode)
from genbench.report import table
from genbench.runners import (MOCK_CARDS, ChatCompletionsRunner, MockRunner, ModelConfig,
                              ModelNotConfigured, ModelRequestError, ResponseCache,
                              load_model_settings, read_dotenv)
from genbench.variants import BASELINE_ID, load_variants, prompts_for

GROUP_SOURCE = "r-huijts/NSE_Applied_GenAI@6a48c37"
COLS = ["model", "sample", "valid_json", "coverage", "minority_preserved",
        "hallucinated_absent", "evidence_validity", "evidence_support",
        "max_leakage", "lang", "words"]


class UsageError(ValueError):
    pass


def stage(number: int, name: str) -> None:
    title = f"STAGE {number}  {name}"
    print(f"\n{title}\n{'-' * len(title)}")


def out_dir(settings: dict) -> Path:
    path = Path(settings.get("GENBENCH_OUT_DIR") or "out")
    return path if path.is_absolute() else ROOT / path


def runner_factory(args, settings: dict):
    """Print the model setup once; return (make_runner(sample), request settings)."""
    if args.model.startswith("mock:"):
        variant = args.model.split(":", 1)[1]
        if variant not in MOCK_CARDS:
            raise UsageError(f"unknown mock {args.model!r}; use one of "
                             + ", ".join(f"mock:{v}" for v in MOCK_CARDS))
        print(f"  runner:  {args.model} (fixed card, no model, no network)")
        return (lambda sample: MockRunner(variant)), None

    spec = load_model_settings().get(args.model)
    overrides = {"max_tokens": args.max_tokens, "temperature": args.temperature}
    if spec is None:
        overrides["model"] = args.model
        label = args.model
    else:
        model_id = str(spec.get("model_id") or "").strip()
        if not model_id:
            raise UsageError(f"{args.model}: model_id is empty in config/models.yml; "
                             f"fill in the id the endpoint expects")
        if args.temperature is not None and not spec.get("temperature", True):
            raise UsageError(f"{args.model} does not accept a temperature (config/models.yml)")
        overrides.update(model=model_id, send_temperature=spec.get("temperature", True),
                         token_limit_param=spec.get("token_limit_param", "max_tokens"))
        label = f"{spec.get('name', args.model)} ({model_id})"

    config = ModelConfig.from_env(ROOT / ".env", **overrides)
    cache = ResponseCache(out_dir(settings) / "cache")
    temperature = config.temperature if config.send_temperature else None
    print(f"  runner:  {label} at {config.base_url}")
    print(f"  request: {config.token_limit_param}={config.max_tokens}  "
          f"temperature={'not sent' if temperature is None else temperature}  seed={config.seed}  "
          f"reasoning=provider default  tools/web search: never sent")
    print(f"  cache:   {cache.root}")
    request = {"token_limit_param": config.token_limit_param, "max_tokens": config.max_tokens,
               "temperature": temperature, "seed": config.seed, "reasoning": "provider default"}
    return (lambda sample: ChatCompletionsRunner(config, cache=cache, sample=sample)), request


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--responses", required=True, help="responses file (JSONL)")
    parser.add_argument("--truth", help="optional ground-truth JSON")
    parser.add_argument("--model", required=True,
                        help="mock:good|mock:bad|mock:leaky, a key from config/models.yml, or a model id")
    parser.add_argument("--variant", default=BASELINE_ID, help="prompt variant from config/variants.yml")
    parser.add_argument("--repeats", type=int, default=1, help="run the same condition N times")
    parser.add_argument("--mode", choices=MODES, help="default: what the group pipeline would pick")
    parser.add_argument("--theme", help="theme to summarise (default: 'theme' in the truth file)")
    parser.add_argument("--temperature", type=float, help="override the default 0.1")
    parser.add_argument("--max-tokens", type=int, help="override the default 8192")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    settings = {**read_dotenv(ROOT / ".env"), **os.environ}
    try:
        if args.repeats < 1:
            raise UsageError("--repeats must be at least 1")

        stage(1, "DATA")
        responses = load_responses(args.responses)
        truth = load_truth(args.truth)
        slice_id = (truth or {}).get("id") or Path(args.responses).stem
        n_ambiguous = sum(r.evidence_type == "ambiguous" for r in responses)
        print(f"  responses: {args.responses}")
        print(f"             {len(responses)} records ({n_ambiguous} ambiguous)")
        if truth is None:
            print("  truth:     none -> coverage, minority, hallucination and support show '-'")
        else:
            print(f"  truth:     {args.truth}")

        stage(2, "PROMPT")
        theme = args.theme or (truth or {}).get("theme")
        if not theme:
            raise UsageError("no theme given: pass --theme NAME, one of: "
                             + ", ".join(t for t in prompts.THEMES_LIST if t != prompts.LOW_INFORMATION_THEME))
        if theme == prompts.LOW_INFORMATION_THEME:
            raise UsageError(f"the group pipeline excludes {theme!r} from generation")
        prompt_module = prompts_for(args.variant)
        if args.variant == BASELINE_ID:
            print(f"  prompts: group pipeline, {GROUP_SOURCE}, variant {BASELINE_ID} (verbatim)")
        else:
            dimension = load_variants()[args.variant]["dimension"]
            print(f"  prompts: group pipeline, {GROUP_SOURCE}, variant {args.variant} "
                  f"(changes: {dimension})")
        known = theme in prompts.THEME_LLM_DEFINITIONS
        print(f"  theme:   {theme}" + ("" if known else "  (not a group theme: the scope line is just the name)"))
        n_analyzed = min(len(responses), MAX_DOCUMENTS)
        if len(responses) > MAX_DOCUMENTS:
            print(f"  cap:     first {MAX_DOCUMENTS} of {len(responses)} responses, as in the group "
                  f"pipeline; the other {len(responses) - MAX_DOCUMENTS} never reach the model")
        default_mode = group_default_mode(n_analyzed)
        mode = args.mode or default_mode
        note = ("what the group pipeline does" if mode == default_mode else
                f"forced; the group pipeline would use {default_mode} for {n_analyzed} responses")
        print(f"  mode:    {mode} ({note})")

        stage(3, "MODEL")
        make_runner, request = runner_factory(args, settings)
        print("  calls:   each step builds a prompt [2], calls the model [3], parses the answer [4]")
        runs = []
        for sample in range(args.repeats):
            if args.repeats > 1:
                print(f"  -- repeat {sample + 1} of {args.repeats}")
            runner = make_runner(sample)
            try:
                gen = generate_card(runner, theme, responses, mode=mode, prompt_module=prompt_module)
                runs.append({"sample": sample, "model": runner.key, "card": gen.card,
                             "calls": gen.calls, "parse_error": None})
            except AnswerParseError as exc:
                print(f"  [4 answer] {exc}; scoring an empty card")
                runs.append({"sample": sample, "model": runner.key, "card": {},
                             "calls": exc.calls, "parse_error": str(exc)})

        stage(4, "ANSWER + SCORE")
        probes = load_probes(ROOT / "config" / "probes.yml")
        rows, records = [], []
        created = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for run in runs:
            sc = score(run["card"], responses, truth, probes, slice_id=slice_id)
            rows.append({"model": run["model"], "sample": run["sample"], **sc.flat()})
            records.append({
                "created_utc": created, "slice": slice_id, "responses_file": str(args.responses),
                "truth_file": args.truth, "theme": theme, "variant": args.variant, "mode": mode,
                "model_key": args.model, "model": run["model"], "sample": run["sample"],
                "request": request, "tokens": billed_tokens(run["calls"]), "calls": run["calls"],
                "parse_error": run["parse_error"], "scores": sc.flat(), "scorecard": sc.to_dict(),
                "card": run["card"],
            })
        print(f"  slice:   {slice_id}   variant: {args.variant}")
        print(table(rows, COLS))

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
            print(line)

        tokens = [rec["tokens"] for rec in records]
        n_calls = sum(t["calls"] for t in tokens)
        cached = sum(t["from_cache"] for t in tokens)
        if request is None:
            print(f"  tokens:  none (mock), {n_calls} calls")
        else:
            print(f"  tokens:  {sum(t['prompt_tokens'] for t in tokens):,} in, "
                  f"{sum(t['completion_tokens'] for t in tokens):,} out "
                  f"({sum(t['reasoning_tokens'] for t in tokens):,} of them reasoning) over "
                  f"{n_calls - cached} new calls; {cached} came from the cache and cost nothing")

        results = out_dir(settings) / "results.jsonl"
        results.parent.mkdir(parents=True, exist_ok=True)
        with results.open("a", encoding="utf-8", newline="\n") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  saved:   {len(records)} run(s) appended to {results}")
        return 0
    except (FileNotFoundError, DataError, UsageError, ModelNotConfigured, ModelRequestError, ValueError) as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
