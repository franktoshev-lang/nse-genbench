#!/usr/bin/env python3
"""Run the benchmark once, stage by stage: data -> prompt -> model -> answer -> score.

  python scripts/run.py --responses PATH [--truth PATH] --model mock:good
                        [--mode single|mapreduce] [--theme NAME]
                        [--temperature T] [--max-tokens N]

--model mock:good | mock:bad | mock:leaky needs no model at all. Any other
value is the model name to request from the endpoint configured in .env
(MODEL_BASE_URL, MODEL_API_KEY); it overrides MODEL_NAME.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from genbench import prompts
from genbench.data import DataError, load_responses, load_truth
from genbench.metrics import load_probes, score
from genbench.pipeline import (MAX_DOCUMENTS, MODES, AnswerParseError, generate_card,
                               group_default_mode)
from genbench.report import table
from genbench.runners import (MOCK_CARDS, ChatCompletionsRunner, MockRunner, ModelConfig,
                              ModelNotConfigured, ModelRequestError, ResponseCache, read_dotenv)

GROUP_SOURCE = "r-huijts/NSE_Applied_GenAI@6a48c37"
COLS = ["model", "slice", "valid_json", "coverage", "minority_preserved",
        "hallucinated_absent", "evidence_validity", "evidence_support",
        "max_leakage", "lang", "words"]


class UsageError(ValueError):
    pass


def stage(number: int, name: str) -> None:
    title = f"STAGE {number}  {name}"
    print(f"\n{title}\n{'-' * len(title)}")


def build_runner(args, settings: dict):
    if args.model.startswith("mock:"):
        variant = args.model.split(":", 1)[1]
        if variant not in MOCK_CARDS:
            raise UsageError(f"unknown mock {args.model!r}; use one of "
                             + ", ".join(f"mock:{v}" for v in MOCK_CARDS))
        print(f"  runner:  {args.model} (fixed card, no model, no network)")
        return MockRunner(variant)

    config = ModelConfig.from_env(ROOT / ".env", model=args.model,
                                  max_tokens=args.max_tokens, temperature=args.temperature)
    cache_dir = ROOT / settings.get("GENBENCH_OUT_DIR", "out") / "cache"
    print(f"  runner:  {config.model} at {config.base_url}")
    print(f"  request: max_tokens={config.max_tokens}  temperature={config.temperature}  "
          f"seed={config.seed}  tools/web search: never sent")
    print(f"  cache:   {cache_dir}")
    return ChatCompletionsRunner(config, cache=ResponseCache(cache_dir))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--responses", required=True, help="responses file (JSONL)")
    parser.add_argument("--truth", help="optional ground-truth JSON")
    parser.add_argument("--model", required=True, help="mock:good|mock:bad|mock:leaky, or a model name")
    parser.add_argument("--mode", choices=MODES, help="default: what the group pipeline would pick")
    parser.add_argument("--theme", help="theme to summarise (default: 'theme' in the truth file)")
    parser.add_argument("--temperature", type=float, help="override the default 0.1")
    parser.add_argument("--max-tokens", type=int, help="override the default 8192")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    settings = {**read_dotenv(ROOT / ".env"), **os.environ}
    try:
        stage(1, "DATA")
        responses = load_responses(args.responses)
        truth = load_truth(args.truth)
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
        print(f"  prompts: group pipeline, {GROUP_SOURCE}, unchanged")
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
        runner = build_runner(args, settings)
        print("  calls:   each step builds a prompt [2], calls the model [3], parses the answer [4]")
        try:
            card = generate_card(runner, theme, responses, mode=mode).card
        except AnswerParseError as exc:
            print(f"  [4 answer] {exc}; scoring an empty card")
            card = {}

        stage(4, "ANSWER + SCORE")
        slice_id = (truth or {}).get("id") or Path(args.responses).stem
        sc = score(card, responses, truth, load_probes(ROOT / "config" / "probes.yml"), slice_id=slice_id)
        row = {"model": runner.key, **sc.flat()}
        print(table([row], COLS))
        return 0
    except (FileNotFoundError, DataError, UsageError, ModelNotConfigured, ModelRequestError, ValueError) as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
