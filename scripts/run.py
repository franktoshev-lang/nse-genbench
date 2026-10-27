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

Every run is scored and appended to out/results.jsonl. The same runs can be
started, scored and compared in the UI: python scripts/ui.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from genbench.data import DataError
from genbench.pipeline import MODES
from genbench.runners import ModelNotConfigured, ModelRequestError
from genbench.runs import RunRequest, UsageError, execute
from genbench.variants import BASELINE_ID


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

    request = RunRequest(responses=args.responses, truth=args.truth, model=args.model,
                         variant=args.variant, repeats=args.repeats, mode=args.mode,
                         theme=args.theme, temperature=args.temperature, max_tokens=args.max_tokens)
    try:
        execute(request)
        return 0
    except (FileNotFoundError, DataError, UsageError, ModelNotConfigured, ModelRequestError, ValueError) as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
