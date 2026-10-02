# nse-genbench

Benchmark harness for the **generation stage** of the NSE Open-Response
Analysis Tool: does a larger or different model produce theme insights that
are more grounded, complete and stable for quality officers, or only more
fluent?

Personal research project (MA-AAI), role: GenAI Engineer. Companion to the
group project at `r-huijts/NSE_Applied_GenAI`.

## The pipeline

```
data -> prompt -> model -> answer -> score
```

| Stage | Module | What it does |
|---|---|---|
| 1 Data | `genbench/data.py` | Loads responses (and optional ground truth) from paths given at run time. Ships no data, generates none. |
| 2 Prompt | `genbench/prompts.py`, `genbench/pipeline.py` | The group pipeline's prompts, copied verbatim, plus its single-pass / map-reduce logic. |
| 3 Model | `genbench/runners.py` | One client for any OpenAI-compatible endpoint, plus `MockRunner` for running with no model. |
| 4 Answer + score | `genbench/schema.py`, `metrics.py`, `report.py` | Parses the answer with the group's own JSON repair, scores it, prints a table. |

`scripts/run.py` walks the four stages in order and prints which one it is in.

## Quick start

```bash
python -m venv .venv            # then activate it
pip install -r requirements.txt
python -m pytest -q

python scripts/run.py --responses PATH [--truth PATH] --model mock:good \
                      [--mode single|mapreduce] [--theme NAME]
```

`--model mock:good`, `mock:bad` or `mock:leaky` runs the whole pipeline with
no model and no network. Any other value is a model name for the endpoint in
`.env`.

## Stage 1 - Data

Nothing in this repo is data, and nothing under `data/` is ever tracked.
Real responses will come from the group project; their export format is not
known yet, so each format is parsed by one function in `genbench/data.py`
(`_PARSERS`). JSONL is the first:

```json
{"evidence_id": "E0001", "document": "...", "evidence_type": "definite"}
```

These are the three fields the group's prompts read. Other fields are
ignored. `evidence_type` must be one the group prompt uses (`definite`,
`ambiguous`, `free_text_retrieval`, `subtheme_candidate`): any other value
would be dropped from the prompt without a warning, so the loader rejects it.
A missing file is an error, never a reason to generate something.

Ground truth is optional (`--truth`, JSON):

```json
{"theme": "Teachers",
 "planted": {"Teachers": 12, "Support / Mentoring": 6},
 "minority_theme": "Support / Mentoring",
 "absent_theme": "Professional Practice",
 "labels": {"E0001": "Teachers", "E0002": null}}
```

`labels` is optional and only feeds evidence support. Without a truth file,
the four metrics that need it show `-` and everything else still runs.

## Stage 2 - Prompt

`genbench/prompts.py` is a verbatim copy of
`src/pipeline/04_generation/prompts.py` and the theme definitions from
`src/config/themes.py`, at commit `6a48c37`. It is the **baseline control**,
so it keeps the group's known problems on purpose: "Gemma 4" hardcoded three
times, `summary` as the first schema key, the English-output rule, and no
caps or evidence-ID rule in the map prompt. Variants will be a separate
layer; none exist yet.

`tests/test_prompts.py` checks that every prompt, plus `parse_llm_json`,
renders byte-identical to the group repo: against a snapshot of the group's
own output (always), and against a local clone when `NSE_REPO_PATH` is set.

`genbench/pipeline.py` mirrors the group's `service.py`: at most 240
responses reach the model (the rest are dropped), 60 per batch, a single
prompt when that is one batch, otherwise one map prompt per batch and a
reduce prompt over the parsed batch answers. `--mode` can force either path;
the run output says when that differs from what the group pipeline would do.

## Stage 3 - Model

Configured in `.env` (copy `.env.example`):

```
MODEL_BASE_URL=   # e.g. http://127.0.0.1:8080/v1 for llama-server
MODEL_API_KEY=
MODEL_NAME=
```

With `MODEL_BASE_URL` empty, any non-mock run stops with a clear error.

Settings are always sent explicitly, never inherited from the group's model
registry:

| Setting | Default | Why |
|---|---|---|
| `max_tokens` | 8192 (`--max-tokens`) | The group's own code default. Its registry uses 12768-18768 with thinking on. |
| `temperature` | 0.1 (`--temperature`) | The group's code default. Its registry overrides it to 1.0 on every model; that is a condition to test, not to inherit. |
| `seed` | 42 (`GENBENCH_SEED`) | Same input, same output where the provider honours it. |
| tools / web search | never sent | The payload is built from an allowlist. |

Requests are one user message with no system message, as in the group
pipeline. The client retries timeouts, connection errors, 429 and 5xx with
backoff, and fails at once on other errors. It uses only the standard library.

Every raw response is cached under `out/cache/`, keyed by model, prompt hash
and an input hash (endpoint, max_tokens, temperature, seed, repeat index), so
scoring can be re-run without calling the model again. The prompt text is
not stored; the model's answers are, so treat `out/` as data.

## Stage 4 - Answer + score

The answer is parsed with the group's `parse_llm_json`, so repairs happen
exactly as in production. An answer it cannot parse is scored as an empty,
invalid card.

| Metric | Needs ground truth? |
|---|---|
| Structural / JSON validity | no |
| Theme coverage | yes |
| Minority preservation | yes |
| Hallucination (absent theme) | yes |
| Evidence-ID validity | no |
| Evidence support (weak proxy) | yes (`labels`) |
| Privacy leakage (8-gram overlap) | no |
| Stability (Jaccard over repeats) | no |
| Output language | no |
| Verbosity | no |

The three mock cards (good, bad, leaky) have known correct scores, and the
tests assert them, with and without ground truth. That is what shows the
scorer works before any model runs.

Coverage, minority and hallucination use the keyword probes in
`config/probes.yml`. Keyword matching is fragile and the probes overlap
across themes; it is used because it is transparent and bilingual. A theme
with no probes is never counted as surfaced.

## Rules

- No data in git. `git ls-files data/` must print nothing.
- Dependencies: PyYAML and pytest only.
- Everything runs and is tested with no GPU and no network.
- Same input, same output.

## Layout

```
config/probes.yml      keyword probes used by the scorer
config/models.yml      candidate model slate (notes; not read by code)
genbench/data.py       stage 1
genbench/prompts.py    stage 2, verbatim group prompts
genbench/pipeline.py   stage 2, single-pass / map-reduce
genbench/runners.py    stage 3
genbench/schema.py     stage 4, card schema
genbench/metrics.py    stage 4
genbench/report.py     stage 4, results table
scripts/run.py         one run, all four stages
tests/                 tests; tests/golden/ holds the group's rendered prompts
```
