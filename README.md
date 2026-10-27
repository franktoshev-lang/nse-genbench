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
`scripts/ui.py` does the same from a web page, and adds scoring and comparison.

## Quick start

```bash
python -m venv .venv            # then activate it
pip install -r requirements.txt
python -m pytest -q

python scripts/ui.py            # then open http://127.0.0.1:8770
```

## The UI

Local web app, standard library only, reachable from this computer only
(127.0.0.1). Five tabs:

| Tab | What you do there |
|---|---|
| Run | Pick a data slice, a model and a prompt, set repeats and mode, and start a run. The log shows each stage live. Runs go one at a time. |
| Results & scoring | Every run, filterable. Open one to read the card, check each claim against the answers it cites (supported / not supported), and rate usefulness, trustworthiness and completeness 1-5, each on its own. "Score blind" picks a random unscored run and hides the model, the prompt and the automatic scores until you save. |
| Matrix | Every prompt x model combination, with the mean of any automatic score or your ratings. Click a cell to see its runs. |
| Prompts | See exactly what each variant sends, with the changes against P0 highlighted. Add a variant by declaring text changes (with a live preview and checks), or remove one. |
| Settings | Endpoint URL and API key (saved in `.env`), model ids and optional prices per model, and the tokens and euros used against the budget. |

Runs land in `out/results.jsonl` and your scores in `out/scores.jsonl`.
Saving a score again keeps the old one as history. Each run keeps a copy of
its prompt variant's edits, so removing a variant never makes old runs
unreadable.

## Command line

```bash
python scripts/run.py --responses PATH [--truth PATH] --model mock:good \
                      [--variant P0] [--repeats N] [--mode single|mapreduce] \
                      [--theme NAME] [--temperature T] [--max-tokens N]
```

`--model mock:good`, `mock:bad` or `mock:leaky` runs the whole pipeline with
no model and no network. A key from `config/models.yml` (`gemma-4-12b`,
`gemma-4-31b`, `claude-opus-5-5`, `gpt-6-astra`) uses that model's settings.
Any other value is a model id for the endpoint in `.env`.

`--repeats N` runs the same condition N times as separate requests and prints
the spread. Every run, with its scores, card and token counts, is appended to
`out/results.jsonl`.

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
{"theme": "Examination & Assessment",
 "planted": {"Late feedback": 20, "Resits": 6},
 "minority_theme": "Resits",
 "absent_theme": "Alignment between exams and the content taught",
 "labels": {"E0001": "Late feedback", "E0002": null},
 "off_theme_ids": ["E0007"]}
```

`labels` and `off_theme_ids` are optional. Without a truth file, the five
metrics that need it show `-` and everything else still runs.

### Planted test slices

The test slices are generated outside this repo, by `generate.py` in the
`nse-genbench-data` folder next to it, into `data/` (ignored by git). One
slice is what the group pipeline hands the model for one theme, so the
planted structure is **subthemes inside that theme**:

| Slice | Answers | Planted |
|---|---|---|
| EA-60 | 60, single-pass | Examination & Assessment: late feedback 20, unclear criteria 12, exam scheduling 10, resits 6 (the minority), 6 noise, 6 off-theme |
| EA-240 | 240, map-reduce at exactly the cap | the same subthemes; resits 7 (2.9%), 30 noise, 33 off-theme |

One subtheme, alignment between exams and the content taught, is never
planted: any mention of it is invented. Off-theme answers come from other
themes and are sent as "ambiguous"; the prompt says to ignore them, so citing
one is a scope error. Before writing anything, the generator checks every
answer against the keyword probes, so no answer can contain the absent
subtheme's words or another subtheme's.

## Stage 2 - Prompt

`genbench/prompts.py` is a verbatim copy of
`src/pipeline/04_generation/prompts.py` and the theme definitions from
`src/config/themes.py`, at commit `6a48c37`. It is the **baseline control**,
so it keeps the group's known problems on purpose: "Gemma 4" hardcoded three
times, `summary` as the first schema key, the English-output rule, and no
caps or evidence-ID rule in the map prompt.

Variants are a separate layer (`config/variants.yml`, `genbench/variants.py`).
Each variant is the baseline code with declared text edits applied, and
changes exactly one dimension. `tests/test_variants.py` fails if a variant's
prompts differ from P0's in anything not declared.

| Variant | Changes |
|---|---|
| `P0` | nothing: the group's prompts, verbatim |
| `P0-neutral` | removes the three "Gemma 4" labels, so no model is told it is Gemma. Used for every comparison across models |

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
MODEL_NAME=       # only for --model values that are not in config/models.yml
```

With `MODEL_BASE_URL` empty, any non-mock run stops with a clear error.

The four models under test are in `config/models.yml`: Gemma 4 12B (the
baseline), Gemma 4 31B, Claude Opus 5.5 and GPT-6 Astra. Per model it holds
the id the endpoint expects, whether the model takes a temperature, and the
field name of its output limit. GPT-6 Astra gets no temperature (it cannot
turn reasoning off, and rejects one while reasoning) and `max_completion_tokens`.
Reasoning stays at each provider's default for now; reasoning tokens are
recorded per run.

Settings are always sent explicitly, never inherited from the group's model
registry:

| Setting | Default | Why |
|---|---|---|
| `max_tokens` | 8192 (`--max-tokens`) | The group's own code default. Its registry uses 12768-18768 with thinking on. |
| `temperature` | 0.1 (`--temperature`) | The group's code default. Its registry overrides it to 1.0 on every model; that is a condition to test, not to inherit. Not sent to a model configured without one. |
| `seed` | 42 (`GENBENCH_SEED`) | Same input, same output where the provider honours it. |
| tools / web search | never sent | The payload is built from an allowlist. |

Requests are one user message with no system message, as in the group
pipeline. The client retries timeouts, connection errors, 429 and 5xx with
backoff, and fails at once on other errors. It uses only the standard library.

Every raw response is cached under `out/cache/`, keyed by model, prompt hash
and an input hash (endpoint, output limit and its field name, temperature,
seed, repeat index), so scoring can be re-run without calling the model
again, and a repeated run costs nothing. The prompt text is not stored; the
model's answers are, so treat `out/` as data. Each run reports the tokens of
the calls that actually reached the endpoint.

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
| Off-theme answers cited | yes (`off_theme_ids`) |
| Privacy leakage (8-gram overlap) | no |
| Stability (Jaccard over repeats) | no |
| Output language | no |
| Verbosity | no |

The three mock cards (good, bad, leaky) have known correct scores, and the
tests assert them, with and without ground truth. That is what shows the
scorer works before any model runs.

Coverage, minority and hallucination use the keyword probes in
`config/probes.yml`, for themes and for the subthemes of the planted slices.
Keyword matching is fragile and the probes overlap across themes; it is used
because it is transparent and bilingual. A theme with no probes is never
counted as surfaced.

Human scores (from the UI) are kept apart from the automatic ones: three
ratings judged separately, never blended into one "quality" number, plus a
yes/no claim check per subtheme, which is the human anchor for a later
automatic faithfulness metric.

## Rules

- No data in git. `git ls-files data/` must print nothing.
- Dependencies: PyYAML and pytest only.
- Everything runs and is tested with no GPU and no network.
- Same input, same output.

## Layout

```
config/probes.yml      keyword probes used by the scorer
config/models.yml      the four models under test and their request settings
config/variants.yml    prompt variants and their declared edits
genbench/data.py       stage 1
genbench/prompts.py    stage 2, verbatim group prompts
genbench/variants.py   stage 2, variants layered on the verbatim prompts
genbench/pipeline.py   stage 2, single-pass / map-reduce
genbench/runners.py    stage 3
genbench/schema.py     stage 4, card schema
genbench/metrics.py    stage 4
genbench/report.py     stage 4, results table
genbench/runs.py       one run through all four stages (used by run.py and the UI)
genbench/store.py      results and human scores, matrix and budget
genbench/config_io.py  careful writers for .env and config/models.yml
genbench/ui/           the web UI: server.py and static/ (HTML, CSS, JS)
scripts/run.py         one run from the command line
scripts/ui.py          start the UI
tests/                 tests; tests/golden/ holds the group's rendered prompts
```
