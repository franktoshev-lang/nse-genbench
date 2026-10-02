# nse-genbench

Benchmark harness for the **generation stage** of the NSE Open-Response
Analysis Tool: does a larger or different model produce theme insights that
are more *grounded*, *complete* and *stable* for quality officers — or only
more *fluent*?

Personal research project (MA-AAI), role: **GenAI Engineer**.
Companion to the group project at `r-huijts/NSE_Applied_GenAI`.

---

## The idea in one paragraph

Generate a synthetic Dutch corpus **from a known theme distribution**, so the
correct answer is known before any model runs. Then score each generated
dashboard card on coverage, minority preservation, hallucination, citation
validity, stability, privacy leakage and output language — and measure
**fluency separately**, so the two can be compared. Finally, cross model
against prompt, because the literature says the best prompt may differ per
model.

## Why the measurement layer comes first

`MockRunner` returns fixed cards whose correct scores are known. That means
**the entire metric harness is validated before any GPU exists** — 23 tests
assert that a known-good card scores 1.0 coverage, a known-bad card loses the
minority theme and invents an evidence ID, and a leaky card is caught at
> 0.9 n-gram overlap. Swapping in a real model is then the only remaining
unknown.

---

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python3 scripts/make_corpus.py      # writes data/synthetic/
python3 scripts/smoke.py            # end-to-end, no model needed
python3 -m pytest -q                # 23 tests
```

Expected smoke output: three mock models scored side by side, with
`mock:good` at coverage 1.000 and `mock:bad` at 0.250.

## Layout

```
config/
  themes.yml     7-theme inventory, Dutch templates, bilingual probes
  slices.yml     slice recipes — the planted ground truth
  models.yml     the slate, with VRAM/GGUF status per entry
genbench/
  corpus.py      deterministic synthetic corpus generator
  schema.py      the dashboard-card schema, mirrored from the pipeline
  metrics.py     all automatic metrics
  runners.py     MockRunner (works now) + LlamaCppRunner (stub)
  report.py      plain-text results table
scripts/
  make_corpus.py
  smoke.py
tests/           23 tests validating the metrics against known cards
data/synthetic/  generated; the .jsonl is gitignored, the recipe is not
```

## How a slice encodes its own ground truth

`config/slices.yml` → slice `S01`, 60 responses:

| Ingredient | Count | Makes measurable |
|---|---|---|
| Dominant theme — Examination & Assessment | 22 | baseline coverage |
| Mid themes — Teachers, Content & Organisation | 12, 10 | ordinary coverage |
| **Minority — Support / Mentoring** | **6** | **erasure** |
| **Absent — Professional Practice** | **0** | **hallucination** |
| Noise ("Prima", "Goed") | 6 | low-information passthrough |
| English responses | 4 | mixed-language handling |

`S02` is the same shape at 300 responses, so the minority theme sits at 2.7%
— which is the map-reduce survival test.

## Metrics

| Metric | Needs a model? | Needs a human? |
|---|---|---|
| Structural / JSON validity | yes | no |
| Theme coverage | yes | no |
| Minority preservation | yes | no |
| Hallucination (absent theme) | yes | no |
| Evidence-ID validity | yes | no |
| Evidence support (weak proxy) | yes | no |
| Privacy leakage (8-gram overlap) | yes | no |
| Stability (Jaccard over repeats) | yes | no |
| Output language | yes | no |
| Verbosity | yes | no |
| Faithfulness (claim-level) | yes | judge + validation |
| Usefulness / trust / completeness | yes | **quality officers (BWS)** |

### Known limitation, stated deliberately

Coverage, minority and hallucination use **keyword probes**. Keyword matching
is fragile. It is used here because it is transparent and tunable, and the
probes are **bilingual** — unlike `insight_metrics.subtheme_mention_rows()`
in the NSE repo, which filters Dutch text with a mostly-English stopword list
and feeds the result to dashboard percentages. See `nse-prompt-limitations.md`.

## Not committed, ever

`.gitignore` excludes `data/real/`, `data/staged/`, `out/`, all `*.jsonl`
except one example, `*.csv`, `*.xlsx`, vector stores, `.env` and `*.gguf`.
Recipes are version-controlled; data is not.

## Blocked work

`LlamaCppRunner.generate()` raises `NotImplementedError` on purpose. Before
any real run:

1. **Fix D-1** — the prompt hardcodes `Gemma 4` three times, which confounds
   every cross-model comparison.
2. **Confirm GPU** — does 26B-A4B (~16.9 GB) fit? Does Mistral Small 24B?
3. **Raise `max_tokens`** — the repo pins 2048; verbose models truncate and
   score 0 for a harness bug.
4. **Choose temperature deliberately** — the repo overrides its own 0.1
   default to 1.0 on all five models.
5. **Disable web search / tools** on every model, and record that you did.
