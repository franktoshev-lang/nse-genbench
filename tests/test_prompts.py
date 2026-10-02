"""Stage 2 is the baseline control: genbench's prompts must render
byte-identical to the group pipeline's for the same inputs.

Two checks over the same CASES:
  snapshot  tests/golden/group_prompts_6a48c37.json holds what the group repo
            itself rendered at commit 6a48c37. Runs everywhere, no clone needed.
  live      renders CASES with the group repo, loaded by file path (its
            package directory starts with a digit). Runs when NSE_REPO_PATH
            points at a clone, skipped otherwise.

The snapshot is only ever written from the group repo, never from genbench:
  NSE_REPO_PATH=<clone> GENBENCH_WRITE_GOLDEN=1 pytest tests/test_prompts.py
"""
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

from genbench import prompts

GOLDEN = Path(__file__).parent / "golden" / "group_prompts_6a48c37.json"

# Inputs only. They cover every branch of _append_evidence: each evidence
# type it keeps, one it silently drops, a missing ID, and a bare string.
DOCS = [
    {"evidence_id": "E0001", "document": "De docenten reageren traag op mail.", "evidence_type": "definite"},
    {"evidence_id": "E0002", "document": "Feedback komt pas na zes weken.", "evidence_type": "ambiguous"},
    {"evidence_id": "E0003", "document": "Rooster wijzigt vaak.", "evidence_type": "free_text_retrieval"},
    {"evidence_id": "E0004", "document": "Mentor is moeilijk bereikbaar.", "evidence_type": "subtheme_candidate"},
    {"evidence_id": "E0005", "document": "Wordt weggelaten door de groep.", "evidence_type": "other"},
    {"document": "Antwoord zonder ID.", "evidence_type": "definite"},
    "Losse tekst zonder metadata.",
]
DEFINITE_ONLY = [d for d in DOCS[:1]]
BATCH_SUMMARIES = [
    {"summary": "Studenten zijn kritisch over feedback.", "positive_comments": [],
     "critical_comments": ["Late feedback"], "student_suggestions": ["Geef feedback binnen één week."],
     "subthemes": ["Feedback"], "subtheme_manifest": [
         {"name": "Feedback", "description": "Te laat", "evidence_ids": ["E0002"]}],
     "batch_number": 1, "source_document_count": 60},
    {"summary": "Gemengd beeld.", "positive_comments": ["Betrokken docenten"], "critical_comments": [],
     "student_suggestions": [], "subthemes": [], "subtheme_manifest": [],
     "batch_number": 2, "source_document_count": 37},
]
PARSE_INPUTS = {
    "clean": '{"summary": "ok", "subthemes": []}',
    "thinking_and_fence": '<think>plan first</think>\n```json\n{"summary": "x", "a": [1, 2,],}\n```',
    "truncated": '{"summary": "afgebroken", "positive_comments": ["een", "tw',
    "two_objects": '{"draft": true} final: {"summary": "laatste"}',
    "control_chars": '{"summary": "regel1\nregel2\tklaar"}',
    "garbage": "no json here",
}


def render_cases(prompts_mod, themes: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for theme in [*themes, "Onbekend thema"]:
        out[f"default_prompt/{theme}"] = prompts_mod.default_prompt(theme)
    for theme in ("Teachers", "Examination & Assessment"):
        out[f"build_prompt/{theme}"] = prompts_mod.build_prompt(theme, DOCS)
        out[f"build_prompt_definite_only/{theme}"] = prompts_mod.build_prompt(theme, DEFINITE_ONLY)
        out[f"build_batch_summary_prompt/{theme}"] = prompts_mod.build_batch_summary_prompt(
            theme, DOCS, batch_number=2, total_batches=4)
        out[f"build_reduce_prompt/{theme}"] = prompts_mod.build_reduce_prompt(
            theme, BATCH_SUMMARIES, source_document_count=240)
    for name, text in PARSE_INPUTS.items():
        try:
            result = json.dumps(prompts_mod.parse_llm_json(text), ensure_ascii=False, sort_keys=True)
        except Exception as exc:
            result = f"{type(exc).__name__}: {exc}"
        out[f"parse_llm_json/{name}"] = result
    return out


def load_group(repo: Path):
    """Load the group's prompts.py by path, and its themes module."""
    root = str(repo)
    sys.path.insert(0, root)
    try:
        themes = importlib.import_module("src.config.themes")
        spec = importlib.util.spec_from_file_location(
            "group_generation_prompts", repo / "src" / "pipeline" / "04_generation" / "prompts.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(root)
    return module, themes


@pytest.fixture(scope="module")
def group():
    repo = os.environ.get("NSE_REPO_PATH", "").strip()
    if not repo:
        pytest.skip("NSE_REPO_PATH not set; the snapshot test still checks against the group's output")
    path = Path(repo)
    if not (path / "src" / "pipeline" / "04_generation" / "prompts.py").is_file():
        pytest.fail(f"NSE_REPO_PATH={repo} does not look like a clone of r-huijts/NSE_Applied_GenAI")
    return load_group(path)


def test_matches_snapshot_of_group_output():
    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))
    actual = render_cases(prompts, prompts.THEMES_LIST)
    assert actual.keys() == expected.keys()
    for name in expected:
        assert actual[name] == expected[name], f"{name} differs from the group's rendering"


def test_matches_group_repo_live(group):
    group_prompts, group_themes = group
    assert prompts.THEMES_LIST == group_themes.THEMES_LIST
    assert prompts.LOW_INFORMATION_THEME == group_themes.LOW_INFORMATION_THEME
    assert prompts.THEME_LLM_DEFINITIONS == group_themes.THEME_LLM_DEFINITIONS
    for name in ("ENGLISH_OUTPUT_INSTRUCTION", "SUBTHEME_MANIFEST_INSTRUCTION",
                 "GEMMA_JSON_RESPONSE_CONTRACT", "DASHBOARD_JSON_SCHEMA"):
        assert getattr(prompts, name) == getattr(group_prompts, name), name

    expected = render_cases(group_prompts, group_themes.THEMES_LIST)
    if os.environ.get("GENBENCH_WRITE_GOLDEN") == "1":
        GOLDEN.parent.mkdir(exist_ok=True)
        GOLDEN.write_text(json.dumps(expected, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8", newline="\n")
    actual = render_cases(prompts, prompts.THEMES_LIST)
    assert actual.keys() == expected.keys()
    for name in expected:
        assert actual[name] == expected[name], f"{name} differs from the group repo"


def test_baseline_keeps_the_known_defects():
    """A guard against someone 'fixing' the control in place."""
    single = prompts.build_prompt("Teachers", DEFINITE_ONLY)
    batch = prompts.build_batch_summary_prompt("Teachers", DEFINITE_ONLY, batch_number=1, total_batches=2)
    assert single.count("Gemma 4") == 3
    assert prompts.DASHBOARD_JSON_SCHEMA.lstrip("{\n ").startswith('"summary"')
    assert "write generated analytical fields in English" in single
    assert "up to 3" in single and "up to 3" not in batch
    assert "Do not invent, rename, or reformat evidence IDs" in single
    assert "Do not invent, rename, or reformat evidence IDs" not in batch
