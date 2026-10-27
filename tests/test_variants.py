"""Prompt variants change exactly what config/variants.yml declares.

The check renders every prompt case with P0 and with the variant: applying
the declared edits to P0's output must give the variant's output exactly, and
every edit must show up in at least one rendered prompt (an edit that only
hits unused code would change nothing while claiming to).
"""
import pytest

from test_prompts import render_cases

from genbench import prompts
from genbench.variants import (BASELINE_ID, VariantError, edits_for, load_variants,
                               prompts_for)

VARIANT_IDS = [v for v in load_variants() if v != BASELINE_ID]


def test_baseline_is_the_verbatim_module():
    assert prompts_for(BASELINE_ID) is prompts
    assert edits_for(BASELINE_ID) == []


@pytest.mark.parametrize("variant_id", VARIANT_IDS)
def test_variant_differs_from_baseline_only_by_its_declared_edits(variant_id):
    edits = edits_for(variant_id)
    assert edits, "a non-baseline variant must declare what it changes"
    baseline = render_cases(prompts, prompts.THEMES_LIST)
    variant = render_cases(prompts_for(variant_id), prompts.THEMES_LIST)
    assert baseline.keys() == variant.keys()
    used = set()
    for name, text in baseline.items():
        expected = text
        for old, new, _ in edits:
            if old in expected:
                used.add(old)
            expected = expected.replace(old, new)
        assert variant[name] == expected, f"{variant_id}: undeclared difference in {name}"
    assert used == {old for old, _, _ in edits}, f"{variant_id}: an edit never reaches a prompt"
    assert baseline != variant


def test_neutral_variant_never_names_a_model():
    neutral = render_cases(prompts_for("P0-neutral"), prompts.THEMES_LIST)
    assert all("Gemma" not in text for text in neutral.values())
    assert any("Gemma 4" in text for text in render_cases(prompts, prompts.THEMES_LIST).values())


def test_unknown_variant_is_rejected():
    with pytest.raises(VariantError, match="unknown variant 'P99'"):
        prompts_for("P99")


@pytest.mark.parametrize("yaml_text, message", [
    ("variants:\n  X:\n    dimension: a\n", "baseline variant 'P0' is missing"),
    ("variants:\n  P0:\n    dimension: none\n    replace: [['a', 'b', 1]]\n", "cannot have edits"),
    ("variants:\n  P0:\n    dimension: none\n  X:\n    dimension: [a, b]\n", "must be one name"),
    ("variants:\n  P0:\n    dimension: none\n  X:\n    dimension: a\n    replace: [['a', 'b']]\n",
     r"\[old, new, count\]"),
])
def test_malformed_config_is_rejected(tmp_path, yaml_text, message):
    path = tmp_path / "variants.yml"
    path.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(VariantError, match=message):
        load_variants(path)


def test_edit_count_must_match_the_baseline_code(tmp_path):
    path = tmp_path / "variants.yml"
    path.write_text("variants:\n  P0:\n    dimension: none\n  X:\n    dimension: model_name\n"
                    "    replace: [['Gemma 4', 'Model', 1]]\n", encoding="utf-8")
    with pytest.raises(VariantError, match="occurs .* times in the baseline, declared 1"):
        prompts_for("X", path)


# ── adding and removing variants (the UI's Prompts tab) ───────────────────
import shutil
from pathlib import Path

from genbench.variants import check_edits, delete_variant, preview, save_variant


@pytest.fixture
def variants_file(tmp_path):
    path = tmp_path / "variants.yml"
    shutil.copy(Path(__file__).resolve().parents[1] / "config" / "variants.yml", path)
    return path


def test_preview_shows_the_change_against_p0():
    data = preview([("Respond EXACTLY in this JSON format:", "Answer in this JSON format:")])
    assert data["problems"] == []
    assert {"op": "ins", "text": "Answer"} in data["diff"]
    # 6: the single, map and reduce prompts, plus the three unused subtheme prompts
    assert data["edits"] == [["Respond EXACTLY in this JSON format:", "Answer in this JSON format:", 6]]

def test_preview_explains_what_is_wrong():
    assert "not in the baseline prompt" in preview([("no such text anywhere", "x")])["problems"][0]
    assert "same text" in preview([("Gemma 4 task:", "Gemma 4 task:")])["problems"][0]
    _, problems = check_edits([("def build_prompt(", "def build_prompt2(")])
    assert "not in any prompt the model sees" in problems[0]

def test_saved_variant_can_be_used_and_removed(variants_file):
    save_variant("P1-test", "schema_order", "test variant",
                 [("Respond EXACTLY in this JSON format:", "Answer in this JSON format:")], variants_file)
    assert "P1-test" in load_variants(variants_file)
    module = prompts_for("P1-test", variants_file)
    assert "Answer in this JSON format:" in module.build_prompt("Teachers", [])
    delete_variant("P1-test", variants_file)
    assert "P1-test" not in load_variants(variants_file)
    assert "P0-neutral" in load_variants(variants_file)

@pytest.mark.parametrize("variant_id, dimension, edits, message", [
    ("P0-neutral", "x", [("Gemma 4 task:", "Task:")], "already exists"),
    ("bad name!", "x", [("Gemma 4 task:", "Task:")], "name:"),
    ("P2", "", [("Gemma 4 task:", "Task:")], "dimension"),
    ("P2", "a, b", [("Gemma 4 task:", "Task:")], "dimension"),
    ("P2", "x", [], "at least one change"),
])
def test_bad_variants_are_not_saved(variants_file, variant_id, dimension, edits, message):
    before = variants_file.read_text(encoding="utf-8")
    with pytest.raises(VariantError, match=message):
        save_variant(variant_id, dimension, "", edits, variants_file)
    assert variants_file.read_text(encoding="utf-8") == before

def test_baseline_cannot_be_removed(variants_file):
    with pytest.raises(VariantError, match="cannot be removed"):
        delete_variant("P0", variants_file)
