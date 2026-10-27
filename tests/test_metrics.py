"""The metrics are validated against cards whose correct scores are known.

This is what lets the measurement apparatus be trusted before any model runs.
Every check is made twice where it matters: with ground truth, and without it,
where the four truth metrics must come back as None and the rest still run.
"""
import json

import pytest

from genbench.metrics import score, stability, leakage
from genbench.report import table
from genbench.runners import MOCK_CARDS, MockRunner

TRUTH_METRICS = ("coverage", "minority_preserved", "hallucinated_absent", "evidence_support")


def _card(variant):
    return json.loads(MockRunner(variant).generate(""))

def _score(variant, responses, truth, probes):
    return score(_card(variant), responses, truth, probes)

# ── the known-good card ────────────────────────────────────────────────────
def test_good_card_full_coverage(responses, truth, probes):
    sc = _score("good", responses, truth, probes)
    assert sc.structure["valid"] is True
    assert sc.coverage["coverage"] == 1.0
    assert sc.coverage["missed"] == []

def test_good_card_preserves_minority(responses, truth, probes):
    assert _score("good", responses, truth, probes).minority["preserved"] is True

def test_good_card_does_not_hallucinate(responses, truth, probes):
    assert _score("good", responses, truth, probes).absent["hallucinated"] is False

def test_good_card_cites_only_real_ids(responses, truth, probes):
    ev = _score("good", responses, truth, probes).evidence
    assert ev["validity"] == 1.0
    assert ev["invented"] == []

def test_good_card_is_dutch(responses, truth, probes):
    assert _score("good", responses, truth, probes).language["dominant"] == "nl"

# ── the known-bad card ─────────────────────────────────────────────────────
def test_bad_card_loses_the_minority_theme(responses, truth, probes):
    assert _score("bad", responses, truth, probes).minority["preserved"] is False

def test_bad_card_hallucinates_the_absent_theme(responses, truth, probes):
    a = _score("bad", responses, truth, probes).absent
    assert a["hallucinated"] is True
    assert a["hit_probes"]

def test_bad_card_invents_an_evidence_id(responses, truth, probes):
    ev = _score("bad", responses, truth, probes).evidence
    assert "E9999" in ev["invented"]
    assert ev["validity"] == 0.5

def test_bad_card_has_low_coverage(responses, truth, probes):
    assert _score("bad", responses, truth, probes).coverage["coverage"] == 0.25

def test_bad_card_is_english(responses, truth, probes):
    assert _score("bad", responses, truth, probes).language["dominant"] == "en"

# ── the leaky card ─────────────────────────────────────────────────────────
def test_leaky_card_is_caught(responses, truth, probes):
    lk = _score("leaky", responses, truth, probes).leakage
    assert lk["max_overlap"] == 1.0
    assert lk["worst_id"] is not None

def test_good_card_does_not_leak(responses, truth, probes):
    assert _score("good", responses, truth, probes).leakage["max_overlap"] < 0.5

def test_suggestions_are_exempt_from_leakage(responses):
    """student_suggestions are meant to be verbatim, so they must not count."""
    verbatim = responses[0].document
    card = {"summary": "Korte samenvatting zonder overlap.",
            "positive_comments": [], "critical_comments": [],
            "student_suggestions": [verbatim],
            "subthemes": ["x"], "subtheme_manifest": []}
    assert leakage(card, responses)["max_overlap"] < 0.5

# ── without ground truth ───────────────────────────────────────────────────
@pytest.mark.parametrize("variant", sorted(MOCK_CARDS))
def test_without_truth_the_truth_metrics_are_none(variant, responses, probes):
    row = _score(variant, responses, None, probes).flat()
    assert all(row[m] is None for m in TRUTH_METRICS)

def test_without_truth_the_other_metrics_still_run(responses, probes):
    good = _score("good", responses, None, probes)
    bad = _score("bad", responses, None, probes)
    leaky = _score("leaky", responses, None, probes)
    assert good.structure["valid"] is True
    assert good.evidence["validity"] == 1.0
    assert bad.evidence["validity"] == 0.5 and "E9999" in bad.evidence["invented"]
    assert leaky.leakage["max_overlap"] == 1.0
    assert (good.language["dominant"], bad.language["dominant"]) == ("nl", "en")

def test_without_truth_the_table_shows_a_dash(responses, probes):
    row = _score("good", responses, None, probes).flat()
    rendered = table([row], list(TRUTH_METRICS))
    assert rendered.splitlines()[-1].split() == ["-", "-", "-", "-"]

def test_truth_without_labels_only_drops_evidence_support(responses, truth, probes):
    unlabelled = {k: v for k, v in truth.items() if k != "labels"}
    row = _score("good", responses, unlabelled, probes).flat()
    assert row["evidence_support"] is None
    assert row["coverage"] == 1.0

def test_evidence_support_uses_the_labels(responses, truth, probes):
    assert _score("good", responses, truth, probes).support["support_rate"] == 1.0
    assert _score("bad", responses, truth, probes).support["support_rate"] == 0.0

# ── stability ──────────────────────────────────────────────────────────────
def test_stability_identical_is_one():
    assert stability([MOCK_CARDS["good"]] * 3)["mean_jaccard"] == 1.0

def test_stability_disjoint_is_zero():
    assert stability([MOCK_CARDS["good"], MOCK_CARDS["leaky"]])["mean_jaccard"] == 0.0

def test_stability_needs_two_runs():
    assert stability([MOCK_CARDS["good"]])["mean_jaccard"] is None

# ── structural validation ──────────────────────────────────────────────────
@pytest.mark.parametrize("with_truth", [True, False])
def test_malformed_card_is_flagged(with_truth, responses, truth, probes):
    sc = score({"summary": 1, "oops": True}, responses, truth if with_truth else None, probes)
    assert sc.structure["valid"] is False
    assert any(p.startswith("missing:") for p in sc.structure["problems"])

# ── off-theme citations ────────────────────────────────────────────────────
def test_off_theme_citations_are_counted(responses, truth, probes):
    with_off_theme = {**truth, "off_theme_ids": ["E0003", "E0005"]}
    assert _score("good", responses, with_off_theme, probes).off_theme == {"n_cited": 1, "ids": ["E0003"]}
    assert _score("good", responses, truth, probes).off_theme == {"n_cited": None, "ids": None}
