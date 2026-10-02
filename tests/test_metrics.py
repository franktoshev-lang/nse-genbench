"""The metrics are validated against cards whose correct scores are known.

This is what lets the measurement apparatus be trusted before any model runs.
"""
import json
from genbench.metrics import score, stability, leakage
from genbench.runners import MOCK_CARDS, MockRunner

def _score(variant, s01, cfg):
    themes, _ = cfg
    card = json.loads(MockRunner(variant).generate(""))
    return score(card, s01, themes)

# ── the known-good card ────────────────────────────────────────────────────
def test_good_card_full_coverage(s01, cfg):
    sc = _score("good", s01, cfg)
    assert sc.structure["valid"] is True
    assert sc.coverage["coverage"] == 1.0
    assert sc.coverage["missed"] == []

def test_good_card_preserves_minority(s01, cfg):
    assert _score("good", s01, cfg).minority["preserved"] is True

def test_good_card_does_not_hallucinate(s01, cfg):
    assert _score("good", s01, cfg).absent["hallucinated"] is False

def test_good_card_cites_only_real_ids(s01, cfg):
    ev = _score("good", s01, cfg).evidence
    assert ev["validity"] == 1.0
    assert ev["invented"] == []

def test_good_card_is_dutch(s01, cfg):
    assert _score("good", s01, cfg).language["dominant"] == "nl"

# ── the known-bad card ─────────────────────────────────────────────────────
def test_bad_card_loses_the_minority_theme(s01, cfg):
    assert _score("bad", s01, cfg).minority["preserved"] is False

def test_bad_card_hallucinates_the_absent_theme(s01, cfg):
    a = _score("bad", s01, cfg).absent
    assert a["hallucinated"] is True
    assert a["hit_probes"]

def test_bad_card_invents_an_evidence_id(s01, cfg):
    ev = _score("bad", s01, cfg).evidence
    assert "E9999" in ev["invented"]
    assert ev["validity"] == 0.5

def test_bad_card_has_low_coverage(s01, cfg):
    assert _score("bad", s01, cfg).coverage["coverage"] < 0.5

def test_bad_card_is_english(s01, cfg):
    assert _score("bad", s01, cfg).language["dominant"] == "en"

# ── the leaky card ─────────────────────────────────────────────────────────
def test_leaky_card_is_caught(s01, cfg):
    lk = _score("leaky", s01, cfg).leakage
    assert lk["max_overlap"] > 0.9
    assert lk["worst_id"] is not None

def test_good_card_does_not_leak(s01, cfg):
    assert _score("good", s01, cfg).leakage["max_overlap"] < 0.5

def test_suggestions_are_exempt_from_leakage(s01, cfg):
    """student_suggestions are meant to be verbatim, so they must not count."""
    themes, _ = cfg
    verbatim = s01["responses"][0]["document"]
    card = {"summary": "Korte samenvatting zonder overlap.",
            "positive_comments": [], "critical_comments": [],
            "student_suggestions": [verbatim],
            "subthemes": ["x"], "subtheme_manifest": []}
    assert leakage(card, s01)["max_overlap"] < 0.5

# ── stability ──────────────────────────────────────────────────────────────
def test_stability_identical_is_one():
    assert stability([MOCK_CARDS["good"]] * 3)["mean_jaccard"] == 1.0

def test_stability_disjoint_is_zero():
    assert stability([MOCK_CARDS["good"], MOCK_CARDS["leaky"]])["mean_jaccard"] == 0.0

def test_stability_needs_two_runs():
    assert stability([MOCK_CARDS["good"]])["mean_jaccard"] is None

# ── structural validation ──────────────────────────────────────────────────
def test_malformed_card_is_flagged(s01, cfg):
    themes, _ = cfg
    sc = score({"summary": 1, "oops": True}, s01, themes)
    assert sc.structure["valid"] is False
    assert any(p.startswith("missing:") for p in sc.structure["problems"])
