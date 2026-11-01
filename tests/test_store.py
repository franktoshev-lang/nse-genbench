"""Runs and human scores are stored, validated and aggregated correctly."""
import pytest

from genbench.store import (ScoreError, append_results, human_summary, load_results, load_scores,
                            load_score_history, matrix, save_score, spend, summary_rows)


def record(run_id, model="m1", variant="P0", slice_="S", coverage=0.5, tokens=None, request=True,
           minority=True, hallucinated=False, off_theme=0):
    return {"run_id": run_id, "created_utc": "2026-10-27T10:00:00+00:00", "slice": slice_,
            "model_key": model, "variant": variant, "sample": 0, "mode": "single",
            "request": {"temperature": 0.1} if request else None,
            "tokens": tokens or {"prompt_tokens": 1000, "completion_tokens": 200, "reasoning_tokens": 50},
            "parse_error": None,
            "scores": {"valid_json": True, "coverage": coverage, "minority_preserved": minority,
                       "hallucinated_absent": hallucinated, "off_theme_cited": off_theme,
                       "evidence_validity": 1.0, "evidence_support": 1.0, "words": 80}}


def test_results_round_trip(tmp_path):
    append_results(tmp_path, [record("a"), record("b")])
    append_results(tmp_path, [record("c")])
    assert [r["run_id"] for r in load_results(tmp_path)] == ["a", "b", "c"]
    assert load_results(tmp_path / "empty") == []

def test_latest_score_counts_and_history_is_kept(tmp_path):
    save_score(tmp_path, "a", checks={"T3": False}, known_run_ids={"a"})
    save_score(tmp_path, "a", checks={"T3": True, "T1:0": True, "T1:1": False, "U4": None},
               note=" ok ", scorer="Frank", known_run_ids={"a"})
    latest = load_scores(tmp_path)["a"]
    assert latest["checks"] == {"T3": True, "T1:0": True, "T1:1": False}    # empty answers are dropped
    assert latest["note"] == "ok" and latest["rubric"]
    assert len(load_score_history(tmp_path)) == 2

def test_older_scores_with_ratings_and_claims_still_load(tmp_path):
    save_score(tmp_path, "a", ratings={"usefulness": 4}, claims={"0": True, "1": False}, known_run_ids={"a"})
    s = human_summary(load_scores(tmp_path)["a"])
    assert s["trustworthy"] == 0.5 and s["claims_supported"] == 0.5        # claims count as check T1

@pytest.mark.parametrize("kwargs, message", [
    ({"ratings": {"usefulness": 6}}, "usefulness must be 1-5"),
    ({"ratings": {"usefulness": True}}, "usefulness must be 1-5"),
    ({"ratings": {"fluency": 3}}, "unknown rating 'fluency'"),
    ({"claims": {"x": True}}, "claims must map"),
    ({"claims": {"0": "yes"}}, "claims must map"),
    ({"checks": {"T9": True}}, "checklist item"),
    ({"checks": {"T2:pos:x": True}}, "checklist item"),
    ({"checks": {"T3": "yes"}}, "checklist item"),
    ({}, "nothing to save"),
    ({"checks": {"T3": None}}, "nothing to save"),
])
def test_bad_scores_are_rejected(tmp_path, kwargs, message):
    with pytest.raises(ScoreError, match=message):
        save_score(tmp_path, "a", known_run_ids={"a"}, **kwargs)

def test_score_for_an_unknown_run_is_rejected(tmp_path):
    with pytest.raises(ScoreError, match="unknown run"):
        save_score(tmp_path, "zz", checks={"T3": True}, known_run_ids={"a"})

def test_human_summary():
    assert human_summary(None)["scored"] is False
    s = human_summary({"checks": {"T1:0": True, "T1:1": False, "T1:2": True, "T3": True,
                                  "U1:pos:0": True, "U1:crit:0": False, "U3": True}})
    # trustworthy = mean(T1 2/3, T3 1) ; useful = mean(U1 1/2, U3 1) ; complete not answered
    assert s["trustworthy"] == 0.8333 and s["useful"] == 0.75 and s["complete"] is None
    assert s["claims_supported"] == 0.6667 and s["checks_answered"] == 7

def test_summary_rows_are_newest_first_with_human_scores(tmp_path):
    results = [record("a"), record("b")]
    rows = summary_rows(results, {"a": {"checks": {"U3": True}}})
    assert [r["run_id"] for r in rows] == ["b", "a"]
    assert rows[1]["scored"] and rows[1]["useful"] == 1.0 and not rows[0]["scored"]

def test_matrix_averages_every_combination():
    results = [record("a", coverage=1.0), record("b", coverage=0.5), record("c", model="m2", coverage=0.25),
               record("d", variant="P0-neutral", coverage=0.75, slice_="T")]
    scores = {"a": {"checks": {"U3": True, "U2": False}}}
    m = matrix(results, scores)
    assert m["variants"] == ["P0", "P0-neutral"] and m["models"] == ["m1", "m2"]
    cell = m["cells"]["P0|m1"]
    assert cell["n"] == 2 and cell["coverage"] == 0.75 and cell["coverage_sd"] == 0.3536
    assert cell["useful"] == 0.5 and cell["n_scored"] == 1 and cell["minority_preserved"] == 1.0
    assert "P0-neutral|m2" not in m["cells"]
    assert matrix(results, scores, "T")["variants"] == ["P0-neutral"]

def test_spend_counts_tokens_and_euros_where_priced():
    results = [record("a"), record("b"), record("c", model="m2"), record("d", model="mock:good", request=False)]
    config = {"budget_eur": 15, "models": {"m1": {"price_in_per_million": 5.0, "price_out_per_million": 25.0},
                                           "m2": {}}}
    s = spend(results, config)
    assert s["per_model"]["m1"] == {"runs": 2, "prompt_tokens": 2000, "completion_tokens": 400,
                                    "reasoning_tokens": 100, "eur": 0.02}
    assert s["per_model"]["m2"]["eur"] is None and not s["all_priced"]
    assert "mock:good" not in s["per_model"] and s["budget_eur"] == 15


def test_ranking_puts_trustworthy_first_then_complete_then_useful():
    results = [record("a", model="m1"), record("b", model="m2"), record("c", model="m3"),
               record("d", model="m4"), record("e", model="m5")]
    scores = {
        "a": {"checks": {"T3": True, "C2": False, "U3": True}},     # T 1.0, C 0.0
        "b": {"checks": {"T3": True, "C2": True, "U3": False}},     # T 1.0, C 1.0, U 0.0
        "c": {"checks": {"T3": False, "C2": True, "U3": True}},     # T 0.0
        "d": {"checks": {"T3": True, "C2": True, "U3": True}},      # T 1.0, C 1.0, U 1.0
    }
    rank = matrix(results, scores)["ranking"]
    assert [r["model"] for r in rank] == ["m4", "m2", "m1", "m3"]      # m5 has no score: not ranked
    assert [r["rank"] for r in rank] == [1, 2, 3, 4] and rank[0]["n_scored"] == 1
