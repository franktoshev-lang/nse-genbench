"""Stage 1: the loaders read supplied files, fail clearly, and never generate."""
import pytest

from conftest import FIXTURE_RESPONSES, FIXTURE_TRUTH, write_json, write_jsonl
from genbench.data import DataError, Response, load_responses, load_truth


def test_loads_responses_in_file_order(responses):
    assert [r.evidence_id for r in responses] == [r["evidence_id"] for r in FIXTURE_RESPONSES]
    assert responses[0] == Response("E0001", FIXTURE_RESPONSES[0]["document"], "definite")
    assert responses[4].evidence_type == "ambiguous"

def test_only_the_three_prompt_fields_are_kept(tmp_path):
    path = write_jsonl(tmp_path / "r.jsonl", [{**FIXTURE_RESPONSES[0], "planted_theme": "Teachers"}])
    assert load_responses(path)[0].as_dict() == FIXTURE_RESPONSES[0]

def test_missing_responses_file_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found.*Nothing is generated"):
        load_responses(tmp_path / "nope.jsonl")

def test_unsupported_format_names_the_supported_ones(tmp_path):
    path = tmp_path / "r.csv"
    path.write_text("evidence_id,document\n", encoding="utf-8")
    with pytest.raises(DataError, match=r"supported: \.jsonl"):
        load_responses(path)

def test_bad_json_line_is_reported_with_its_line_number(tmp_path):
    path = tmp_path / "r.jsonl"
    path.write_text('{"evidence_id": "E1", "document": "x", "evidence_type": "definite"}\n{oops\n',
                    encoding="utf-8")
    with pytest.raises(DataError, match=r"r\.jsonl:2:"):
        load_responses(path)

def test_missing_field_is_rejected(tmp_path):
    path = write_jsonl(tmp_path / "r.jsonl", [{"evidence_id": "E1", "evidence_type": "definite"}])
    with pytest.raises(DataError, match="missing document"):
        load_responses(path)

def test_unknown_evidence_type_is_rejected_not_dropped(tmp_path):
    path = write_jsonl(tmp_path / "r.jsonl", [{**FIXTURE_RESPONSES[0], "evidence_type": "maybe"}])
    with pytest.raises(DataError, match="evidence_type 'maybe'"):
        load_responses(path)

def test_duplicate_ids_are_rejected(tmp_path):
    path = write_jsonl(tmp_path / "r.jsonl", [FIXTURE_RESPONSES[0], FIXTURE_RESPONSES[0]])
    with pytest.raises(DataError, match="duplicate evidence_id 'E0001'"):
        load_responses(path)

def test_empty_file_is_rejected(tmp_path):
    path = tmp_path / "r.jsonl"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(DataError, match="no responses"):
        load_responses(path)

# ── ground truth ───────────────────────────────────────────────────────────
def test_no_truth_path_means_no_truth():
    assert load_truth(None) is None

def test_truth_is_loaded(truth):
    assert truth == FIXTURE_TRUTH

def test_missing_truth_file_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="Ground-truth file not found"):
        load_truth(tmp_path / "nope.json")

@pytest.mark.parametrize("change, message", [
    ({"planted": {}}, "'planted' must be"),
    ({"minority_theme": "Special Circumstances"}, "minority_theme 'Special Circumstances' is not in"),
    ({"absent_theme": "Teachers"}, "absent_theme 'Teachers' is also in"),
    ({"labels": ["E0001"]}, "'labels' must be"),
    ({"off_theme_ids": "E0001"}, "'off_theme_ids' must be"),
])
def test_inconsistent_truth_is_rejected(tmp_path, change, message):
    path = write_json(tmp_path / "t.json", {**FIXTURE_TRUTH, **change})
    with pytest.raises(DataError, match=message):
        load_truth(path)
