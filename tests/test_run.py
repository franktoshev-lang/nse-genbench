"""scripts/run.py walks the four stages in order, end to end, with no model."""
import importlib.util
import json
import re
from pathlib import Path

import pytest

from conftest import FIXTURE_RESPONSES, write_jsonl

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def run_main():
    spec = importlib.util.spec_from_file_location("genbench_run", ROOT / "scripts" / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


@pytest.fixture(autouse=True)
def out_dir(tmp_path, monkeypatch):
    """Results and cache go to a temp folder, and a local .env can't reach a real endpoint."""
    out = tmp_path / "out"
    monkeypatch.setenv("GENBENCH_OUT_DIR", str(out))
    monkeypatch.setenv("MODEL_BASE_URL", "")
    return out


def stage_order(out):
    return [int(n) for n in re.findall(r"^STAGE (\d)", out, flags=re.M)]


def table_rows(out, model):
    return [line.split() for line in out.splitlines() if line.startswith(model + " ")]


def results(out_dir):
    path = out_dir / "results.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_full_run_with_truth(run_main, fixture_paths, capsys, out_dir):
    responses, truth = fixture_paths
    assert run_main(["--responses", str(responses), "--truth", str(truth), "--model", "mock:good"]) == 0
    out = capsys.readouterr().out
    assert stage_order(out) == [1, 2, 3, 4]
    assert "[2 prompt] single" in out and "[3 model ] mock:good" in out and "[4 answer] single" in out
    assert "variant P0 (verbatim)" in out
    assert table_rows(out, "mock:good") == [["mock:good", "0", "yes", "1.000", "yes", "no", "-",
                                             "1.000", "1.000", "0.000", "nl", "87"]]
    [record] = results(out_dir)
    assert (record["slice"], record["variant"], record["sample"]) == ("fixture", "P0", 0)
    assert record["scores"]["coverage"] == 1.0 and record["card"]["subthemes"]

def test_full_run_without_truth_shows_dashes(run_main, fixture_paths, capsys):
    responses, _ = fixture_paths
    assert run_main(["--responses", str(responses), "--model", "mock:bad", "--theme", "Teachers"]) == 0
    out = capsys.readouterr().out
    assert stage_order(out) == [1, 2, 3, 4]
    assert "truth:     none" in out
    [row] = table_rows(out, "mock:bad")
    assert row[3:6] == ["-", "-", "-"]

def test_repeats_are_separate_runs_and_all_saved(run_main, fixture_paths, capsys, out_dir):
    responses, truth = fixture_paths
    assert run_main(["--responses", str(responses), "--truth", str(truth), "--model", "mock:good",
                     "--repeats", "3"]) == 0
    out = capsys.readouterr().out
    assert [r[1] for r in table_rows(out, "mock:good")] == ["0", "1", "2"]
    assert "repeats: 3   valid JSON 3/3   coverage mean 1.000 sd 0.000   subtheme stability 1.000" in out
    assert [r["sample"] for r in results(out_dir)] == [0, 1, 2]

def test_variant_is_used_and_reported(run_main, fixture_paths, capsys, out_dir):
    responses, truth = fixture_paths
    assert run_main(["--responses", str(responses), "--truth", str(truth), "--model", "mock:good",
                     "--variant", "P0-neutral"]) == 0
    assert "variant P0-neutral (changes: model_name)" in capsys.readouterr().out
    assert results(out_dir)[0]["variant"] == "P0-neutral"

def test_unknown_variant_is_a_clear_error(run_main, fixture_paths, capsys):
    responses, truth = fixture_paths
    assert run_main(["--responses", str(responses), "--truth", str(truth), "--model", "mock:good",
                     "--variant", "P99"]) == 2
    assert "unknown variant 'P99'" in capsys.readouterr().err

def test_mapreduce_run_shows_every_step(run_main, tmp_path, capsys):
    records = [{**FIXTURE_RESPONSES[0], "evidence_id": f"E{i:04d}"} for i in range(1, 62)]
    path = write_jsonl(tmp_path / "r.jsonl", records)
    assert run_main(["--responses", str(path), "--model", "mock:good", "--theme", "Teachers"]) == 0
    out = capsys.readouterr().out
    assert "mode:    mapreduce (what the group pipeline does)" in out
    assert out.index("[2 prompt] map 1/2") < out.index("[2 prompt] map 2/2") < out.index("[2 prompt] reduce")

def test_forced_mode_is_reported(run_main, fixture_paths, capsys):
    responses, truth = fixture_paths
    run_main(["--responses", str(responses), "--truth", str(truth), "--model", "mock:good",
              "--mode", "mapreduce"])
    assert "forced; the group pipeline would use single for 6 responses" in capsys.readouterr().out

def test_missing_theme_is_a_clear_error(run_main, fixture_paths, capsys):
    responses, _ = fixture_paths
    assert run_main(["--responses", str(responses), "--model", "mock:good"]) == 2
    assert "no theme given" in capsys.readouterr().err

def test_missing_responses_file_is_a_clear_error(run_main, tmp_path, capsys):
    assert run_main(["--responses", str(tmp_path / "nope.jsonl"), "--model", "mock:good"]) == 2
    assert "Responses file not found" in capsys.readouterr().err

def test_repeats_must_be_positive(run_main, fixture_paths, capsys):
    responses, truth = fixture_paths
    assert run_main(["--responses", str(responses), "--truth", str(truth), "--model", "mock:good",
                     "--repeats", "0"]) == 2
    assert "--repeats must be at least 1" in capsys.readouterr().err

def test_real_model_without_endpoint_is_a_clear_error(run_main, fixture_paths, capsys):
    responses, truth = fixture_paths
    assert run_main(["--responses", str(responses), "--truth", str(truth), "--model", "some-model"]) == 2
    assert "MODEL_BASE_URL is empty" in capsys.readouterr().err

def test_model_key_without_an_id_is_a_clear_error(run_main, fixture_paths, capsys):
    responses, truth = fixture_paths
    assert run_main(["--responses", str(responses), "--truth", str(truth), "--model", "gemma-4-12b"]) == 2
    assert "gemma-4-12b: model_id is empty in config/models.yml" in capsys.readouterr().err

def test_temperature_is_refused_for_a_model_that_rejects_it(run_main, fixture_paths, capsys):
    responses, truth = fixture_paths
    assert run_main(["--responses", str(responses), "--truth", str(truth), "--model", "gpt-6-astra",
                     "--temperature", "1.0"]) == 2
    assert "gpt-6-astra does not accept a temperature" in capsys.readouterr().err
