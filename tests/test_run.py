"""scripts/run.py walks the four stages in order, end to end, with no model."""
import importlib.util
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


def stage_order(out):
    return [int(n) for n in re.findall(r"^STAGE (\d)", out, flags=re.M)]


def test_full_run_with_truth(run_main, fixture_paths, capsys):
    responses, truth = fixture_paths
    assert run_main(["--responses", str(responses), "--truth", str(truth), "--model", "mock:good"]) == 0
    out = capsys.readouterr().out
    assert stage_order(out) == [1, 2, 3, 4]
    assert "[2 prompt] single" in out and "[3 model ] mock:good" in out and "[4 answer] single" in out
    row = out.strip().splitlines()[-1].split()
    assert row[:4] == ["mock:good", "fixture", "yes", "1.000"]

def test_full_run_without_truth_shows_dashes(run_main, fixture_paths, capsys):
    responses, _ = fixture_paths
    assert run_main(["--responses", str(responses), "--model", "mock:bad", "--theme", "Teachers"]) == 0
    out = capsys.readouterr().out
    assert stage_order(out) == [1, 2, 3, 4]
    assert "truth:     none" in out
    row = out.strip().splitlines()[-1].split()
    assert row[0] == "mock:bad" and row[3:6] == ["-", "-", "-"]

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

def test_real_model_without_endpoint_is_a_clear_error(run_main, fixture_paths, capsys, monkeypatch):
    monkeypatch.setenv("MODEL_BASE_URL", "")      # a local .env must not make this test call out
    responses, truth = fixture_paths
    assert run_main(["--responses", str(responses), "--truth", str(truth), "--model", "some-model"]) == 2
    assert "MODEL_BASE_URL is empty" in capsys.readouterr().err
