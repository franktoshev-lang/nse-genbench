"""The UI's API, end to end: a real server on a free local port, with every
file it writes (config copies, .env, results) in a temp folder."""
import json
import shutil
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from conftest import FIXTURE_RESPONSES, FIXTURE_TRUTH, write_json, write_jsonl
from genbench.runners import GOOD_CARD
from genbench.ui.server import App, serve

ROOT = Path(__file__).resolve().parents[1]


class FakeTransport:
    """Answers every chat-completions call with the good card, plus usage."""

    def __init__(self):
        self.payloads = []

    def __call__(self, url, body, headers, timeout):
        self.payloads.append(json.loads(body))
        answer = {"choices": [{"message": {"content": json.dumps(GOOD_CARD)}, "finish_reason": "stop"}],
                  "usage": {"prompt_tokens": 1000, "completion_tokens": 300}}
        return 200, json.dumps(answer), {}


@pytest.fixture
def ui(tmp_path):
    root = tmp_path / "root"
    (root / "config").mkdir(parents=True)
    for name in ("probes.yml", "models.yml", "variants.yml"):
        shutil.copy(ROOT / "config" / name, root / "config" / name)
    shutil.copy(ROOT / ".env.example", root / ".env.example")
    data = root / "data"
    data.mkdir()
    write_jsonl(data / "fixture.jsonl", FIXTURE_RESPONSES)
    write_json(data / "fixture.truth.json", FIXTURE_TRUTH)
    transport = FakeTransport()
    app = App(root, settings={"GENBENCH_OUT_DIR": str(root / "out"), "MODEL_BASE_URL": ""},
              transport=transport, models_path=root / "config" / "models.yml",
              variants_path=root / "config" / "variants.yml")
    server = serve(0, app=app)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def call(method, path, body=None):
        data_ = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(base + path, data=data_, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as res:
                return res.status, json.loads(res.read())
        except urllib.error.HTTPError as err:
            return err.code, json.loads(err.read())

    call.base, call.root, call.transport = base, root, transport
    yield call
    server.shutdown()
    server.server_close()


def run_and_wait(ui, **body):
    status, started = ui("POST", "/api/run", {"slice": "fixture", "variant": "P0", "repeats": 1, **body})
    assert status == 200, started
    for _ in range(200):
        _, job = ui("GET", f"/api/jobs/{started['job_id']}")
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_page_and_state(ui):
    with urllib.request.urlopen(ui.base + "/") as res:
        assert "NSE GenBench" in res.read().decode()
    status, state = ui("GET", "/api/state")
    assert status == 200
    assert [s["id"] for s in state["slices"]] == ["fixture"]
    assert state["slices"][0]["theme"] == "Examination & Assessment"
    assert [m["key"] for m in state["models"]] == ["gemma-4-12b", "gemma-4-31b", "claude-opus-5-5", "gpt-6-astra"]
    assert not any(m["ready"] for m in state["models"])
    assert {v["id"] for v in state["variants"]} >= {"P0", "P0-neutral"}

def test_run_score_and_matrix(ui):
    job = run_and_wait(ui, model="mock:good", repeats=2)
    assert job["status"] == "done" and len(job["run_ids"]) == 2
    assert any("STAGE 4" in line for line in job["log"])
    _, results = ui("GET", "/api/results")
    assert len(results["rows"]) == 2 and not results["rows"][0]["scored"]

    run_id = job["run_ids"][0]
    _, detail = ui("GET", f"/api/results/{run_id}")
    assert detail["record"]["scores"]["coverage"] == 1.0
    assert detail["cited"]["E0002"]["exists"] and "Projectmanagement" in detail["cited"]["E0002"]["text"]

    status, err = ui("POST", "/api/scores", {"run_id": run_id, "ratings": {"usefulness": 9}})
    assert status == 400 and "usefulness must be 1-5" in err["error"]
    status, _ = ui("POST", "/api/scores", {"run_id": run_id, "ratings": {"usefulness": 4, "trustworthiness": 3},
                                           "claims": {"0": True, "1": False}, "note": "fine", "scorer": "F"})
    assert status == 200
    _, results = ui("GET", "/api/results?scored=yes")
    assert [r["run_id"] for r in results["rows"]] == [run_id]

    _, m = ui("GET", "/api/matrix")
    cell = m["cells"]["P0|mock:good"]
    assert cell["n"] == 2 and cell["coverage"] == 1.0 and cell["usefulness"] == 4.0 and cell["claims_supported"] == 0.5

    _, pick = ui("GET", "/api/blind")
    assert pick["run_id"] is None                         # mock runs are left out of blind scoring
    _, pick = ui("GET", "/api/blind?include_mocks=yes")
    assert pick["run_id"] == job["run_ids"][1] and pick["remaining"] == 1

def test_real_model_run_through_the_endpoint(ui):
    status, _ = ui("POST", "/api/settings", {"base_url": "https://uni.example/v1", "api_key": "secret"})
    assert status == 200
    assert "MODEL_API_KEY=secret" in (ui.root / ".env").read_text(encoding="utf-8")
    _, state = ui("GET", "/api/state")
    assert state["endpoint"] == {"base_url": "https://uni.example/v1", "api_key_set": True}
    ready = {m["key"]: m["ready"] for m in state["models"]}
    assert ready == {"gemma-4-12b": False, "gemma-4-31b": False, "claude-opus-5-5": True, "gpt-6-astra": True}

    job = run_and_wait(ui, model="gpt-6-astra", variant="P0-neutral")
    assert job["status"] == "done", job
    payload = ui.transport.payloads[-1]
    assert payload["model"] == "gpt-6-astra" and "temperature" not in payload
    assert payload["max_completion_tokens"] == 8192
    assert payload["messages"][0]["content"].startswith("Task: act as")
    _, state = ui("GET", "/api/state")
    assert state["spend"]["per_model"]["gpt-6-astra"]["prompt_tokens"] == 1000

def test_models_are_configured_from_the_ui(ui):
    ui("POST", "/api/settings", {"base_url": "https://uni.example/v1"})
    status, _ = ui("POST", "/api/settings", {"models": {"gemma-4-12b": {
        "model_id": "google/gemma-4-12b-it", "price_in_per_million": "0.1", "price_out_per_million": ""}}})
    assert status == 200
    _, state = ui("GET", "/api/state")
    gemma = next(m for m in state["models"] if m["key"] == "gemma-4-12b")
    assert gemma["ready"] and gemma["model_id"] == "google/gemma-4-12b-it"
    assert gemma["price_in_per_million"] == 0.1 and gemma["price_out_per_million"] is None
    assert "# no standard id" in (ui.root / "config" / "models.yml").read_text(encoding="utf-8")
    status, err = ui("POST", "/api/settings", {"base_url": "ftp://nope"})
    assert status == 400 and "http" in err["error"]

def test_prompt_variants_can_be_previewed_added_and_removed(ui):
    edits = [{"old": "Respond EXACTLY in this JSON format:", "new": "Answer in this JSON format:"}]
    _, data = ui("POST", "/api/variants/preview", {"edits": edits, "kind": "map"})
    assert data["problems"] == [] and any(p["op"] == "ins" for p in data["diff"])
    status, _ = ui("POST", "/api/variants", {"id": "P1-test", "dimension": "instruction_wording",
                                             "description": "test", "edits": edits})
    assert status == 200
    _, prompt = ui("GET", "/api/variants/P1-test/prompt?kind=reduce&theme=Teachers")
    assert prompt["changed"]
    job = run_and_wait(ui, model="mock:good", variant="P1-test")
    assert job["status"] == "done"
    status, _ = ui("DELETE", "/api/variants/P1-test")
    assert status == 200
    _, state = ui("GET", "/api/state")
    assert "P1-test" not in {v["id"] for v in state["variants"]}
    _, results = ui("GET", "/api/results")
    run_id = results["rows"][0]["run_id"]
    _, detail = ui("GET", f"/api/results/{run_id}")
    assert detail["record"]["variant_edits"][0][1] == "Answer in this JSON format:"   # kept with the run
    status, err = ui("DELETE", "/api/variants/P0")
    assert status == 400 and "cannot be removed" in err["error"]

def test_bad_requests_get_clear_errors(ui):
    assert ui("POST", "/api/run", {"slice": "nope", "model": "mock:good"})[1]["error"] == "unknown slice 'nope'"
    assert "repeats" in ui("POST", "/api/run", {"slice": "fixture", "model": "mock:good", "repeats": 99})[1]["error"]
    job = run_and_wait(ui, model="gemma-4-12b")
    assert job["status"] == "error" and "model_id is empty" in job["error"]
    assert ui("GET", "/api/nothing")[0] == 404
