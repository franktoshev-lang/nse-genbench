"""The Excel export: a valid workbook holding every run's model, variant,
card, cited answers with the answer key, the input, and the exact prompts."""
import json
import re
import urllib.error
import urllib.request
import zipfile
from io import BytesIO
from xml.etree import ElementTree

import pytest

from conftest import FIXTURE_RESPONSES, write_jsonl
from genbench.export import CELL_LIMIT, Workbook, build_workbook, export_filename, rebuild_steps
from genbench.runners import GOOD_CARD
from genbench.runs import ROOT, RunRequest, execute
from genbench.store import save_score

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


class FakeTransport:
    def __init__(self):
        self.calls = 0

    def __call__(self, url, body, headers, timeout):
        self.calls += 1
        answer = {"choices": [{"message": {"content": json.dumps(GOOD_CARD)}, "finish_reason": "stop"}],
                  "usage": {"prompt_tokens": 100, "completion_tokens": 50}}
        return 200, json.dumps(answer), {}


def sheets(body: bytes) -> dict[str, list[list[str]]]:
    """{sheet name: rows of cell texts}, parsing every part as XML on the way."""
    z = zipfile.ZipFile(BytesIO(body))
    for name in z.namelist():
        ElementTree.fromstring(z.read(name))           # every part is well-formed XML
    book = ElementTree.fromstring(z.read("xl/workbook.xml"))
    names = [s.get("name") for s in book.find("m:sheets", NS)]
    out = {}
    for i, name in enumerate(names, start=1):
        root = ElementTree.fromstring(z.read(f"xl/worksheets/sheet{i}.xml"))
        rows = []
        for row in root.iter(f"{{{NS['m']}}}row"):
            cells = {}
            for c in row:
                col = re.match(r"[A-Z]+", c.get("r")).group()
                t = c.find("m:is/m:t", NS)
                v = c.find("m:v", NS)
                cells[col] = t.text if t is not None else (v.text if v is not None else "")
            rows.append(cells)
        out[name] = rows
    return out


def column(rows, header):
    col = next(k for k, v in rows[0].items() if v == header)
    return [r.get(col) for r in rows[1:]]


@pytest.fixture
def settings(tmp_path):
    return {"GENBENCH_OUT_DIR": str(tmp_path / "out"), "MODEL_BASE_URL": "http://127.0.0.1:9/v1",
            "MODEL_API_KEY": ""}


def run(settings, responses, truth=None, model="mock:good", variant="P0-neutral", transport=None, **kw):
    return execute(RunRequest(responses=str(responses), truth=str(truth) if truth else None,
                              model=model, variant=variant, theme="Examination & Assessment", **kw),
                   log=lambda _: None, settings=settings, transport=transport)


def test_workbook_holds_model_variant_card_claims_input_and_prompts(fixture_paths, settings, tmp_path):
    responses, truth = fixture_paths
    records = run(settings, responses, truth)
    save_score(tmp_path / "out", records[0]["run_id"], ratings={"usefulness": 4}, claims={"0": True})
    from genbench.store import load_scores
    body = build_workbook(records, load_scores(tmp_path / "out"), root=ROOT, settings=settings,
                          out_dir=tmp_path / "out", filters={"model": "mock:good"})
    book = sheets(body)
    assert list(book) == ["Read me", "Runs", "Claims", "Input fixture", "Prompts"]

    runs = book["Runs"]
    assert column(runs, "Model") == ["mock:good"]
    assert column(runs, "Prompt variant") == ["P0-neutral"]
    assert column(runs, "Variant changes") == ["model_name"]
    assert column(runs, "Summary") == [GOOD_CARD["summary"]]
    assert column(runs, "Usefulness (1-5)") == ["4"]

    claims = book["Claims"]
    cited = column(claims, "Cited answer ID")
    assert cited == ["E0002", "E0003", None]          # the second subtheme cites nothing
    texts = {r["evidence_id"]: r["document"] for r in FIXTURE_RESPONSES}
    assert column(claims, "Student answer") == [texts["E0002"], texts["E0003"], "(no ID cited)"]
    assert "Examination & Assessment" in column(claims, "Planted as (answer key)")
    assert column(claims, "Supported? (yes/no)")[0] == "yes"

    assert column(book["Input fixture"], "Answer ID") == [r["evidence_id"] for r in FIXTURE_RESPONSES]

    prompt_sent = column(book["Prompts"], "Prompt sent")
    assert len(prompt_sent) == 1 and prompt_sent[0].startswith("Task: act as")
    assert "Gemma" not in prompt_sent[0]
    assert column(book["Prompts"], "Check") == [None]


def test_mapreduce_prompts_are_rebuilt_from_the_cache(tmp_path, settings):
    records_in = [{**FIXTURE_RESPONSES[0], "evidence_id": f"E{i:04d}"} for i in range(1, 62)]
    path = write_jsonl(tmp_path / "r.jsonl", records_in)
    transport = FakeTransport()
    record = run(settings, path, model="some-model", transport=transport, variant="P2-batch-20")[0]
    assert transport.calls == 5                                   # four maps of up to 20, then reduce
    steps, note = rebuild_steps(record, ROOT, settings, tmp_path / "out")
    assert note == ""
    assert [s["step"] for s in steps] == ["map 1/4", "map 2/4", "map 3/4", "map 4/4", "reduce"]
    assert all(s["chars_ok"] and s["answer"] for s in steps)
    assert transport.calls == 5                                   # rebuilding sent nothing

    for f in (tmp_path / "out" / "cache").rglob("*.json"):       # without the cache:
        f.unlink()
    steps, note = rebuild_steps(record, ROOT, settings, tmp_path / "out")
    assert [s["step"] for s in steps] == ["map 1/4", "map 2/4", "map 3/4", "map 4/4"]
    assert "reduce prompt" in note


def test_long_text_is_cut_to_what_a_cell_holds():
    wb = Workbook()
    wb.add_sheet("S", ["a"], [["x" * (CELL_LIMIT + 10)]], [10])
    text = sheets(wb.to_bytes())["S"][1]["A"]
    assert len(text) == CELL_LIMIT and text.endswith("can hold]")


def test_filename_names_the_filters():
    name = export_filename({"slice": "EA-60", "model": "gpt-5.6-sol", "variant": None})
    assert re.fullmatch(r"genbench-export_\d{4}-\d\d-\d\d_\d{4}_EA-60_gpt-5\.6-sol\.xlsx", name)


def test_export_route_downloads_the_filtered_runs(tmp_path):
    import shutil
    import threading
    from conftest import FIXTURE_TRUTH, write_json
    from genbench.ui.server import App, serve
    root = tmp_path / "root"
    (root / "config").mkdir(parents=True)
    for name in ("probes.yml", "models.yml", "variants.yml"):
        shutil.copy(ROOT / "config" / name, root / "config" / name)
    (root / "data").mkdir()
    write_jsonl(root / "data" / "fixture.jsonl", FIXTURE_RESPONSES)
    write_json(root / "data" / "fixture.truth.json", FIXTURE_TRUTH)
    settings = {"GENBENCH_OUT_DIR": str(root / "out"), "MODEL_BASE_URL": ""}
    for model in ("mock:good", "mock:bad"):
        execute(RunRequest(responses=str(root / "data" / "fixture.jsonl"),
                           truth=str(root / "data" / "fixture.truth.json"), model=model),
                log=lambda _: None, settings=settings, root=root)
    app = App(root, settings=settings, models_path=root / "config" / "models.yml",
              variants_path=root / "config" / "variants.yml")
    server = serve(0, app=app)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        with urllib.request.urlopen(base + "/api/export?model=mock:bad") as res:
            assert res.headers["Content-Type"].endswith("spreadsheetml.sheet")
            assert "attachment" in res.headers["Content-Disposition"]
            book = sheets(res.read())
        assert column(book["Runs"], "Model") == ["mock:bad"]
        try:
            urllib.request.urlopen(base + "/api/export?model=nothing")
            raise AssertionError("expected an error")
        except urllib.error.HTTPError as err:
            assert err.code == 400 and "nothing to export" in json.loads(err.read())["error"]
    finally:
        server.shutdown()
        server.server_close()
