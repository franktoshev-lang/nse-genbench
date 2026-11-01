"""Export runs to an Excel workbook, to read and score them by hand.

One workbook, standard library only (an .xlsx file is a zip of XML files):

  Read me       what each sheet holds and how to fill it in
  Runs          one row per run: model, prompt variant, the card, the
                automatic scores, and empty columns for your 1-5 ratings
  Claims        one row per answer a card cites, with the answer text and
                what the answer key says it is about, and a yes/no column
  Input <slice> the student answers the model was given, with the answer key
  Prompts       the exact prompts sent, step by step, and the model's answer
                to each step

The prompts are not stored with a run, so they are rebuilt here: the same
slice, variant edits and settings give the same prompts. A map-reduce run's
reduce prompt is built from the model's batch answers, which come from the
response cache (out/cache); if those are missing, the reduce prompt is
marked as not available. Every rebuilt prompt is checked against the
length recorded when the run was made.
"""
from __future__ import annotations

import json
import re
import zipfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape

from . import prompts as baseline_prompts
from .data import load_responses
from .pipeline import BATCH_DOCUMENTS, AnswerParseError, generate_card
from .runners import (MOCK_CARDS, ChatCompletionsRunner, MockRunner, ModelConfig, ResponseCache,
                      _answer_text, sha256_text)
from .store import RATING_DIMENSIONS
from .variants import VariantError, _build, load_variants

CELL_LIMIT = 32767          # Excel's maximum characters in one cell
_ILLEGAL_XML = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


# ── a minimal .xlsx writer ─────────────────────────────────────────────────
STYLE_DEFAULT, STYLE_HEADER, STYLE_WRAP = 0, 1, 2

_STYLES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="11"/><name val="Calibri"/></font></fonts>
<fills count="3"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill>
<fill><patternFill patternType="solid"><fgColor rgb="FFE8EEF2"/><bgColor indexed="64"/></patternFill></fill></fills>
<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
<cellXfs count="3">
<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0" applyAlignment="1"><alignment vertical="top"/></xf>
<xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1" applyAlignment="1"><alignment vertical="top" wrapText="1"/></xf>
<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0" applyAlignment="1"><alignment vertical="top" wrapText="1"/></xf>
</cellXfs>
<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>"""


def _col(n: int) -> str:
    """0 -> A, 25 -> Z, 26 -> AA."""
    out = ""
    n += 1
    while n:
        n, r = divmod(n - 1, 26)
        out = chr(65 + r) + out
    return out


def _cell_text(value) -> str:
    text = _ILLEGAL_XML.sub("", str(value))
    if len(text) > CELL_LIMIT:
        note = "\n[cut here: longer than one Excel cell can hold]"
        text = text[:CELL_LIMIT - len(note)] + note
    return text


class Workbook:
    def __init__(self) -> None:
        self.sheets: list[tuple[str, str]] = []

    def add_sheet(self, name: str, header: list[str], rows: list[list], widths: list[int],
                  wrap: bool = True) -> None:
        name = re.sub(r"[\[\]:*?/\\]", "-", name)[:31]
        cols = "".join(f'<col min="{i + 1}" max="{i + 1}" width="{w}" customWidth="1"/>'
                       for i, w in enumerate(widths))
        lines = []
        for r, row in enumerate([header, *rows], start=1):
            style = STYLE_HEADER if r == 1 else (STYLE_WRAP if wrap else STYLE_DEFAULT)
            cells = []
            for c, value in enumerate(row):
                ref = f"{_col(c)}{r}"
                if value is None or value == "":
                    continue
                if isinstance(value, bool):
                    value = "yes" if value else "no"
                if isinstance(value, (int, float)):
                    cells.append(f'<c r="{ref}" s="{style}"><v>{value}</v></c>')
                else:
                    cells.append(f'<c r="{ref}" s="{style}" t="inlineStr"><is><t xml:space="preserve">'
                                 f'{escape(_cell_text(value))}</t></is></c>')
            lines.append(f'<row r="{r}">{"".join(cells)}</row>')
        last = f"{_col(max(len(header), 1) - 1)}{len(rows) + 1}"
        xml = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
               '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
               '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" '
               'activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
               f'<cols>{cols}</cols><sheetData>{"".join(lines)}</sheetData>'
               f'<autoFilter ref="A1:{last}"/></worksheet>')
        self.sheets.append((name, xml))

    def to_bytes(self) -> bytes:
        buf = BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("[Content_Types].xml",
                       '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                       '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                       '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                       '<Default Extension="xml" ContentType="application/xml"/>'
                       '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
                       '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
                       + "".join(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                                 for i in range(1, len(self.sheets) + 1))
                       + "</Types>")
            z.writestr("_rels/.rels",
                       '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                       '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                       '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
                       "</Relationships>")
            z.writestr("xl/workbook.xml",
                       '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                       '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
                       'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>'
                       + "".join(f'<sheet name="{escape(name, {chr(34): "&quot;"})}" sheetId="{i}" r:id="rId{i}"/>'
                                 for i, (name, _) in enumerate(self.sheets, start=1))
                       + "</sheets></workbook>")
            z.writestr("xl/_rels/workbook.xml.rels",
                       '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                       '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                       + "".join(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>'
                                 for i in range(1, len(self.sheets) + 1))
                       + f'<Relationship Id="rId{len(self.sheets) + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
                       "</Relationships>")
            z.writestr("xl/styles.xml", _STYLES)
            for i, (_, xml) in enumerate(self.sheets, start=1):
                z.writestr(f"xl/worksheets/sheet{i}.xml", xml)
        return buf.getvalue()


# ── rebuilding the prompts a run sent ─────────────────────────────────────
class _Replay:
    """Records every prompt and answers it from the run's own cached
    responses, so nothing is sent to a model. A missing answer is replaced by
    an empty object so the later batch prompts can still be rebuilt; the
    reduce prompt is then dropped, because it would be built from that gap."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.key = inner.key
        self.steps: list[dict] = []
        self.missing: list[int] = []

    def generate(self, prompt: str) -> str:
        step = {"prompt": prompt, "answer": None}
        self.steps.append(step)
        if isinstance(self.inner, MockRunner):
            step["answer"] = self.inner.generate(prompt)
            return step["answer"]
        record = self.inner.cache.get(self.inner.key, sha256_text(prompt), self.inner.input_hash())
        if record is None:
            self.missing.append(len(self.steps) - 1)
            return "{}"
        step["answer"] = _answer_text(record["response"], self.inner.key)
        return step["answer"]


def _resolve(path: str | None, root: Path) -> Path | None:
    if not path:
        return None
    p = Path(str(path).replace("\\", "/"))
    return p if p.is_absolute() else root / p


def rebuild_steps(record: dict, root: Path, settings: dict, out_dir: Path) -> tuple[list[dict], str]:
    """[{step, prompt, answer, chars_ok}] for one run, and a note when not everything
    could be rebuilt."""
    responses_path = _resolve(record.get("responses_file"), root)
    if responses_path is None or not responses_path.is_file():
        return [], "the responses file is no longer there"
    edits = tuple(tuple(e) for e in record.get("variant_edits") or [])
    try:
        module = _build(f"export:{record.get('variant')}", edits) if edits else baseline_prompts
    except VariantError as exc:
        return [], f"the variant's edits no longer apply ({exc})"

    model_key = str(record.get("model_key") or "")
    if model_key.startswith("mock:") and model_key.split(":", 1)[1] in MOCK_CARDS:
        inner = MockRunner(model_key.split(":", 1)[1])
    else:
        req = record.get("request") or {}
        base_url = (settings.get("MODEL_BASE_URL") or "").strip().rstrip("/") or "http://endpoint-not-set"
        config = ModelConfig(
            base_url=base_url, model=str(record.get("model") or "unknown"),
            max_tokens=int(req.get("max_tokens") or 8192),
            temperature=float(req["temperature"]) if req.get("temperature") is not None else 0.1,
            send_temperature=req.get("temperature") is not None, seed=req.get("seed"),
            token_limit_param=req.get("token_limit_param") or "max_tokens",
            response_format=req.get("response_format"))
        inner = ChatCompletionsRunner(config, cache=ResponseCache(out_dir / "cache"),
                                      sample=int(record.get("sample") or 0))
    replay = _Replay(inner)
    note = ""
    try:
        generate_card(replay, record.get("theme") or "", load_responses(responses_path),
                      mode=record.get("mode"), prompt_module=module,
                      batch_documents=int(record.get("batch_documents") or BATCH_DOCUMENTS),
                      log=lambda _msg: None)
    except AnswerParseError:
        pass                     # the run itself failed here; the steps so far are complete
    rebuilt = replay.steps
    if replay.missing:
        note = "the model's answer to some steps is not in out/cache"
        if record.get("mode") == "mapreduce" and any(i < len(rebuilt) - 1 for i in replay.missing):
            rebuilt = rebuilt[:-1]
            note += "; the reduce prompt is built from the batch answers, so it cannot be rebuilt"
    calls = record.get("calls") or []
    steps = []
    for i, step in enumerate(rebuilt):
        call = calls[i] if i < len(calls) else {}
        steps.append({"step": call.get("step") or f"step {i + 1}", "prompt": step["prompt"],
                      "answer": step["answer"],
                      "chars_ok": call.get("prompt_chars") in (None, len(step["prompt"]))})
    if any(not s["chars_ok"] for s in steps):
        note = (note + "; " if note else "") + "a rebuilt prompt differs in length from the one sent"
    return steps, note


# ── the workbook ──────────────────────────────────────────────────────────
def _lines(items) -> str:
    return "\n".join(f"- {x}" for x in items or [] if str(x).strip())


def _subthemes(card: dict) -> str:
    out = []
    for entry in card.get("subtheme_manifest") or []:
        if isinstance(entry, dict):
            ids = ", ".join(str(e) for e in entry.get("evidence_ids") or [])
            out.append(f"- {entry.get('name', '')}: {entry.get('description', '')} [{ids}]")
    return "\n".join(out) or _lines(card.get("subthemes"))


def _slice_data(record: dict, root: Path, cache: dict) -> dict:
    key = (record.get("responses_file"), record.get("truth_file"))
    if key in cache:
        return cache[key]
    texts, kinds, labels, off, langs = {}, {}, {}, set(), {}
    rpath = _resolve(record.get("responses_file"), root)
    if rpath and rpath.is_file():
        for r in load_responses(rpath):
            texts[r.evidence_id], kinds[r.evidence_id] = r.document, r.evidence_type
    tpath = _resolve(record.get("truth_file"), root)
    if tpath and tpath.is_file():
        truth = json.loads(tpath.read_text(encoding="utf-8"))
        labels, off = truth.get("labels") or {}, set(truth.get("off_theme_ids") or [])
        langs = truth.get("languages") or {}
    cache[key] = {"slice": record.get("slice"), "texts": texts, "kinds": kinds, "labels": labels,
                  "off": off, "langs": langs, "has_truth": bool(tpath and tpath.is_file())}
    return cache[key]


def _planted_as(data: dict, eid: str) -> str:
    if not data["has_truth"]:
        return ""
    if eid in data["off"]:
        return "OFF-THEME (should be ignored)"
    if eid not in data["texts"]:
        return "not in the input"
    return data["labels"].get(eid) or "noise (no subtheme)"


def _yes_no(value) -> str:
    return "" if value is None else ("yes" if value else "no")


def build_workbook(records: list[dict], scores: dict[str, dict], *, root: Path, settings: dict,
                   out_dir: Path, variants_path=None, filters: dict | None = None) -> bytes:
    variants = {}
    try:
        variants = load_variants(variants_path) if variants_path else load_variants()
    except (OSError, VariantError):
        pass
    slices: dict = {}
    runs_rows, claim_rows, prompt_rows = [], [], []
    for rec in records:
        card = rec.get("card") or {}
        s = rec.get("scores") or {}
        tokens = rec.get("tokens") or {}
        score = scores.get(rec["run_id"]) or {}
        ratings = score.get("ratings") or {}
        spec = variants.get(rec.get("variant")) or {}
        settings_text = ", ".join(f"{k}={v}" for k, v in (rec.get("variant_settings") or {}).items())
        changes = rec.get("variant_dimension") or ""
        what = " ".join(str(spec.get("description") or "").split())
        if settings_text:
            what = (what + " " if what else "") + f"[settings: {settings_text}]"
        runs_rows.append([
            rec["run_id"], (rec.get("created_utc") or "")[:19].replace("T", " "), rec.get("slice"),
            rec.get("model_key"), rec.get("model"), rec.get("sample"), rec.get("variant"), changes, what,
            rec.get("mode"), rec.get("batch_documents") or (BATCH_DOCUMENTS if rec.get("mode") == "mapreduce" else ""),
            card.get("summary") or "", _lines(card.get("positive_comments")),
            _lines(card.get("critical_comments")), _lines(card.get("student_suggestions")),
            _subthemes(card), s.get("valid_json"), s.get("coverage"), s.get("minority_preserved"),
            s.get("hallucinated_absent"), s.get("off_theme_cited"), s.get("evidence_validity"),
            s.get("evidence_support"), s.get("lang"), s.get("words"),
            tokens.get("prompt_tokens"), tokens.get("completion_tokens"), tokens.get("reasoning_tokens"),
            rec.get("parse_error") or "",
            *[ratings.get(d) for d in RATING_DIMENSIONS], score.get("note") or "",
        ])
        data = _slice_data(rec, root, slices)
        claims = score.get("claims") or {}
        for index, entry in enumerate(card.get("subtheme_manifest") or []):
            if not isinstance(entry, dict):
                continue
            for eid in entry.get("evidence_ids") or [] or [""]:
                eid = str(eid)
                claim_rows.append([
                    rec["run_id"], rec.get("model_key"), rec.get("variant"), index + 1,
                    entry.get("name", ""), entry.get("description", ""), eid,
                    data["texts"].get(eid, "(this ID is not in the input: invented)") if eid else "(no ID cited)",
                    _planted_as(data, eid) if eid else "", _yes_no(claims.get(str(index))),
                ])
        steps, note = rebuild_steps(rec, root, settings, out_dir)
        for step in steps:
            prompt_rows.append([rec["run_id"], rec.get("model_key"), rec.get("variant"), step["step"],
                                len(step["prompt"]), step["prompt"],
                                step["answer"] if step["answer"] is not None else "(not in out/cache)",
                                "" if step["chars_ok"] else "length differs from the run"])
        if note:
            prompt_rows.append([rec["run_id"], rec.get("model_key"), rec.get("variant"), "note", "", note, "", ""])

    wb = Workbook()
    made = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    shown = ", ".join(f"{k} = {v}" for k, v in (filters or {}).items() if v) or "none (all runs)"
    readme = [
        ["Exported", made],
        ["Runs in this file", len(records)],
        ["Filters", shown],
        ["Runs", "One row per run: which model and which prompt variant, the card the model wrote, "
                 "the automatic scores, and empty columns for your ratings (1 = poor, 5 = excellent). "
                 "Ratings and notes already given in the app are filled in."],
        ["Claims", "One row per student answer a card cites for a subtheme. 'Planted as' is the answer "
                   "key: the subtheme that answer was written for. If it matches the card's subtheme, "
                   "the citation supports the claim. Fill in 'Supported?' with yes or no per subtheme. "
                   "An ID that is not in the input was invented by the model."],
        ["Input sheets", "The student answers of each test slice, as the model got them, with the answer "
                         "key. Everything in them is invented test data, not real survey answers."],
        ["Prompts", "The exact prompt of every step of every run, rebuilt from the slice and the "
                    "variant, and the model's answer to each step. In map-reduce runs the map steps "
                    "summarise one batch each and the reduce step merges them into the card."],
        ["Scores in this file", "Scores typed here stay in this file. The app's Matrix only counts "
                                "scores saved in the app."],
    ]
    wb.add_sheet("Read me", ["Item", "What it is"], readme, [22, 110])
    wb.add_sheet("Runs", [
        "Run ID", "When (UTC)", "Slice", "Model", "Model id", "Repeat", "Prompt variant",
        "Variant changes", "What the variant does", "Mode", "Batch size",
        "Summary", "Positive comments", "Critical comments", "Student suggestions",
        "Subthemes (with cited answer IDs)", "Valid output", "Coverage", "Rare subtheme kept",
        "Invented subtheme", "Off-theme answers cited", "Cited IDs exist", "Cited answers match",
        "Language", "Words", "Tokens in", "Tokens out", "of which reasoning", "Parse error",
        *[f"{d.capitalize()} (1-5)" for d in RATING_DIMENSIONS], "Notes",
    ], runs_rows, [13, 17, 8, 18, 22, 7, 16, 18, 40, 10, 8, 50, 45, 45, 45, 60,
                   8, 9, 9, 9, 9, 9, 9, 8, 7, 9, 9, 9, 20, 12, 14, 13, 40])
    wb.add_sheet("Claims", [
        "Run ID", "Model", "Prompt variant", "Subtheme #", "Subtheme (as the card names it)",
        "Description", "Cited answer ID", "Student answer", "Planted as (answer key)",
        "Supported? (yes/no)",
    ], claim_rows, [13, 18, 16, 10, 30, 45, 10, 60, 28, 14])
    for data in sorted(slices.values(), key=lambda d: str(d["slice"])):
        rows = [[eid, data["kinds"].get(eid), text, _planted_as(data, eid), data["langs"].get(eid, "")]
                for eid, text in data["texts"].items()]
        wb.add_sheet(f"Input {data['slice']}", ["Answer ID", "Type", "Student answer",
                                                 "Planted as (answer key)", "Language"],
                     rows, [10, 12, 80, 30, 9])
    wb.add_sheet("Prompts", ["Run ID", "Model", "Prompt variant", "Step", "Characters",
                             "Prompt sent", "Model answer to this step", "Check"],
                 prompt_rows, [13, 18, 16, 10, 10, 100, 80, 14])
    return wb.to_bytes()


def export_filename(filters: dict | None = None) -> str:
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    parts = [str(v) for k, v in (filters or {}).items() if v and k in ("slice", "model", "variant")]
    tail = ("_" + "_".join(re.sub(r"[^A-Za-z0-9._-]+", "-", p) for p in parts)) if parts else ""
    return f"genbench-export_{stamp}{tail}.xlsx"
