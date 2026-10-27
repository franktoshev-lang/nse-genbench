"""The UI's web server: a JSON API over the same code as scripts/run.py.

Standard library only (http.server), bound to 127.0.0.1, so it runs in the
supervised offline environment too. Model runs go through one worker thread,
one at a time, so nothing is ever called in parallel against the budget.
"""
from __future__ import annotations

import json
import mimetypes
import queue
import random
import threading
import traceback
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .. import prompts
from ..config_io import update_dotenv, update_model_field
from ..data import DataError, load_responses
from ..pipeline import MAX_DOCUMENTS, MODES, group_default_mode
from ..runners import (MODELS_CONFIG, MOCK_CARDS, ModelNotConfigured, ModelRequestError,
                       load_models_config)
from ..runs import ROOT, RunRequest, UsageError, current_settings, execute, out_dir
from ..store import (MATRIX_METRICS, RATING_DIMENSIONS, ScoreError, load_results, load_scores,
                     matrix, save_score, spend, summary_rows)
from ..variants import (BASELINE_ID, PROMPT_KINDS, VARIANTS_CONFIG, VariantError, delete_variant,
                        load_variants, preview, prompts_for, render, save_variant, word_diff)

STATIC = Path(__file__).resolve().parent / "static"
EXPECTED_ERRORS = (FileNotFoundError, DataError, UsageError, ModelNotConfigured, ModelRequestError,
                   VariantError, ScoreError, ValueError, KeyError)


class ApiError(Exception):
    def __init__(self, message: str, status: int = HTTPStatus.BAD_REQUEST) -> None:
        super().__init__(message)
        self.status = status


class Jobs:
    """Model runs, executed one at a time on a background thread."""

    def __init__(self, app: "App") -> None:
        self.app = app
        self.jobs: dict[str, dict] = {}
        self.queue: queue.Queue = queue.Queue()
        threading.Thread(target=self._work, daemon=True).start()

    def submit(self, request: RunRequest, label: str) -> dict:
        job = {"id": uuid.uuid4().hex[:10], "label": label, "status": "queued", "log": [],
               "run_ids": [], "error": None}
        self.jobs[job["id"]] = job
        self.queue.put((job, request))
        return job

    def _work(self) -> None:
        while True:
            job, request = self.queue.get()
            job["status"] = "running"
            try:
                records = execute(request, log=job["log"].append, transport=self.app.transport,
                                  settings=self.app.settings(), root=self.app.root,
                                  models_path=self.app.models_path,
                                  variants_path=self.app.variants_path)
                job["run_ids"] = [r["run_id"] for r in records]
                job["status"] = "done"
            except EXPECTED_ERRORS as exc:
                job["error"], job["status"] = str(exc), "error"
            except Exception as exc:                      # keep the worker alive
                job["error"], job["status"] = f"{type(exc).__name__}: {exc}", "error"
                job["log"].append(traceback.format_exc())


class App:
    def __init__(self, root: Path = ROOT, *, data_dir: Path | None = None, settings: dict | None = None,
                 transport=None, models_path: Path = MODELS_CONFIG,
                 variants_path: Path = VARIANTS_CONFIG) -> None:
        self.root = root
        self.data_dir = data_dir or root / "data"
        self.models_path = models_path
        self.variants_path = variants_path
        self._settings = settings
        self.transport = transport
        self.jobs = Jobs(self)
        self._responses_cache: dict[str, dict[str, str]] = {}

    # ── helpers ────────────────────────────────────────────────────────────
    def settings(self) -> dict:
        return dict(self._settings) if self._settings is not None else current_settings(self.root)

    def out(self) -> Path:
        return out_dir(self.settings(), self.root)

    def slices(self) -> list[dict]:
        data = self.data_dir
        found = []
        for path in sorted(data.glob("*.jsonl")) if data.is_dir() else []:
            truth = path.with_name(path.stem + ".truth.json")
            info = {"id": path.stem, "responses": str(path),
                    "truth": str(truth) if truth.is_file() else None}
            try:
                n = len(load_responses(path))
                info["n"] = n
                info["group_mode"] = group_default_mode(min(n, MAX_DOCUMENTS))
            except EXPECTED_ERRORS as exc:
                info["error"] = str(exc)
            if info["truth"]:
                t = json.loads(truth.read_text(encoding="utf-8"))
                info.update(theme=t.get("theme"), planted=t.get("planted"),
                            minority=t.get("minority_theme"), absent=t.get("absent_theme"))
            found.append(info)
        return found

    def evidence_texts(self, responses_file: str) -> dict[str, str]:
        if responses_file not in self._responses_cache:
            path = Path(responses_file)
            path = path if path.is_absolute() else self.root / path
            try:
                self._responses_cache[responses_file] = {r.evidence_id: r.document
                                                         for r in load_responses(path)}
            except EXPECTED_ERRORS:
                return {}
        return self._responses_cache[responses_file]

    # ── API ────────────────────────────────────────────────────────────────
    def state(self) -> dict:
        settings = self.settings()
        models_config = load_models_config(self.models_path)
        endpoint_set = bool(settings.get("MODEL_BASE_URL", "").strip())
        results = load_results(self.out())
        variants = load_variants(self.variants_path)
        runs_per_variant = {}
        for rec in results:
            runs_per_variant[rec.get("variant")] = runs_per_variant.get(rec.get("variant"), 0) + 1
        models = []
        for key, spec in models_config["models"].items():
            model_id = str(spec.get("model_id") or "").strip()
            reason = None if model_id and endpoint_set else (
                "no endpoint yet (Settings)" if not endpoint_set else "model id missing (Settings)")
            models.append({"key": key, "name": spec.get("name", key), "role": spec.get("role", ""),
                           "model_id": model_id, "temperature": spec.get("temperature", True),
                           "token_limit_param": spec.get("token_limit_param", "max_tokens"),
                           "price_in_per_million": spec.get("price_in_per_million"),
                           "price_out_per_million": spec.get("price_out_per_million"),
                           "ready": reason is None, "not_ready_reason": reason})
        return {
            "slices": self.slices(),
            "models": models,
            "mocks": [{"key": f"mock:{v}", "name": f"Mock: {v} card", "ready": True} for v in MOCK_CARDS],
            "variants": [{"id": vid, "dimension": spec["dimension"],
                          "description": spec.get("description", ""),
                          "edits": spec.get("replace") or [], "runs": runs_per_variant.get(vid, 0),
                          "locked": vid == BASELINE_ID} for vid, spec in variants.items()],
            "endpoint": {"base_url": settings.get("MODEL_BASE_URL", ""),
                         "api_key_set": bool(settings.get("MODEL_API_KEY", "").strip())},
            "spend": spend(results, models_config),
            "themes": [t for t in prompts.THEMES_LIST if t != prompts.LOW_INFORMATION_THEME],
            "modes": list(MODES), "prompt_kinds": list(PROMPT_KINDS),
            "rating_dimensions": list(RATING_DIMENSIONS), "matrix_metrics": MATRIX_METRICS,
            "n_results": len(results),
        }

    def start_run(self, body: dict) -> dict:
        slice_id = body.get("slice")
        chosen = next((s for s in self.slices() if s["id"] == slice_id), None)
        if chosen is None:
            raise ApiError(f"unknown slice {slice_id!r}")
        if not body.get("model"):
            raise ApiError("pick a model")
        try:
            repeats = int(body.get("repeats") or 1)
        except ValueError:
            raise ApiError("repeats must be a number") from None
        if not 1 <= repeats <= 20:
            raise ApiError("repeats must be between 1 and 20")
        mode = body.get("mode") or None
        if mode not in (None, *MODES):
            raise ApiError(f"mode must be one of {MODES}")

        def number(name, cast):
            value = body.get(name)
            if value in (None, ""):
                return None
            try:
                return cast(value)
            except ValueError:
                raise ApiError(f"{name} must be a number") from None

        request = RunRequest(
            responses=chosen["responses"], truth=chosen.get("truth"),
            model=body["model"], variant=body.get("variant") or BASELINE_ID, repeats=repeats,
            mode=mode, theme=body.get("theme") or None,
            temperature=number("temperature", float), max_tokens=number("max_tokens", int))
        label = f"{request.model} / {request.variant} / {slice_id} x{repeats}"
        job = self.jobs.submit(request, label)
        return {"job_id": job["id"]}

    def job(self, job_id: str) -> dict:
        job = self.jobs.jobs.get(job_id)
        if job is None:
            raise ApiError(f"unknown job {job_id!r}", HTTPStatus.NOT_FOUND)
        return {k: (list(v) if k == "log" else v) for k, v in job.items()}

    def results(self, query: dict) -> dict:
        rows = summary_rows(load_results(self.out()), load_scores(self.out()))
        for field in ("slice", "model", "variant"):
            if query.get(field):
                rows = [r for r in rows if r[field] == query[field]]
        if query.get("scored") in ("yes", "no"):
            rows = [r for r in rows if r["scored"] == (query["scored"] == "yes")]
        if query.get("run_ids"):
            wanted = set(query["run_ids"].split(","))
            rows = [r for r in rows if r["run_id"] in wanted]
        return {"rows": rows}

    def result(self, run_id: str) -> dict:
        record = next((r for r in load_results(self.out()) if r["run_id"] == run_id), None)
        if record is None:
            raise ApiError(f"unknown run {run_id!r}", HTTPStatus.NOT_FOUND)
        texts = self.evidence_texts(record.get("responses_file", ""))
        truth_labels, off_theme = {}, set()
        if record.get("truth_file"):
            tpath = Path(record["truth_file"])
            tpath = tpath if tpath.is_absolute() else self.root / tpath
            if tpath.is_file():
                truth = json.loads(tpath.read_text(encoding="utf-8"))
                truth_labels = truth.get("labels") or {}
                off_theme = set(truth.get("off_theme_ids") or [])
        cited = {}
        for entry in (record.get("card") or {}).get("subtheme_manifest") or []:
            if isinstance(entry, dict):
                for eid in entry.get("evidence_ids") or []:
                    eid = str(eid)
                    cited[eid] = {"text": texts.get(eid), "exists": eid in texts,
                                  "label": truth_labels.get(eid), "off_theme": eid in off_theme}
        return {"record": record, "cited": cited, "score": load_scores(self.out()).get(run_id)}

    def save_score(self, body: dict) -> dict:
        known = {r["run_id"] for r in load_results(self.out())}
        claims = {str(k): v for k, v in (body.get("claims") or {}).items() if v is not None}
        return save_score(self.out(), body.get("run_id", ""), ratings=body.get("ratings"),
                          claims=claims, note=body.get("note", ""), scorer=body.get("scorer", ""),
                          known_run_ids=known)

    def blind_pick(self, query: dict) -> dict:
        """A random unscored run, so scoring order says nothing about the model."""
        include_mocks = query.get("include_mocks") == "yes"
        rows = [r for r in summary_rows(load_results(self.out()), load_scores(self.out()))
                if not r["scored"] and not r["parse_error"]
                and (not query.get("slice") or r["slice"] == query["slice"])
                and (include_mocks or not str(r["model"]).startswith("mock:"))]
        if not rows:
            return {"run_id": None, "remaining": 0}
        return {"run_id": random.choice(rows)["run_id"], "remaining": len(rows)}

    def matrix(self, query: dict) -> dict:
        return matrix(load_results(self.out()), load_scores(self.out()), query.get("slice") or None)

    def variant_prompt(self, variant_id: str, query: dict) -> dict:
        kind = query.get("kind") or "single"
        theme = query.get("theme") or "Examination & Assessment"
        before = render(prompts, kind, theme)
        after = render(prompts_for(variant_id, self.variants_path), kind, theme)
        return {"diff": word_diff(before, after), "changed": before != after}

    def variant_preview(self, body: dict) -> dict:
        edits = [(str(e.get("old", "")), str(e.get("new", ""))) for e in body.get("edits") or []]
        return preview(edits, body.get("kind") or "single",
                       body.get("theme") or "Examination & Assessment")

    def variant_create(self, body: dict) -> dict:
        edits = [(str(e.get("old", "")), str(e.get("new", ""))) for e in body.get("edits") or []]
        return save_variant(str(body.get("id", "")).strip(), str(body.get("dimension", "")),
                            str(body.get("description", "")), edits, path=self.variants_path)

    def variant_delete(self, variant_id: str) -> dict:
        delete_variant(variant_id, self.variants_path)
        return {"ok": True}

    def save_settings(self, body: dict) -> dict:
        updates = {}
        if "base_url" in body:
            base_url = str(body["base_url"]).strip().rstrip("/")
            if base_url and not base_url.startswith(("http://", "https://")):
                raise ApiError("the endpoint must start with http:// or https://")
            updates["MODEL_BASE_URL"] = base_url
        if body.get("api_key"):
            updates["MODEL_API_KEY"] = str(body["api_key"]).strip()
        if updates:
            update_dotenv(self.root / ".env", updates, example=self.root / ".env.example")
            if self._settings is not None:
                self._settings.update(updates)
        for key, fields in (body.get("models") or {}).items():
            for field, value in fields.items():
                update_model_field(key, field, value, self.models_path)
        return {"ok": True}


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload) -> None:
            self._send(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if not length:
                return {}
            try:
                data = json.loads(self.rfile.read(length))
            except json.JSONDecodeError:
                raise ApiError("request body is not valid JSON") from None
            if not isinstance(data, dict):
                raise ApiError("request body must be a JSON object")
            return data

        def _static(self, path: str) -> None:
            name = "index.html" if path in ("", "/") else path.removeprefix("/static/")
            file = (STATIC / name).resolve()
            if STATIC.resolve() not in file.parents or not file.is_file():
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            ctype = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
            if ctype.startswith("text/") or ctype in ("application/javascript",):
                ctype += "; charset=utf-8"
            self._send(HTTPStatus.OK, file.read_bytes(), ctype)

        def _dispatch(self, method: str) -> None:
            url = urlparse(self.path)
            path = url.path
            query = {k: v[-1] for k, v in parse_qs(url.query).items()}
            try:
                if not path.startswith("/api/"):
                    if method == "GET":
                        return self._static(path)
                    raise ApiError("not found", HTTPStatus.NOT_FOUND)
                parts = path.strip("/").split("/")[1:]
                route = (method, parts[0] if parts else "", len(parts))
                if route == ("GET", "state", 1):
                    result = app.state()
                elif route == ("POST", "run", 1):
                    result = app.start_run(self._body())
                elif route == ("GET", "jobs", 2):
                    result = app.job(parts[1])
                elif route == ("GET", "results", 1):
                    result = app.results(query)
                elif route == ("GET", "results", 2):
                    result = app.result(parts[1])
                elif route == ("POST", "scores", 1):
                    result = app.save_score(self._body())
                elif route == ("GET", "blind", 1):
                    result = app.blind_pick(query)
                elif route == ("GET", "matrix", 1):
                    result = app.matrix(query)
                elif route == ("POST", "variants", 2) and parts[1] == "preview":
                    result = app.variant_preview(self._body())
                elif route == ("POST", "variants", 1):
                    result = app.variant_create(self._body())
                elif route == ("DELETE", "variants", 2):
                    result = app.variant_delete(parts[1])
                elif route == ("GET", "variants", 3) and parts[2] == "prompt":
                    result = app.variant_prompt(parts[1], query)
                elif route == ("POST", "settings", 1):
                    result = app.save_settings(self._body())
                else:
                    raise ApiError("not found", HTTPStatus.NOT_FOUND)
                self._json(HTTPStatus.OK, result)
            except ApiError as exc:
                self._json(exc.status, {"error": str(exc)})
            except EXPECTED_ERRORS as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc).strip("'\"")})
            except Exception as exc:
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR,
                           {"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()})

        def do_GET(self) -> None:
            self._dispatch("GET")

        def do_POST(self) -> None:
            self._dispatch("POST")

        def do_DELETE(self) -> None:
            self._dispatch("DELETE")

    return Handler


def serve(port: int = 8770, host: str = "127.0.0.1", app: App | None = None) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(app or App()))
    return server
