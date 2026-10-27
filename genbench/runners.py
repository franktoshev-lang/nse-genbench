"""Stage 3 - MODEL. Runners turn one prompt string into one raw answer string.

MockRunner needs no GPU and no network. It exists so the whole pipeline can
be exercised end to end, and so the metrics can be unit-tested against cards
whose correct scores are known in advance.

ChatCompletionsRunner talks to any OpenAI-compatible /chat/completions
endpoint (llama-server, a hosted API, the Fontys gateway if it exposes one),
configured from .env. It is written out fully but stays unconnected until
MODEL_BASE_URL is set. Standard library only, so no extra dependencies.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import yaml


class Runner(Protocol):
    key: str
    def generate(self, prompt: str) -> str: ...


# ── Mock cards: known-good, known-bad, known-fluent-but-ungrounded ──────────
GOOD_CARD = {
    "summary": "Studenten zijn overwegend kritisch over toetsing en beoordeling; "
               "feedback komt laat en criteria zijn onduidelijk.",
    "positive_comments": ["Docenten zijn betrokken en bereikbaar tijdens de les"],
    "critical_comments": [
        "Feedback op toetsen komt te laat om nog nuttig te zijn",
        "Beoordelingscriteria verschillen per docent",
        "Studiebelasting is ongelijk verdeeld en het rooster wijzigt laat",
    ],
    "student_suggestions": ["Beoordelingscriteria voor Projectmanagement zijn onduidelijk; je weet niet waar je op beoordeeld wordt."],
    "subthemes": ["Feedback en beoordeling", "Docenten en uitleg",
                  "Rooster en studiebelasting", "Studieloopbaanbegeleiding"],
    "subtheme_manifest": [
        {"name": "Feedback en beoordeling",
         "description": "Late feedback en onduidelijke beoordelingscriteria bij toetsen",
         "evidence_ids": ["E0002", "E0003"]},
        {"name": "Studieloopbaanbegeleiding",
         "description": "Weinig ondersteuning en begeleiding bij achterstand",
         "evidence_ids": []},
    ],
}

# Misses the minority theme, invents an ID, mentions the absent theme.
BAD_CARD = {
    "summary": "Students are broadly positive about the programme and the "
               "internship placements in particular.",
    "positive_comments": ["Strong industry links and professional practice"],
    "critical_comments": ["Some assessment feedback is slow"],
    "student_suggestions": [],
    "subthemes": ["Assessment", "Professional practice and internships"],
    "subtheme_manifest": [
        {"name": "Professional practice and internships",
         "description": "Students value the internship and industry contact",
         "evidence_ids": ["E0001", "E9999"]},
    ],
}

# Reproduces a single response near-verbatim in the analytical prose.
LEAKY_CARD = {
    "summary": "Beoordelingscriteria voor Projectmanagement zijn onduidelijk; "
               "je weet niet waar je op beoordeeld wordt.",
    "positive_comments": [],
    "critical_comments": ["Feedback is vaak alleen een cijfer zonder uitleg, daar leer ik niets van."],
    "student_suggestions": [],
    "subthemes": ["Beoordeling"],
    "subtheme_manifest": [{"name": "Beoordeling", "description": "toetsing",
                           "evidence_ids": ["E0002"]}],
}

MOCK_CARDS = {"good": GOOD_CARD, "bad": BAD_CARD, "leaky": LEAKY_CARD}


class MockRunner:
    """Returns a fixed card. No model, no network."""

    def __init__(self, variant: str = "good") -> None:
        if variant not in MOCK_CARDS:
            raise KeyError(f"unknown mock variant {variant!r}; choose from {list(MOCK_CARDS)}")
        self.key = f"mock:{variant}"
        self.variant = variant

    def generate(self, prompt: str) -> str:
        return json.dumps(MOCK_CARDS[self.variant], ensure_ascii=False)


# ── OpenAI-compatible chat-completions client ───────────────────────────────
# The group repo's code default is max_tokens=8192 (LlamaCppGenerationSettings
# in llama_cpp_models.py); its model registry raises that to 12768-18768 with
# thinking on. Never inherit a small value: verbose models truncate.
DEFAULT_MAX_TOKENS = 8192
# The group's code default. Its registry overrides it to 1.0 on every model;
# that override is an experimental condition, not something to inherit.
DEFAULT_TEMPERATURE = 0.1
DEFAULT_SEED = 42
DEFAULT_TIMEOUT_S = 600.0
DEFAULT_MAX_RETRIES = 3

TOKEN_LIMIT_PARAMS = ("max_tokens", "max_completion_tokens")
MODELS_CONFIG = Path(__file__).resolve().parents[1] / "config" / "models.yml"

RETRY_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504})
# Never sent: anything that lets the model call tools or search the web.
FORBIDDEN_PAYLOAD_KEYS = frozenset({
    "tools", "tool_choice", "functions", "function_call", "parallel_tool_calls",
    "plugins", "web_search_options",
})

# (url, body, headers, timeout_s) -> (status, response text, response headers)
Transport = Callable[[str, bytes, dict, float], "tuple[int, str, dict]"]


class ModelNotConfigured(RuntimeError):
    """No endpoint is configured, so no real model can be called."""


class ModelRequestError(RuntimeError):
    """The request to the endpoint did not produce an answer."""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_dotenv(path) -> dict[str, str]:
    """Minimal KEY=VALUE reader for a .env file. A missing file gives {}."""
    p = Path(path)
    if not p.is_file():
        return {}
    values: dict[str, str] = {}
    for line in p.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


@dataclass(frozen=True)
class ModelConfig:
    base_url: str
    model: str
    api_key: str = ""
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    seed: int | None = DEFAULT_SEED
    timeout_s: float = DEFAULT_TIMEOUT_S
    max_retries: int = DEFAULT_MAX_RETRIES
    # Per-model request shape (config/models.yml): some models reject a
    # temperature, or want the output limit under another field name.
    send_temperature: bool = True
    token_limit_param: str = "max_tokens"

    def __post_init__(self) -> None:
        if not self.base_url.strip():
            raise ModelNotConfigured(
                "MODEL_BASE_URL is empty, so there is no model to call. Set MODEL_BASE_URL, "
                "MODEL_NAME and MODEL_API_KEY in .env (see .env.example), or run with "
                "--model mock:good to exercise the pipeline without a model."
            )
        if not re.match(r"^https?://", self.base_url):
            raise ValueError(f"MODEL_BASE_URL must start with http:// or https://, got {self.base_url!r}")
        if not self.model.strip():
            raise ModelNotConfigured("MODEL_NAME is empty. Set it in .env or pass the model name with --model.")
        if self.model.endswith(":online"):
            raise ValueError(f"{self.model!r} turns on provider-side web search, which is never allowed")
        if not isinstance(self.max_tokens, int) or self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be a positive integer, got {self.max_tokens!r}")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature must be between 0 and 2, got {self.temperature!r}")
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries!r}")
        if self.token_limit_param not in TOKEN_LIMIT_PARAMS:
            raise ValueError(f"token_limit_param must be one of {TOKEN_LIMIT_PARAMS}, "
                             f"got {self.token_limit_param!r}")

    @classmethod
    def from_env(cls, dotenv_path=".env", environ=None, **overrides) -> "ModelConfig":
        """Settings in rising priority: the .env file, the process environment,
        then explicit keyword overrides (CLI flags). None overrides are ignored.

        GENBENCH_SEED unset -> 42; set but empty -> no seed is sent (for a
        provider that rejects the field)."""
        values = {**read_dotenv(dotenv_path), **(os.environ if environ is None else environ)}
        seed_raw = values.get("GENBENCH_SEED")
        settings = {
            "base_url": values.get("MODEL_BASE_URL", "").strip().rstrip("/"),
            "model": values.get("MODEL_NAME", "").strip(),
            "api_key": values.get("MODEL_API_KEY", "").strip(),
            "seed": DEFAULT_SEED if seed_raw is None else (int(seed_raw) if seed_raw.strip() else None),
        }
        settings.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**settings)


def load_model_settings(path=MODELS_CONFIG) -> dict[str, dict]:
    """The models under test, keyed by the name used with run.py --model."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    models = data.get("models") or {}
    for key, spec in models.items():
        if spec.get("token_limit_param", "max_tokens") not in TOKEN_LIMIT_PARAMS:
            raise ValueError(f"{path}: {key}: token_limit_param must be one of {TOKEN_LIMIT_PARAMS}")
        if not isinstance(spec.get("temperature", True), bool):
            raise ValueError(f"{path}: {key}: temperature must be true or false")
    return models


class ResponseCache:
    """Every raw response on disk, keyed by (model, prompt hash, input hash),
    so scoring can be re-run without calling the model again.

    The input hash covers everything besides the prompt that changes the
    answer: endpoint, max_tokens, temperature, seed and the repeat index.
    Without the repeat index, five repeats of one prompt would all return the
    first cached answer. Prompt text is not stored, only its hash; the model's
    answer is, so treat the cache directory as data.
    """

    def __init__(self, root) -> None:
        self.root = Path(root)

    def path(self, model: str, prompt_hash: str, input_hash: str) -> Path:
        safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("_") or "model"
        return self.root / safe_model / prompt_hash[:16] / f"{input_hash[:16]}.json"

    def get(self, model: str, prompt_hash: str, input_hash: str) -> dict | None:
        p = self.path(model, prompt_hash, input_hash)
        if not p.is_file():
            return None
        record = json.loads(p.read_text(encoding="utf-8"))
        if record.get("prompt_sha256") != prompt_hash or record.get("input_sha256") != input_hash:
            return None   # truncated-hash collision: treat as a miss
        return record

    def put(self, model: str, prompt_hash: str, input_hash: str, record: dict) -> Path:
        p = self.path(model, prompt_hash, input_hash)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True),
                       encoding="utf-8", newline="\n")
        os.replace(tmp, p)
        return p


def _urllib_post(url: str, body: bytes, headers: dict, timeout_s: float) -> tuple[int, str, dict]:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as resp:
            return resp.status, resp.read().decode("utf-8", "replace"), dict(resp.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace"), dict(exc.headers or {})


def _header(headers: dict, name: str) -> str | None:
    for key, value in (headers or {}).items():
        if key.lower() == name.lower():
            return value
    return None


def _backoff_seconds(attempt: int, retry_after: str | None) -> float:
    """1, 2, 4 ... seconds, capped; a numeric Retry-After header wins."""
    if retry_after is not None:
        try:
            return min(max(float(retry_after), 0.0), 60.0)
        except ValueError:
            pass
    return min(2.0 ** attempt, 30.0)


def _answer_text(data: dict, model: str) -> str:
    choices = data.get("choices") or []
    if not choices:
        raise ModelRequestError(f"response from {model!r} has no choices: {json.dumps(data)[:300]}")
    choice = choices[0]
    finish_reason = choice.get("finish_reason")
    if finish_reason == "length":
        print(f"[model] warning: {model!r} hit max_tokens (finish_reason=length); the answer may be truncated")
    content = (choice.get("message") or {}).get("content")
    if not isinstance(content, str):
        print(f"[model] warning: {model!r} returned no message content (finish_reason={finish_reason!r})")
        return ""
    return content


class ChatCompletionsRunner:
    """One client for any OpenAI-compatible /chat/completions endpoint.

    The request mirrors the group pipeline's: a single user message, no
    system message. The output limit, temperature and seed are always sent
    explicitly, except a temperature for a model configured not to take one;
    tool and web-search fields are never sent.
    """

    def __init__(self, config: ModelConfig, *, cache: ResponseCache | None = None,
                 sample: int = 0, transport: Transport | None = None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.config = config
        self.cache = cache
        self.sample = sample
        self.key = config.model
        self.last_from_cache = False
        self.last_usage: dict | None = None
        self._transport = transport or _urllib_post
        self._sleep = sleep

    @property
    def url(self) -> str:
        return f"{self.config.base_url.rstrip('/')}/chat/completions"

    def payload(self, prompt: str) -> dict:
        c = self.config
        payload = {
            "model": c.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            c.token_limit_param: c.max_tokens,
        }
        if c.send_temperature:
            payload["temperature"] = c.temperature
        if c.seed is not None:
            payload["seed"] = c.seed
        if FORBIDDEN_PAYLOAD_KEYS & payload.keys():
            raise RuntimeError("tool or web-search fields must never be sent")
        return payload

    def input_hash(self) -> str:
        c = self.config
        return sha256_text(json.dumps({
            "base_url": c.base_url.rstrip("/"), "max_tokens": c.max_tokens,
            "token_limit_param": c.token_limit_param,
            "temperature": c.temperature if c.send_temperature else None,
            "seed": c.seed, "sample": self.sample,
        }, sort_keys=True))

    def generate(self, prompt: str) -> str:
        prompt_hash, input_hash = sha256_text(prompt), self.input_hash()
        record = self.cache.get(self.key, prompt_hash, input_hash) if self.cache else None
        self.last_from_cache = record is not None
        if record is None:
            payload = self.payload(prompt)
            record = {
                "model": self.key,
                "base_url": self.config.base_url.rstrip("/"),
                "prompt_sha256": prompt_hash,
                "input_sha256": input_hash,
                "sample": self.sample,
                "request": {k: v for k, v in payload.items() if k != "messages"},
                "response": self._post(payload),
            }
            if self.cache:
                self.cache.put(self.key, prompt_hash, input_hash, record)
        usage = record["response"].get("usage")
        self.last_usage = usage if isinstance(usage, dict) else None
        return _answer_text(record["response"], self.key)

    def _post(self, payload: dict) -> dict:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json",
                   "User-Agent": "nse-genbench"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        attempts = self.config.max_retries + 1
        last_error = ""
        for attempt in range(attempts):
            retry_after = None
            try:
                status, text, resp_headers = self._transport(self.url, body, headers, self.config.timeout_s)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last_error = f"{type(exc).__name__}: {getattr(exc, 'reason', exc)}"
            else:
                if 200 <= status < 300:
                    try:
                        data = json.loads(text)
                    except json.JSONDecodeError as exc:
                        raise ModelRequestError(
                            f"{self.url} returned non-JSON (HTTP {status}): {text[:300]!r}") from exc
                    if not isinstance(data, dict):
                        raise ModelRequestError(f"{self.url} returned JSON that is not an object")
                    return data
                last_error = f"HTTP {status}: {text[:300]}"
                if status not in RETRY_STATUSES:
                    hint = " (check MODEL_API_KEY)" if status in (401, 403) else ""
                    raise ModelRequestError(f"{self.url} returned {last_error}{hint}")
                retry_after = _header(resp_headers, "Retry-After")
            if attempt + 1 < attempts:
                self._sleep(_backoff_seconds(attempt, retry_after))
        raise ModelRequestError(f"{self.url} failed after {attempts} attempt(s); last error: {last_error}")
