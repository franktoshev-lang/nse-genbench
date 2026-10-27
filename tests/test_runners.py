"""Stage 3: the chat-completions client, tested with a fake transport.

No network: every HTTP exchange goes through FakeTransport, and retries
sleep through a recorder instead of the clock.
"""
import json
import urllib.error

import pytest

from genbench.runners import (DEFAULT_MAX_TOKENS, FORBIDDEN_PAYLOAD_KEYS, ChatCompletionsRunner,
                              ModelConfig, ModelNotConfigured, ModelRequestError, ResponseCache,
                              load_model_settings, read_dotenv)

URL = "http://127.0.0.1:8080/v1"


def ok(content='{"summary": "ok"}', finish_reason="stop"):
    return 200, json.dumps({"choices": [{"message": {"role": "assistant", "content": content},
                                         "finish_reason": finish_reason}]}), {}


class FakeTransport:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def __call__(self, url, body, headers, timeout):
        self.calls.append({"url": url, "payload": json.loads(body), "headers": headers, "timeout": timeout})
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make(transport, cache=None, sample=0, **config):
    sleeps = []
    cfg = ModelConfig(base_url=config.pop("base_url", URL), model=config.pop("model", "test-model"), **config)
    runner = ChatCompletionsRunner(cfg, cache=cache, sample=sample, transport=transport, sleep=sleeps.append)
    return runner, sleeps

# ── configuration ──────────────────────────────────────────────────────────
def test_unconfigured_endpoint_errors_clearly(tmp_path):
    with pytest.raises(ModelNotConfigured, match="MODEL_BASE_URL is empty.*mock:good"):
        ModelConfig.from_env(tmp_path / "no.env", environ={})

def test_empty_base_url_in_environment_beats_a_dotenv_value(tmp_path):
    env = tmp_path / ".env"
    env.write_text("MODEL_BASE_URL=https://example.invalid/v1\nMODEL_NAME=m\n", encoding="utf-8")
    with pytest.raises(ModelNotConfigured):
        ModelConfig.from_env(env, environ={"MODEL_BASE_URL": ""})

def test_missing_model_name_errors_clearly(tmp_path):
    with pytest.raises(ModelNotConfigured, match="MODEL_NAME is empty"):
        ModelConfig.from_env(tmp_path / "no.env", environ={"MODEL_BASE_URL": URL})

def test_settings_come_from_dotenv_then_environment_then_overrides(tmp_path):
    env = tmp_path / ".env"
    env.write_text('# comment\nMODEL_BASE_URL=https://a.example/v1/\nMODEL_NAME=from-dotenv\n'
                   'MODEL_API_KEY="secret"\n', encoding="utf-8")
    cfg = ModelConfig.from_env(env, environ={"MODEL_NAME": "from-env"})
    assert (cfg.base_url, cfg.model, cfg.api_key) == ("https://a.example/v1", "from-env", "secret")
    cfg = ModelConfig.from_env(env, environ={}, model="from-cli", temperature=None)
    assert cfg.model == "from-cli" and cfg.temperature == 0.1

def test_defaults_are_explicit_and_not_inherited_from_the_group_registry():
    cfg = ModelConfig(base_url=URL, model="m")
    assert cfg.max_tokens == DEFAULT_MAX_TOKENS == 8192
    assert cfg.temperature == 0.1
    assert cfg.seed == 42

def test_empty_seed_setting_sends_no_seed(tmp_path):
    cfg = ModelConfig.from_env(tmp_path / "no.env", environ={"MODEL_BASE_URL": URL, "MODEL_NAME": "m",
                                                             "GENBENCH_SEED": ""})
    assert cfg.seed is None
    runner, _ = make(FakeTransport(ok()), seed=None)
    assert "seed" not in runner.payload("p")

@pytest.mark.parametrize("bad", [{"temperature": 2.5}, {"max_tokens": 0}, {"model": "x/y:online"},
                                 {"base_url": "localhost:8080"}])
def test_invalid_settings_are_rejected(bad):
    with pytest.raises(ValueError):
        ModelConfig(**{"base_url": URL, "model": "m", **bad})

def test_read_dotenv_handles_quotes_comments_and_missing_files(tmp_path):
    assert read_dotenv(tmp_path / "missing") == {}
    env = tmp_path / ".env"
    env.write_text("# c\n\nexport A=1\nB='two words'\nC=\n", encoding="utf-8")
    assert read_dotenv(env) == {"A": "1", "B": "two words", "C": ""}

# ── request shape ──────────────────────────────────────────────────────────
def test_payload_mirrors_the_group_request_and_never_allows_tools():
    runner, _ = make(FakeTransport(ok()), max_tokens=12000, temperature=0.3)
    payload = runner.payload("PROMPT")
    assert payload["messages"] == [{"role": "user", "content": "PROMPT"}]
    assert (payload["max_tokens"], payload["temperature"], payload["seed"]) == (12000, 0.3, 42)
    assert payload["stream"] is False
    assert not FORBIDDEN_PAYLOAD_KEYS & payload.keys()

def test_request_goes_to_chat_completions_with_auth_only_when_a_key_is_set():
    transport = FakeTransport(ok())
    runner, _ = make(transport, base_url=URL + "/", api_key="k")
    runner.generate("p")
    call = transport.calls[0]
    assert call["url"] == URL + "/chat/completions"
    assert call["headers"]["Authorization"] == "Bearer k"
    transport = FakeTransport(ok())
    make(transport)[0].generate("p")
    assert "Authorization" not in transport.calls[0]["headers"]

# ── retries and errors ─────────────────────────────────────────────────────
def test_retries_server_errors_with_backoff_then_succeeds():
    transport = FakeTransport((503, "busy", {}), (502, "busy", {}), ok("ANSWER"))
    runner, sleeps = make(transport)
    assert runner.generate("p") == "ANSWER"
    assert len(transport.calls) == 3
    assert sleeps == [1.0, 2.0]

def test_retry_after_header_is_honoured():
    runner, sleeps = make(FakeTransport((429, "slow down", {"retry-after": "7"}), ok()))
    runner.generate("p")
    assert sleeps == [7.0]

def test_connection_errors_are_retried():
    transport = FakeTransport(urllib.error.URLError("refused"), ok("ANSWER"))
    assert make(transport)[0].generate("p") == "ANSWER"

def test_client_errors_fail_at_once_with_a_hint():
    transport = FakeTransport((401, "unauthorized", {}))
    runner, sleeps = make(transport)
    with pytest.raises(ModelRequestError, match="HTTP 401.*MODEL_API_KEY"):
        runner.generate("p")
    assert len(transport.calls) == 1 and sleeps == []

def test_gives_up_after_max_retries():
    transport = FakeTransport((500, "down", {}))
    runner, sleeps = make(transport, max_retries=2)
    with pytest.raises(ModelRequestError, match="failed after 3 attempt"):
        runner.generate("p")
    assert len(transport.calls) == 3 and len(sleeps) == 2

def test_non_json_success_is_an_error():
    with pytest.raises(ModelRequestError, match="non-JSON"):
        make(FakeTransport((200, "<html>", {})))[0].generate("p")

def test_truncation_is_reported_and_missing_content_gives_empty_answer(capsys):
    assert make(FakeTransport(ok("{\"a\": ", finish_reason="length")))[0].generate("p") == '{"a": '
    assert "finish_reason=length" in capsys.readouterr().out
    empty = (200, json.dumps({"choices": [{"message": {"content": None}, "finish_reason": "stop"}]}), {})
    assert make(FakeTransport(empty))[0].generate("p") == ""

# ── cache ──────────────────────────────────────────────────────────────────
def test_cache_returns_the_stored_answer_without_calling_again(tmp_path):
    cache = ResponseCache(tmp_path)
    transport = FakeTransport(ok("FIRST"), ok("SECOND"))
    runner, _ = make(transport, cache=cache)
    assert runner.generate("p") == "FIRST" and runner.last_from_cache is False
    assert runner.generate("p") == "FIRST" and runner.last_from_cache is True
    assert len(transport.calls) == 1

def test_cache_key_changes_with_settings_and_repeat_index(tmp_path):
    cache = ResponseCache(tmp_path)
    transport = FakeTransport(ok("A"), ok("B"), ok("C"))
    assert make(transport, cache=cache)[0].generate("p") == "A"
    assert make(transport, cache=cache, temperature=1.0)[0].generate("p") == "B"
    assert make(transport, cache=cache, sample=1)[0].generate("p") == "C"
    assert len(transport.calls) == 3

def test_cache_record_keeps_the_raw_response_but_not_the_prompt(tmp_path):
    cache = ResponseCache(tmp_path)
    runner, _ = make(FakeTransport(ok("ANSWER")), cache=cache)
    runner.generate("SECRET PROMPT TEXT")
    files = list(tmp_path.rglob("*.json"))
    assert len(files) == 1
    text = files[0].read_text(encoding="utf-8")
    record = json.loads(text)
    assert "SECRET PROMPT TEXT" not in text
    assert record["response"]["choices"][0]["message"]["content"] == "ANSWER"
    assert len(record["prompt_sha256"]) == 64 and record["request"]["temperature"] == 0.1

# ── per-model request settings ─────────────────────────────────────────────
def test_a_model_can_be_set_to_never_receive_a_temperature():
    runner, _ = make(FakeTransport(ok()), send_temperature=False)
    assert "temperature" not in runner.payload("p")

def test_the_output_limit_field_name_is_configurable():
    runner, _ = make(FakeTransport(ok()), token_limit_param="max_completion_tokens", max_tokens=9000)
    payload = runner.payload("p")
    assert payload["max_completion_tokens"] == 9000 and "max_tokens" not in payload
    with pytest.raises(ValueError, match="token_limit_param"):
        ModelConfig(base_url=URL, model="m", token_limit_param="limit")

def test_request_shape_is_part_of_the_cache_key(tmp_path):
    cache = ResponseCache(tmp_path)
    transport = FakeTransport(ok("A"), ok("B"), ok("C"))
    assert make(transport, cache=cache)[0].generate("p") == "A"
    assert make(transport, cache=cache, send_temperature=False)[0].generate("p") == "B"
    assert make(transport, cache=cache, token_limit_param="max_completion_tokens")[0].generate("p") == "C"

def test_token_usage_is_kept_from_fresh_and_cached_answers(tmp_path):
    usage = {"prompt_tokens": 120, "completion_tokens": 30,
             "completion_tokens_details": {"reasoning_tokens": 10}}
    body = json.dumps({"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
                       "usage": usage})
    cache = ResponseCache(tmp_path)
    runner, _ = make(FakeTransport((200, body, {})), cache=cache)
    runner.generate("p")
    assert runner.last_usage == usage
    again, _ = make(FakeTransport((500, "must not be called", {})), cache=cache)
    again.generate("p")
    assert again.last_from_cache and again.last_usage == usage

def test_models_config_holds_the_four_models_under_test():
    models = load_model_settings()
    assert list(models) == ["gemma-4-12b", "gemma-4-31b", "claude-opus-5-5", "gpt-6-astra"]
    assert models["gemma-4-12b"]["role"].startswith("baseline")
    assert models["gpt-6-astra"]["temperature"] is False

def test_bad_models_config_is_rejected(tmp_path):
    path = tmp_path / "models.yml"
    path.write_text("models:\n  m:\n    token_limit_param: limit\n", encoding="utf-8")
    with pytest.raises(ValueError, match="token_limit_param"):
        load_model_settings(path)
