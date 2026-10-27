"""The settings writers change only what they are asked to, and keep comments."""
import shutil
from pathlib import Path

import pytest

from genbench.config_io import update_dotenv, update_model_field
from genbench.runners import load_model_settings, read_dotenv

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def models_file(tmp_path):
    path = tmp_path / "models.yml"
    shutil.copy(ROOT / "config" / "models.yml", path)
    return path


def test_dotenv_keys_are_replaced_and_other_lines_kept(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# my notes\nMODEL_BASE_URL=old\nGENBENCH_SEED=7\n", encoding="utf-8")
    update_dotenv(env, {"MODEL_BASE_URL": "https://new/v1", "MODEL_API_KEY": "k"})
    text = env.read_text(encoding="utf-8")
    assert text.startswith("# my notes\n")
    assert read_dotenv(env) == {"MODEL_BASE_URL": "https://new/v1", "GENBENCH_SEED": "7", "MODEL_API_KEY": "k"}

def test_missing_dotenv_starts_from_the_example(tmp_path):
    example = tmp_path / ".env.example"
    example.write_text("# example\nMODEL_BASE_URL=\nGENBENCH_SEED=42\n", encoding="utf-8")
    update_dotenv(tmp_path / ".env", {"MODEL_BASE_URL": "https://x/v1"}, example=example)
    assert read_dotenv(tmp_path / ".env") == {"MODEL_BASE_URL": "https://x/v1", "GENBENCH_SEED": "42"}

def test_dotenv_rejects_odd_keys_and_multiline_values(tmp_path):
    with pytest.raises(ValueError):
        update_dotenv(tmp_path / ".env", {"bad key": "x"})
    with pytest.raises(ValueError):
        update_dotenv(tmp_path / ".env", {"MODEL_API_KEY": "a\nb"})

def test_model_field_changes_in_place_and_comments_survive(models_file):
    update_model_field("gemma-4-12b", "model_id", "google/gemma-4-12b-it", models_file)
    update_model_field("gpt-6-astra", "price_in_per_million", "2.5", models_file)
    models = load_model_settings(models_file)
    assert models["gemma-4-12b"]["model_id"] == "google/gemma-4-12b-it"
    assert models["gpt-6-astra"]["price_in_per_million"] == 2.5
    assert models["gemma-4-31b"]["model_id"] == ""
    text = models_file.read_text(encoding="utf-8")
    assert "# no standard id; take it from the university" in text
    assert "# OpenAI's docs (checked 19 October 2026)" in text

def test_empty_price_becomes_null(models_file):
    update_model_field("gpt-6-astra", "price_in_per_million", "3", models_file)
    update_model_field("gpt-6-astra", "price_in_per_million", "", models_file)
    assert load_model_settings(models_file)["gpt-6-astra"]["price_in_per_million"] is None

@pytest.mark.parametrize("key, field, value, message", [
    ("nope", "model_id", "x", "unknown model"),
    ("gemma-4-12b", "temperature", "false", "cannot be changed"),
    ("gemma-4-12b", "price_in_per_million", "-1", "cannot be negative"),
])
def test_bad_model_updates_are_rejected_and_the_file_is_untouched(models_file, key, field, value, message):
    before = models_file.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        update_model_field(key, field, value, models_file)
    assert models_file.read_text(encoding="utf-8") == before
