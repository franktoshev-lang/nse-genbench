"""Small, careful writers for the two settings files the UI can change.

.env              only the keys given are replaced or added; other lines stay
config/models.yml one field of one model is changed in place, so the
                  comments in the file survive; the result must still load
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .runners import MODELS_CONFIG, load_models_config

EDITABLE_MODEL_FIELDS = ("model_id", "price_in_per_million", "price_out_per_million")


def update_dotenv(path, updates: dict[str, str], example=None) -> None:
    """Set KEY=value lines in .env. A missing .env starts from .env.example."""
    p = Path(path)
    if p.is_file():
        lines = p.read_text(encoding="utf-8-sig").splitlines()
    elif example is not None and Path(example).is_file():
        lines = Path(example).read_text(encoding="utf-8").splitlines()
    else:
        lines = []
    for key, value in updates.items():
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            raise ValueError(f"not a settings key: {key!r}")
        if "\n" in value or "\r" in value:
            raise ValueError(f"{key}: value must be one line")
        new_line = f"{key}={value}"
        for i, line in enumerate(lines):
            if re.match(rf"\s*(export\s+)?{key}\s*=", line):
                lines[i] = new_line
                break
        else:
            lines.append(new_line)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _yaml_value(field: str, value) -> str:
    if field == "model_id":
        return json.dumps(str(value or "").strip())
    if value is None or value == "":
        return "null"
    number = float(value)
    if number < 0:
        raise ValueError(f"{field} cannot be negative")
    return repr(number)


def update_model_field(key: str, field: str, value, path=MODELS_CONFIG) -> None:
    if field not in EDITABLE_MODEL_FIELDS:
        raise ValueError(f"{field!r} cannot be changed here")
    p = Path(path)
    original = p.read_text(encoding="utf-8")
    lines = original.splitlines()
    try:
        start = lines.index(f"  {key}:")
    except ValueError:
        raise ValueError(f"unknown model {key!r}") from None
    end = next((i for i in range(start + 1, len(lines)) if re.match(r"  \S", lines[i])), len(lines))
    pattern = re.compile(rf"^(    {field}:\s*)([^#]*?)(\s*#.*)?$")
    for i in range(start + 1, end):
        m = pattern.match(lines[i])
        if m:
            lines[i] = m.group(1) + _yaml_value(field, value) + (m.group(3) or "")
            break
    else:
        lines.insert(end, f"    {field}: {_yaml_value(field, value)}")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    try:
        load_models_config(p)
    except Exception:
        p.write_text(original, encoding="utf-8", newline="\n")
        raise
