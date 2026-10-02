import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest
from genbench.data import load_responses, load_truth
from genbench.metrics import load_probes

# Test fixture, not benchmark data: the smallest set of responses the three
# mock cards need. The cards cite E0001-E0003, and the leaky card copies
# E0002 and E0003 word for word. Written to temp files so the real loaders
# are exercised.
FIXTURE_RESPONSES = [
    {"evidence_id": "E0001", "document": "De docenten zijn erg betrokken maar reageren traag op mail.",
     "evidence_type": "definite"},
    {"evidence_id": "E0002", "document": "Beoordelingscriteria voor Projectmanagement zijn onduidelijk; "
                                         "je weet niet waar je op beoordeeld wordt.",
     "evidence_type": "definite"},
    {"evidence_id": "E0003", "document": "Feedback is vaak alleen een cijfer zonder uitleg, daar leer ik niets van.",
     "evidence_type": "definite"},
    {"evidence_id": "E0004", "document": "Het rooster verandert te vaak, soms pas een dag vooraf.",
     "evidence_type": "definite"},
    {"evidence_id": "E0005", "document": "Er is weinig ondersteuning als je achterloopt.",
     "evidence_type": "ambiguous"},
    {"evidence_id": "E0006", "document": "Prima", "evidence_type": "definite"},
]
FIXTURE_TRUTH = {
    "id": "fixture",
    "theme": "Examination & Assessment",
    "planted": {"Examination & Assessment": 2, "Teachers": 1,
                "Content and Organisation": 1, "Support / Mentoring": 1},
    "minority_theme": "Support / Mentoring",
    "absent_theme": "Professional Practice",
    "labels": {"E0001": "Teachers", "E0002": "Examination & Assessment",
               "E0003": "Examination & Assessment", "E0004": "Content and Organisation",
               "E0005": "Support / Mentoring", "E0006": None},
}


def write_jsonl(path: Path, records: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                    encoding="utf-8", newline="\n")
    return path


def write_json(path: Path, obj) -> Path:
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def fixture_paths(tmp_path_factory):
    d = tmp_path_factory.mktemp("fixture")
    return (write_jsonl(d / "responses.jsonl", FIXTURE_RESPONSES),
            write_json(d / "truth.json", FIXTURE_TRUTH))


@pytest.fixture(scope="session")
def responses(fixture_paths):
    return load_responses(fixture_paths[0])


@pytest.fixture(scope="session")
def truth(fixture_paths):
    return load_truth(fixture_paths[1])


@pytest.fixture(scope="session")
def probes():
    return load_probes(ROOT / "config" / "probes.yml")
