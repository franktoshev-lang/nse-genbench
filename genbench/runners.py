"""Model runners.

MockRunner needs no GPU and no network. It exists so the whole pipeline can
be exercised end to end TODAY, and so the metrics can be unit-tested against
cards whose correct scores are known in advance.

LlamaCppRunner is a stub: the HTTP shape is written out but it is not wired
up, because hardware and gateway access are still open questions.
"""
from __future__ import annotations

import json
from typing import Protocol


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


class LlamaCppRunner:
    """STUB — not wired up. Blocked on hardware confirmation (open question 3).

    The payload shape mirrors llama_cpp_models.chat_completion_payload in the
    NSE repo. Note the three guards that must be set before any real run:
      * max_tokens raised (repo pins 2048 -> truncation risk)
      * temperature chosen deliberately (repo overrides its own 0.1 to 1.0)
      * no web search / tools
    """

    def __init__(self, model_id: str, base_url: str, api_key: str = "no-key",
                 temperature: float = 0.1, max_tokens: int = 8192,
                 enable_thinking: bool = False) -> None:
        self.key = model_id
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking

    def payload(self, prompt: str) -> dict:
        return {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }

    def generate(self, prompt: str) -> str:
        raise NotImplementedError(
            "LlamaCppRunner is not wired up yet. Blocked on: GPU confirmation, "
            "and the D-1 fix (hardcoded 'Gemma 4' in the prompt) before any "
            "cross-model run. See nse-prompt-limitations.md."
        )
