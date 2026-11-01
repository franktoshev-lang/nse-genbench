"""The scoring checklist: the yes/no checks a person answers for each card.

The same list is used by the researcher and, later, by the quality officers.
Every check is a yes/no question, because yes/no judgements on small units
are more consistent between scorers than one 1-5 rating for a whole card
(LongEval, CheckEval). The checks sit under the three headings fixed in the
scoring research document: trustworthy, complete, useful.

Item keys, as stored in out/scores.jsonl under "checks":
  T3, C2, U2, U3, U4     one per card
  C1                     one per card, when the slice has no answer key
  C1:<subtheme>          one per planted subtheme in the answer key
  T1:<n>                 one per subtheme the card lists (n = its position)
  T2:pos:<n> T2:crit:<n> one per positive / critical point
  U1:pos:<n> U1:crit:<n> one per positive / critical point
  T4:<n>                 one per student suggestion

A heading's score is the mean, over its checks, of each check's share of
"yes" answers, so every check counts equally however many items it has.
Unanswered items are left out (for example U4, which only officers answer).
"""
from __future__ import annotations

import re
import statistics

RUBRIC_VERSION = "2026-11-01"
HEADINGS = ("trustworthy", "complete", "useful")

CHECKS: dict[str, dict] = {
    "T1": {"heading": "trustworthy", "per": "subtheme",
           "question": "Do the cited answers really say what this subtheme claims? Look at up to 3.",
           "hint": "Yes if the answers you look at are about this subtheme. Use the answer key next to each answer."},
    "T2": {"heading": "trustworthy", "per": "point",
           "question": "Can this point be traced back to the student answers?",
           "hint": "No if nothing in the answers says this, or it adds a cause or detail the answers do not give."},
    "T3": {"heading": "trustworthy", "per": "card",
           "question": "Does the summary lead with the biggest group, without overstating how many students said something?",
           "hint": "No if it says 'most students' about a small group, or opens with a minor point."},
    "T4": {"heading": "trustworthy", "per": "suggestion",
           "question": "Is this a word-for-word quote from an answer, and an actual suggestion?",
           "hint": "No if it is reworded, made up, or only a complaint without a proposal."},
    "C1": {"heading": "complete", "per": "planted",
           "question": "Is this main subtheme from the answers in the card?",
           "hint": "Yes if any part of the card covers it, under any name."},
    "C2": {"heading": "complete", "per": "card",
           "question": "Is the small group (the rare subtheme) still in the card?",
           "hint": "On the test slices the answer key names it. A small group dropped from the card counts as no."},
    "U1": {"heading": "useful", "per": "point",
           "question": "Is this point specific rather than generic?",
           "hint": "Yes: 'Feedback on the project report comes 4-6 weeks after hand-in.' No: 'Communication could be better.'"},
    "U2": {"heading": "useful", "per": "card",
           "question": "Do the subthemes stay distinct, without overlapping or repeating each other?",
           "hint": "No if two subthemes are the same issue in other words."},
    "U3": {"heading": "useful", "per": "card",
           "question": "Could a quality officer decide on an action from this card?",
           "hint": "Yes if it is clear what is wrong, for whom and roughly how big, so someone could act on it."},
    "U4": {"heading": "useful", "per": "card", "officers_only": True,
           "question": "Does the Dutch read naturally? (quality officers only)",
           "hint": "Leave empty if you do not read Dutch, or if the card is in English."},
}

ITEM_KEY = re.compile(r"^(T3|C2|U2|U3|U4|C1|C1:.{1,120}|T1:\d{1,3}|(T2|U1):(pos|crit):\d{1,3}|T4:\d{1,3})$")


def check_id(item_key: str) -> str:
    return item_key.split(":", 1)[0]


def valid_key(item_key: str) -> bool:
    return bool(ITEM_KEY.match(str(item_key)))


def checks_from_score(score: dict | None) -> dict[str, bool]:
    """A score's checks; an older score's per-subtheme claim checks count as T1."""
    if not score:
        return {}
    checks = dict(score.get("checks") or {})
    for index, value in (score.get("claims") or {}).items():
        checks.setdefault(f"T1:{index}", value)
    return checks


def heading_scores(checks: dict[str, bool]) -> dict:
    """{"trustworthy": share, "complete": share, "useful": share, "checks_answered": n}."""
    per_check: dict[str, list[bool]] = {}
    for key, value in checks.items():
        if value is None or check_id(key) not in CHECKS:
            continue
        per_check.setdefault(check_id(key), []).append(bool(value))
    out: dict = {}
    for heading in HEADINGS:
        shares = [sum(v) / len(v) for cid, v in per_check.items() if CHECKS[cid]["heading"] == heading]
        out[heading] = round(statistics.mean(shares), 4) if shares else None
    out["checks_answered"] = sum(len(v) for v in per_check.values())
    return out


def rank_key(row: dict) -> tuple:
    """Sort key for the ranking rule fixed in advance: trustworthy first, ties
    broken by complete, then useful. A missing score sorts last."""
    return tuple(-(row[h] if row.get(h) is not None else -1.0) for h in HEADINGS)


def as_state() -> dict:
    """The checklist as the UI needs it."""
    return {"version": RUBRIC_VERSION, "headings": list(HEADINGS),
            "checks": [{"id": cid, **spec} for cid, spec in CHECKS.items()]}
