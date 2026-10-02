"""Automatic metrics. Every one is computable from (card, slice truth) alone.

No model, no GPU, no humans. That is the point: the measurement apparatus
can be built and validated before any inference runs, by scoring
hand-written cards whose correct scores are known.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any, Iterable

from .schema import validate_card, card_text, cited_ids

WORD = re.compile(r"[a-zA-ZÀ-ÿ0-9]+")

# Function words used for a crude NL/EN discriminator. Deliberately small and
# symmetric. NOTE: insight_metrics.py in the NSE repo filters Dutch text with a
# mostly-English stopword list; this is the corrected, bilingual version.
NL_MARKERS = {"de","het","een","en","van","is","niet","dat","te","voor","met",
              "op","zijn","er","maar","ook","naar","bij","veel","geen","wordt","worden"}
EN_MARKERS = {"the","and","of","is","not","that","to","for","with","on","are",
              "there","but","also","at","many","no","was","were","be","this"}


def _words(text: str) -> list[str]:
    return [w.lower() for w in WORD.findall(text or "")]


def _ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)} if len(tokens) >= n else set()


# ── 1. theme coverage ───────────────────────────────────────────────────────
def theme_coverage(card: dict, truth: dict, themes_cfg: dict) -> dict:
    """Fraction of planted themes the card surfaced, via bilingual probes."""
    probes = {t["name"]: [p.lower() for p in t["probes"]] for t in themes_cfg["themes"]}
    text = card_text(card).lower()
    planted = list(truth["planted"].keys())

    surfaced, missed = [], []
    for theme in planted:
        if any(p in text for p in probes.get(theme, [])):
            surfaced.append(theme)
        else:
            missed.append(theme)

    return {
        "coverage": round(len(surfaced) / len(planted), 4) if planted else None,
        "surfaced": surfaced,
        "missed": missed,
        "n_planted": len(planted),
    }


# ── 2. minority preservation ────────────────────────────────────────────────
def minority_preserved(card: dict, truth: dict, themes_cfg: dict) -> dict:
    probes = {t["name"]: [p.lower() for p in t["probes"]] for t in themes_cfg["themes"]}
    theme = truth["minority_theme"]
    text = card_text(card).lower()
    hit = any(p in text for p in probes.get(theme, []))
    return {
        "minority_theme": theme,
        "minority_n": truth["planted"].get(theme),
        "minority_share": round(truth["planted"].get(theme, 0) / truth["n_responses"], 4),
        "preserved": bool(hit),
    }


# ── 3. hallucination on the absent topic ────────────────────────────────────
def absent_topic_mentions(card: dict, truth: dict, themes_cfg: dict) -> dict:
    """The absent theme appears in NO response. Any mention is invented."""
    probes = {t["name"]: [p.lower() for p in t["probes"]] for t in themes_cfg["themes"]}
    theme = truth["absent_theme"]
    text = card_text(card).lower()
    hits = [p for p in probes.get(theme, []) if p in text]
    return {"absent_theme": theme, "hit_probes": hits, "hallucinated": bool(hits)}


# ── 4 & 5. evidence-ID validity and scope ───────────────────────────────────
def evidence_ids(card: dict, truth: dict) -> dict:
    valid = {r["evidence_id"] for r in truth["responses"]}
    cited = cited_ids(card)
    if not cited:
        return {"n_cited": 0, "validity": None, "invented": [], "n_manifest_entries":
                len(card.get("subtheme_manifest") or [])}
    invented = [c for c in cited if c not in valid]
    return {
        "n_cited": len(cited),
        "n_unique_cited": len(set(cited)),
        "validity": round(1 - len(invented) / len(cited), 4),
        "invented": sorted(set(invented)),
        "n_manifest_entries": len(card.get("subtheme_manifest") or []),
    }


def evidence_support(card: dict, truth: dict, themes_cfg: dict) -> dict:
    """Weak automatic proxy for ALCE citation precision.

    Checks whether each cited response's planted theme is plausibly related to
    the manifest entry that cites it. A real precision figure needs an NLI
    model or a human; this flags the obvious mismatches only.
    """
    by_id = {r["evidence_id"]: r for r in truth["responses"]}
    probes = {t["name"]: [p.lower() for p in t["probes"]] for t in themes_cfg["themes"]}

    checked = mismatched = 0
    details: list[dict] = []
    for entry in card.get("subtheme_manifest") or []:
        if not isinstance(entry, dict):
            continue
        blob = f"{entry.get('name','')} {entry.get('description','')}".lower()
        for cid in entry.get("evidence_ids") or []:
            row = by_id.get(str(cid))
            if row is None or row.get("is_noise"):
                continue
            theme = row.get("planted_theme")
            if theme is None:
                continue
            checked += 1
            if not any(p in blob for p in probes.get(theme, [])):
                mismatched += 1
                details.append({"entry": entry.get("name"), "id": cid, "planted_theme": theme})

    return {
        "n_checked": checked,
        "support_rate": round(1 - mismatched / checked, 4) if checked else None,
        "mismatches": details[:20],
    }


# ── 6. privacy leakage ──────────────────────────────────────────────────────
def leakage(card: dict, truth: dict, n: int = 8) -> dict:
    """Max n-gram overlap between generated prose and any SINGLE response.

    student_suggestions are excluded: they are supposed to be verbatim.
    High overlap elsewhere means a summary is reproducing one student's words.
    """
    gen = _words(card_text(card, include_suggestions=False))
    gen_ng = _ngrams(gen, n)
    if not gen_ng:
        return {"n": n, "max_overlap": 0.0, "worst_id": None, "n_gen_ngrams": 0}

    worst_id, worst = None, 0.0
    for r in truth["responses"]:
        rng_ = _ngrams(_words(r["document"]), n)
        if not rng_:
            continue
        share = len(gen_ng & rng_) / len(rng_)
        if share > worst:
            worst, worst_id = share, r["evidence_id"]
    return {"n": n, "max_overlap": round(worst, 4), "worst_id": worst_id,
            "n_gen_ngrams": len(gen_ng)}


# ── 7. stability across repeat runs ─────────────────────────────────────────
def stability(cards: Iterable[dict]) -> dict:
    """Mean pairwise Jaccard of lowercased subtheme sets."""
    sets = [{str(s).strip().lower() for s in (c.get("subthemes") or [])} for c in cards]
    sets = [s for s in sets if s]
    if len(sets) < 2:
        return {"n_runs": len(sets), "mean_jaccard": None, "pairs": 0}
    scores = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            u = sets[i] | sets[j]
            scores.append(len(sets[i] & sets[j]) / len(u) if u else 1.0)
    return {"n_runs": len(sets), "mean_jaccard": round(sum(scores) / len(scores), 4),
            "min_jaccard": round(min(scores), 4), "pairs": len(scores)}


# ── 8. output language ──────────────────────────────────────────────────────
def output_language(card: dict) -> dict:
    """Which language did the analytical fields actually come back in?

    Relevant to variant P1: the current prompt REQUIRES English output while
    quality officers need Dutch.
    """
    def guess(text: str) -> str:
        w = _words(text)
        if not w:
            return "unknown"
        nl = sum(1 for x in w if x in NL_MARKERS)
        en = sum(1 for x in w if x in EN_MARKERS)
        if nl == en:
            return "unknown"
        return "nl" if nl > en else "en"

    fields: dict[str, str] = {"summary": guess(str(card.get("summary") or ""))}
    for k in ("positive_comments", "critical_comments", "subthemes"):
        fields[k] = guess(" ".join(str(x) for x in (card.get(k) or [])))

    langs = [v for v in fields.values() if v != "unknown"]
    dominant = max(set(langs), key=langs.count) if langs else "unknown"
    return {"per_field": fields, "dominant": dominant,
            "english_share": round(sum(1 for v in langs if v == "en") / len(langs), 4) if langs else None}


# ── 9. structural validity ──────────────────────────────────────────────────
def structure(card: Any) -> dict:
    v = validate_card(card)
    return {"valid": v.ok, "problems": v.problems}


# ── 10. length / verbosity (pair with any quality score) ───────────────────
def verbosity(card: dict) -> dict:
    t = card_text(card)
    return {
        "chars": len(t),
        "words": len(_words(t)),
        "n_positive": len(card.get("positive_comments") or []),
        "n_critical": len(card.get("critical_comments") or []),
        "n_suggestions": len(card.get("student_suggestions") or []),
        "n_subthemes": len(card.get("subthemes") or []),
    }


# ── aggregate ───────────────────────────────────────────────────────────────
@dataclass
class Scorecard:
    slice_id: str
    structure: dict
    coverage: dict
    minority: dict
    absent: dict
    evidence: dict
    support: dict
    leakage: dict
    language: dict
    verbosity: dict

    def to_dict(self) -> dict:
        return asdict(self)

    def flat(self) -> dict:
        """One row, for the results table."""
        return {
            "slice": self.slice_id,
            "valid_json": self.structure["valid"],
            "coverage": self.coverage["coverage"],
            "minority_preserved": self.minority["preserved"],
            "hallucinated_absent": self.absent["hallucinated"],
            "evidence_validity": self.evidence["validity"],
            "evidence_support": self.support["support_rate"],
            "max_leakage": self.leakage["max_overlap"],
            "lang": self.language["dominant"],
            "words": self.verbosity["words"],
        }


def score(card: dict, truth: dict, themes_cfg: dict) -> Scorecard:
    return Scorecard(
        slice_id=truth["id"],
        structure=structure(card),
        coverage=theme_coverage(card, truth, themes_cfg),
        minority=minority_preserved(card, truth, themes_cfg),
        absent=absent_topic_mentions(card, truth, themes_cfg),
        evidence=evidence_ids(card, truth),
        support=evidence_support(card, truth, themes_cfg),
        leakage=leakage(card, truth),
        language=output_language(card),
        verbosity=verbosity(card),
    )
