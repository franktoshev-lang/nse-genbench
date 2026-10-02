#!/usr/bin/env python3
"""End-to-end smoke test with NO model: corpus -> mock card -> scores.

If this prints a table, the measurement apparatus works. Everything after
this point is swapping MockRunner for a real one.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from genbench.corpus import load_configs, build_slice, write_slice
from genbench.metrics import score, stability
from genbench.runners import MockRunner, MOCK_CARDS
from genbench.report import table

def main(slice_id="S01"):
    themes, slices = load_configs()
    spec = next(s for s in slices["slices"] if s["id"] == slice_id)
    sl = build_slice(spec, themes, seed=spec.get("seed", 42))
    write_slice(sl, "data/synthetic")
    truth = sl.as_dict()

    rows = []
    for variant in MOCK_CARDS:
        card = json.loads(MockRunner(variant).generate(prompt="<not used by mock>"))
        sc = score(card, truth, themes)
        row = sc.flat(); row["model"] = f"mock:{variant}"
        rows.append(row)

    cols = ["model","slice","valid_json","coverage","minority_preserved",
            "hallucinated_absent","evidence_validity","evidence_support",
            "max_leakage","lang","words"]
    print(f"\nSlice {sl.id} — {sl.label}")
    print(f"  n={len(sl.responses)}  planted={sl.planted}")
    print(f"  minority={sl.minority_theme!r} ({spec['minority']['count']} responses)")
    print(f"  absent={sl.absent_theme!r}\n")
    print(table(rows, cols))

    same = [MOCK_CARDS["good"]] * 3
    mixed = [MOCK_CARDS["good"], MOCK_CARDS["bad"], MOCK_CARDS["leaky"]]
    print("\nstability (identical runs):", stability(same))
    print("stability (differing runs):", stability(mixed))

if __name__ == "__main__":
    main(*sys.argv[1:])
