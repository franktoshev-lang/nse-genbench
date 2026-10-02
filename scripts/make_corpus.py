#!/usr/bin/env python3
"""Generate every slice in config/slices.yml."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from genbench.corpus import load_configs, build_slice, write_slice

def main(out_dir="data/synthetic"):
    themes, slices = load_configs()
    seed = slices.get("defaults", {}).get("seed", 42)
    for spec in slices["slices"]:
        sl = build_slice(spec, themes, seed=spec.get("seed", seed))
        jsonl, truth = write_slice(sl, out_dir)
        print(f"{sl.id}  n={len(sl.responses):<4} planted={sl.planted} "
              f"minority={sl.minority_theme!r} absent={sl.absent_theme!r}")
        print(f"      -> {jsonl}")
        print(f"      -> {truth}")

if __name__ == "__main__":
    main(*sys.argv[1:])
