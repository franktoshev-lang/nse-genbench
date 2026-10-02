import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from genbench.corpus import load_configs, build_slice

@pytest.fixture(scope="session")
def cfg():
    return load_configs()

@pytest.fixture(scope="session")
def s01(cfg):
    themes, slices = cfg
    spec = next(s for s in slices["slices"] if s["id"] == "S01")
    return build_slice(spec, themes, seed=42).as_dict()
