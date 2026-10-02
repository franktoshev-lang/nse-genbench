from genbench.corpus import build_slice

def test_counts_match_the_recipe(cfg, s01):
    themes, slices = cfg
    spec = next(s for s in slices["slices"] if s["id"] == "S01")
    assert s01["n_responses"] == spec["total"] == 60
    assert s01["planted"][spec["dominant"]["theme"]] == spec["dominant"]["count"]
    assert s01["planted"][spec["minority"]["theme"]] == spec["minority"]["count"]

def test_absent_theme_is_truly_absent(s01):
    assert s01["absent_theme"] not in s01["planted"]
    assert all(r["planted_theme"] != s01["absent_theme"] for r in s01["responses"])

def test_ids_are_unique_and_sequential(s01):
    ids = [r["evidence_id"] for r in s01["responses"]]
    assert len(set(ids)) == len(ids)
    assert ids[0] == "E0001"

def test_deterministic_for_a_given_seed(cfg):
    themes, slices = cfg
    spec = next(s for s in slices["slices"] if s["id"] == "S01")
    a = build_slice(spec, themes, seed=7).as_dict()
    b = build_slice(spec, themes, seed=7).as_dict()
    assert a == b

def test_different_seed_changes_order(cfg):
    themes, slices = cfg
    spec = next(s for s in slices["slices"] if s["id"] == "S01")
    a = build_slice(spec, themes, seed=1).as_dict()
    b = build_slice(spec, themes, seed=2).as_dict()
    assert [r["document"] for r in a["responses"]] != [r["document"] for r in b["responses"]]

def test_noise_and_english_present(s01):
    assert sum(r["is_noise"] for r in s01["responses"]) == 6
    assert sum(r["language"] == "en" for r in s01["responses"]) == 4
