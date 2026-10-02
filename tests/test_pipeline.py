"""Single-pass vs map-reduce must follow the group pipeline's service.py:
cap at 240, batches of 60, one batch -> single prompt, reduce over the
parsed batch answers tagged with batch_number and source_document_count."""
import json

import pytest

from genbench import prompts
from genbench.data import Response
from genbench.pipeline import AnswerParseError, generate_card, group_default_mode
from genbench.runners import GOOD_CARD


class RecordingRunner:
    key = "recording"

    def __init__(self, answer=None):
        self.answer = json.dumps(GOOD_CARD, ensure_ascii=False) if answer is None else answer
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.answer


def placeholders(n):
    """Placeholder inputs for counting batches; not survey text."""
    return [Response(f"E{i:04d}", f"placeholder {i}", "definite") for i in range(1, n + 1)]


def run(n, mode=None, runner=None):
    runner = runner or RecordingRunner()
    gen = generate_card(runner, "Teachers", placeholders(n), mode=mode, log=lambda _: None)
    return gen, runner.prompts


def test_group_default_mode_switches_above_one_batch():
    assert [group_default_mode(n) for n in (1, 60, 61, 240)] == ["single", "single", "mapreduce", "mapreduce"]

def test_up_to_60_responses_is_one_single_prompt():
    gen, sent = run(60)
    assert gen.mode == "single" and len(sent) == 1
    assert sent[0] == prompts.build_prompt("Teachers", [r.as_dict() for r in placeholders(60)])
    assert gen.card == GOOD_CARD

def test_61_responses_is_two_maps_and_a_reduce():
    gen, sent = run(61)
    assert gen.mode == "mapreduce" and len(sent) == 3
    assert "batch 1 of 2" in sent[0] and "[E0060]" in sent[0] and "[E0061]" not in sent[0]
    assert "batch 2 of 2" in sent[1] and "[E0061]" in sent[1]
    reduce_prompt = sent[2]
    assert "together represent 61 source student responses" in reduce_prompt
    assert '"batch_number": 2' in reduce_prompt and '"source_document_count": 1' in reduce_prompt

def test_more_than_240_responses_are_capped_like_the_group():
    gen, sent = run(300)
    assert (gen.n_supplied, gen.n_analyzed) == (300, 240)
    assert len(sent) == 5                                   # 4 maps + reduce
    assert all("[E0241]" not in p for p in sent)
    assert "together represent 240 source student responses" in sent[-1]

def test_forced_modes_are_honoured():
    gen, sent = run(61, mode="single")
    assert gen.mode == "single" and len(sent) == 1 and "[E0061]" in sent[0]
    gen, sent = run(6, mode="mapreduce")
    assert gen.mode == "mapreduce" and len(sent) == 2 and "batch 1 of 1" in sent[0]

def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        run(6, mode="both")

def test_unparseable_answer_names_the_step():
    with pytest.raises(AnswerParseError, match="map 1/2"):
        run(61, runner=RecordingRunner(answer="no json at all"))

def test_same_input_same_prompts():
    assert run(130)[1] == run(130)[1]
