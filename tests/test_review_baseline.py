"""Baseline identity preservation and interrupted-export evidence."""
import json

import pytest

from scripts.generate_baseline import export, load_bank
from pathlib import Path
from tests.test_review_generation import DecodeToy, GenerateToy


def test_actual_fixed_bank_loader():
    path = Path(__file__).resolve().parents[2] / 'training-data/alpaca_eval/eval.jsonl'
    if not path.exists():
        pytest.skip('full evaluation input supplied separately on Kaggle')
    assert len(load_bank(path)) == 300


def test_export_preserves_ids_order_text_and_source_dataset(tmp_path):
    tokenizer = DecodeToy()
    rows = [{'id': 'b', 'instruction': 'Question', 'dataset': 'oasst'},
            {'id': 'a', 'instruction': 'Other', 'dataset': 'vicuna'}]
    output = tmp_path / 'base'
    state = export(GenerateToy(tokenizer), tokenizer, rows, output)
    saved = json.loads((output / 'outputs.json').read_text())
    assert [r['id'] for r in saved] == ['b', 'a']
    assert saved[0]['instruction'] == 'Question' and saved[0]['output'] == 'ababab'
    assert saved[0]['dataset'] == 'alpaca_eval' and saved[0]['source_dataset'] == 'oasst'
    assert state['status'] == 'completed' and state['completed'] == 2
    assert state['decoding']['repetition_penalty'] == 1.05
    assert state['decoding']['no_repeat_ngram_size'] == 4
    assert len((output / 'partial.jsonl').read_text().splitlines()) == 2
    with pytest.raises(FileExistsError):
        export(GenerateToy(tokenizer), tokenizer, rows, output)


def test_interrupted_export_keeps_completed_rows_without_final_json(tmp_path):
    tokenizer = DecodeToy()
    model = GenerateToy(tokenizer)
    generate = model.generate
    count = 0
    def fail_second(**kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError('interrupted')
        return generate(**kwargs)
    model.generate = fail_second
    output = tmp_path / 'base'
    with pytest.raises(RuntimeError):
        export(model, tokenizer, [{'id': '1', 'instruction': 'One'},
                                  {'id': '2', 'instruction': 'Two'}], output)
    assert not (output / 'outputs.json').exists()
    assert len((output / 'partial.jsonl').read_text().splitlines()) == 1
    state = json.loads((output / 'run.json').read_text())
    assert state['status'] == 'failed' and state['completed'] == 1
    assert model.training
