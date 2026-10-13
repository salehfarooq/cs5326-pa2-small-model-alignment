"""Diagnostics must be observational: modes, RNG and updates remain intact."""
import copy
import json
import random

import pytest
import torch

from src.training import train
from src.training_diagnostics import create_generation_callback
from tests.test_review_generation import DecodeToy, GenerateToy
from tests.test_review_training import config, rows

PROMPTS = [{'id': 'probe-1', 'group': 'code', 'instruction': 'Question'}]
DECODING = {'max_new_tokens': 32, 'repetition_penalty': 1.05, 'no_repeat_ngram_size': 0}


class RandomGenerate(GenerateToy):
    def generate(self, *args, **kwargs):
        random.random()
        torch.rand(5)
        return super().generate(*args, **kwargs)


def test_rng_and_mixed_module_modes_restored(tmp_path):
    t = DecodeToy()
    m = RandomGenerate(t)
    m.child = torch.nn.Dropout()
    m.train()
    m.child.eval()
    m.config.use_cache = False
    random.seed(17)
    torch.manual_seed(19)
    py_rng, tensor_rng = random.getstate(), torch.get_rng_state().clone()
    callback = create_generation_callback(PROMPTS, {0}, DECODING)
    callback(m, t, 0, {}, tmp_path)
    assert random.getstate() == py_rng
    assert torch.equal(torch.get_rng_state(), tensor_rng)
    assert m.training and not m.child.training
    assert m.config.use_cache is False


def test_no_callback_update_is_not_written(tmp_path):
    t = DecodeToy()
    callback = create_generation_callback(PROMPTS, {0, 2}, DECODING)
    callback(GenerateToy(t), t, 1, {}, tmp_path)
    assert not list(tmp_path.rglob('update-1*'))


def test_callback_does_not_change_actual_optimizer_updates(tmp_path):
    t = DecodeToy()
    m = RandomGenerate(t)
    baseline = copy.deepcopy(m)
    c = config('sft')
    c.update(eval_every=1)
    train(baseline, t, rows('sft'), rows('sft', 3, 'val'), c, tmp_path/'plain')
    hook = create_generation_callback(PROMPTS, {0, 1, 2, 3}, DECODING)
    result = train(m, t, rows('sft'), rows('sft', 3, 'val'), c,
                   tmp_path/'observed', validation_callback=hook)
    assert result['updates'] == 3
    for name, expected in baseline.state_dict().items():
        torch.testing.assert_close(m.state_dict()[name], expected, rtol=0, atol=0)


def test_callback_failure_persists_failed_training(tmp_path):
    t = DecodeToy()
    c = config('sft')
    def fail(*args):
        raise RuntimeError('independent diagnostic failure')
    with pytest.raises(RuntimeError, match='diagnostic failure'):
        train(GenerateToy(t), t, rows('sft'), rows('sft', 3, 'val'), c,
              tmp_path/'failed', validation_callback=fail)
    state = json.loads((tmp_path/'failed/run.json').read_text())
    assert state['status'] == 'failed'
    assert not (tmp_path/'failed/final').exists()


def test_reference_mutation_is_rejected(tmp_path):
    t = DecodeToy()
    m = GenerateToy(t)
    ref = copy.deepcopy(m).eval().requires_grad_(False)
    callback = create_generation_callback(PROMPTS, {0, 2}, DECODING, reference=ref)
    callback(m, t, 0, {}, tmp_path)
    with torch.no_grad():
        next(ref.parameters()).add_(0.125)
    with pytest.raises((ValueError, RuntimeError), match='reference|fingerprint|changed'):
        callback(m, t, 2, {}, tmp_path)


def test_empty_output_fails_after_preserving_evidence(tmp_path):
    t = DecodeToy()
    m = GenerateToy(t, '')
    callback = create_generation_callback(PROMPTS, {0}, DECODING)
    with pytest.raises((ValueError, RuntimeError), match='empty|blank'):
        callback(m, t, 0, {}, tmp_path)
    assert list(tmp_path.rglob('*.json*'))


def test_generation_failure_restores_rng_and_mode(tmp_path):
    t = DecodeToy()
    m = RandomGenerate(t)
    def broken(**kwargs):
        random.random()
        torch.rand(3)
        raise RuntimeError('generation interrupted')
    m.generate = broken
    before_py, before_torch = random.getstate(), torch.get_rng_state().clone()
    hook = create_generation_callback(PROMPTS, {0}, DECODING)
    with pytest.raises(RuntimeError, match='interrupted'):
        hook(m, t, 0, {}, tmp_path)
    assert m.training
    assert random.getstate() == before_py
    assert torch.equal(torch.get_rng_state(), before_torch)


def test_mid_generation_interruption_preserves_finished_rows(tmp_path):
    t = DecodeToy()
    m = GenerateToy(t)
    normal = m.generate
    calls = []
    def interrupted(**kwargs):
        calls.append(1)
        if len(calls) == 3:
            raise KeyboardInterrupt('third prompt')
        return normal(**kwargs)
    m.generate = interrupted
    prompts = [{'id': f'p{i}', 'group':'code','instruction':f'Question {i}'} for i in range(4)]
    hook = create_generation_callback(prompts,{0},DECODING)
    with pytest.raises(KeyboardInterrupt): hook(m,t,0,{},tmp_path)
    saved = [json.loads(line) for line in (tmp_path/'development/update-0-partial.jsonl').read_text().splitlines()]
    assert [r['id'] for r in saved] == ['p0','p1']
    assert json.loads((tmp_path/'development/update-0.json').read_text())['status'] == 'failed'
    assert m.training and not hasattr(m.config,'use_cache')


def test_caller_cannot_mutate_locked_settings(tmp_path):
    prompts = copy.deepcopy(PROMPTS)
    updates = {0}
    decoding = dict(DECODING)
    hook = create_generation_callback(prompts,updates,decoding)
    prompts[0]['instruction'] = 'tampered'
    updates.clear()
    decoding['repetition_penalty'] = 2.0
    t = DecodeToy()
    hook(GenerateToy(t),t,0,{},tmp_path)
    state = json.loads((tmp_path/'development/update-0.json').read_text())
    assert state['decoding'] == DECODING
    assert state['rows'][0]['instruction'] == 'Question'


@pytest.mark.parametrize('defect',['weight','requires_grad','grad'])
def test_reference_checked_on_validation_without_generation(tmp_path,defect):
    t = DecodeToy()
    m = GenerateToy(t)
    ref = copy.deepcopy(m).eval().requires_grad_(False)
    hook = create_generation_callback(PROMPTS,{0},DECODING,reference=ref)
    p = next(ref.parameters())
    if defect == 'weight':
        with torch.no_grad(): p.add_(1)
    if defect == 'requires_grad': p.requires_grad_(True)
    if defect == 'grad': p.grad = torch.ones_like(p)
    with pytest.raises((ValueError,RuntimeError),match='reference|fingerprint'):
        hook(m,t,1,{},tmp_path)


def test_keyboard_interrupt_marks_training_failed(tmp_path):
    t = DecodeToy()
    def stop(*args): raise KeyboardInterrupt('diagnostic interrupt')
    with pytest.raises(KeyboardInterrupt):
        train(GenerateToy(t),t,rows('sft'),rows('sft',3,'val'),config('sft'),
              tmp_path/'interrupted',validation_callback=stop)
    assert json.loads((tmp_path/'interrupted/run.json').read_text())['status'] in ('failed','interrupted')


def test_reference_mutation_during_generation_fails_immediately(tmp_path):
    t = DecodeToy()
    m = GenerateToy(t)
    ref = copy.deepcopy(m).eval().requires_grad_(False)
    normal = m.generate
    def corrupt(**kwargs):
        with torch.no_grad(): next(ref.parameters()).add_(1)
        return normal(**kwargs)
    m.generate = corrupt
    hook = create_generation_callback(PROMPTS,{0},DECODING,reference=ref)
    with pytest.raises((ValueError,RuntimeError),match='reference|fingerprint'):
        hook(m,t,0,{},tmp_path)
    state = json.loads((tmp_path/'development/update-0.json').read_text())
    assert state['status'] == 'failed' and state['reference_fingerprint_mismatch'] is True


def test_numpy_rng_restored(tmp_path):
    import numpy as np
    t = DecodeToy()
    m = GenerateToy(t)
    normal = m.generate
    def consume(**kwargs):
        np.random.random(9)
        return normal(**kwargs)
    m.generate = consume
    np.random.seed(73)
    before = np.random.get_state()
    hook = create_generation_callback(PROMPTS,{0},DECODING)
    hook(m,t,0,{},tmp_path)
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1],after[1])
    assert before[2:] == after[2:]
