"""Reviewer tests: optimizer math, failures and saved artifacts beyond smoke tests."""

import copy
import json
import math
import random
from pathlib import Path

import pytest
import torch

from src.data import format_prompt
from src.losses import compute_sft_loss, compute_dpo_loss
from src.training import default_config, train, evaluate, _checkpoint
from tests.toy_lm import ToyCausalLM, ToyTokenizer as PublicToyTokenizer


class ToyTokenizer(PublicToyTokenizer):
    # Test-only extension for the required multiline instruction template.
    def __init__(self):
        super().__init__()
        self.vocab['\n'] = len(self.vocab)
        self.inv_vocab = {i: token for token, i in self.vocab.items()}


def rows(kind, n=5, prefix='train'):
    data = []
    for i in range(n):
        row = {'prompt': prefix + str(i)}
        if kind == 'sft':
            row['response'] = 'abc'[:i % 3 + 1] * (i + 1)
        else:
            row.update(chosen='a' * (i + 1), rejected='b' * (i % 3 + 1))
        data.append(row)
    return data


def config(kind):
    c = default_config(kind)
    c.update(batch_size=1, accumulation=2, lr=0.01, warmup_ratio=0,
             schedule='fixed', clip_norm=100, eval_every=10, max_length=80)
    return c


def oracle(model, reference, tokenizer, data, c):
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])
    order = list(range(len(data)))
    random.Random(c['seed']).shuffle(order)
    for start in range(0, len(data), c['batch_size'] * c['accumulation']):
        batch = [data[i] for i in order[start:start + c['batch_size'] * c['accumulation']]]
        optimizer.zero_grad(set_to_none=True)
        prompts = [format_prompt(r['prompt']) for r in batch]
        if c['kind'] == 'sft':
            loss = compute_sft_loss(model, tokenizer, prompts, [r['response'] for r in batch], c['max_length'])
        else:
            loss = compute_dpo_loss(model, reference, tokenizer, prompts, [r['chosen'] for r in batch],
                                    [r['rejected'] for r in batch], c['beta'], c['max_length'])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), c['clip_norm'])
        optimizer.step()


@pytest.mark.parametrize('kind', ['sft', 'dpo'])
def test_real_loop_matches_full_window_oracle_including_remainder(tmp_path, kind):
    tokenizer = ToyTokenizer()
    model = ToyCausalLM(tokenizer.vocab_size, bias_scale=1)
    expected = copy.deepcopy(model)
    reference = copy.deepcopy(model) if kind == 'dpo' else None
    frozen = copy.deepcopy(reference.state_dict()) if reference is not None else None
    c = config(kind)
    data = rows(kind)
    oracle(expected, reference, tokenizer, data, c)
    result = train(model, tokenizer, data, rows(kind, 3, 'val'), c, tmp_path / 'run', reference)
    torch.testing.assert_close(model.logit_bias, expected.logit_bias, rtol=1e-5, atol=1e-7)
    assert result['updates'] == result['attempts'] == 3
    assert result['skipped_updates'] == 0
    records = [json.loads(line) for line in (tmp_path / 'run/metrics.jsonl').read_text().splitlines()]
    updates = [r for r in records if r['event'] == 'train']
    assert [r['examples'] for r in updates] == [2, 2, 1]
    assert records[0]['updates'] == 0 and records[-1]['updates'] == 3
    weights = torch.load(tmp_path / 'run/final/weights.pt', weights_only=True)
    reloaded = ToyCausalLM(tokenizer.vocab_size)
    reloaded.load_state_dict(weights)
    torch.testing.assert_close(reloaded.logit_bias, model.logit_bias)
    assert (tmp_path / 'run/best/state.json').exists()
    assert result['resume_supported'] is False
    if reference is not None:
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(value, frozen[name], rtol=0, atol=0)
        assert all(not p.requires_grad and p.grad is None for p in reference.parameters())
        assert not reference.training
        assert records[0]['reward_accuracy'] == 0
        assert records[0]['reward_margin'] == 0
        assert records[0]['loss'] == pytest.approx(math.log(2), abs=1e-7)


def test_sft_eval_is_token_weighted_and_mode_restored():
    tokenizer = ToyTokenizer()
    model = ToyCausalLM(tokenizer.vocab_size, bias_scale=2)
    data = rows('sft', 3)
    c = config('sft')
    actual = evaluate(model, None, tokenizer, data, c)
    expected = compute_sft_loss(model, tokenizer, [format_prompt(r['prompt']) for r in data],
                                [r['response'] for r in data], c['max_length']).item()
    assert actual['loss'] == pytest.approx(expected, abs=1e-6)
    assert model.training
    model.eval()
    evaluate(model, None, tokenizer, data, c)
    assert not model.training
    with pytest.raises(ValueError):
        evaluate(model, None, tokenizer, [{'prompt': 'P', 'response': ''}], {**c, 'max_length': 1})
    assert not model.training


class SkipOnce:
    """CPU fake scaler tests scheduler bookkeeping without claiming CUDA coverage."""
    def __init__(self):
        self.scale_value = 128
        self.steps = 0
        self.unscaled = False
    def scale(self, value):
        return value
    def unscale_(self, optimizer):
        self.unscaled = True
    def step(self, optimizer):
        assert self.unscaled
        if self.steps:
            optimizer.step()
        self.unscaled = False
    def update(self):
        if not self.steps:
            self.scale_value /= 2
        self.steps += 1
    def get_scale(self):
        return self.scale_value


def test_skipped_update_does_not_advance_schedule_or_pilot_budget(tmp_path):
    tokenizer = ToyTokenizer()
    model = ToyCausalLM(tokenizer.vocab_size)
    c = config('sft')
    c.update(schedule='cosine', max_updates=2, accumulation=1)
    state = train(model, tokenizer, rows('sft', 5), rows('sft', 2, 'val'), c,
                  tmp_path / 'run', scaler=SkipOnce())
    records = [json.loads(line) for line in (tmp_path / 'run/metrics.jsonl').read_text().splitlines()]
    steps = [r for r in records if r['event'] == 'train']
    assert state['updates'] == 2 and state['attempts'] == 3 and state['skipped_updates'] == 1
    assert not steps[0]['optimizer_step']
    assert steps[0]['lr'] == steps[1]['lr'] == c['lr']
    assert steps[2]['lr'] < steps[1]['lr']


def test_checkpoint_save_failure_keeps_existing_best(tmp_path):
    tokenizer = ToyTokenizer()
    model = ToyCausalLM(tokenizer.vocab_size)
    _checkpoint(model, tokenizer, tmp_path, 'best', {'updates': 0})
    before = (tmp_path / 'best/weights.pt').read_bytes()
    def fail(*args, **kwargs):
        raise OSError('disk full')
    model.save_pretrained = fail
    with pytest.raises(OSError, match='disk full'):
        _checkpoint(model, tokenizer, tmp_path, 'best', {'updates': 1})
    assert (tmp_path / 'best/weights.pt').read_bytes() == before
    assert json.loads((tmp_path / 'best/state.json').read_text())['updates'] == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == ['best']


@pytest.mark.parametrize('issue', ['overlap', 'shared', 'half', 'existing', 'amp_cpu', 'bad_config'])
def test_invalid_runs_rejected_before_output(tmp_path, issue):
    tokenizer = ToyTokenizer()
    model = ToyCausalLM(tokenizer.vocab_size)
    reference = copy.deepcopy(model)
    data, val, c = rows('dpo'), rows('dpo', 2, 'val'), config('dpo')
    if issue == 'overlap': val = data[:2]
    if issue == 'shared': reference.logit_bias = model.logit_bias
    if issue == 'half': model.half()
    if issue == 'amp_cpu': c['amp'] = True
    if issue == 'bad_config': c['accumulation'] = True
    out = tmp_path / 'run'
    if issue == 'existing':
        out.mkdir()
        (out / 'sentinel').write_text('keep')
    with pytest.raises((ValueError, FileExistsError)):
        train(model, tokenizer, data, val, c, out, reference)
    if issue == 'existing': assert (out / 'sentinel').read_text() == 'keep'
    else: assert not out.exists()


def test_nonfinite_failure_is_saved_without_final_success(tmp_path):
    tokenizer = ToyTokenizer()
    model = ToyCausalLM(tokenizer.vocab_size)
    model.logit_bias.data[0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        train(model, tokenizer, rows('sft'), rows('sft', 2, 'val'), config('sft'), tmp_path / 'run')
    state = json.loads((tmp_path / 'run/run.json').read_text())
    assert state['status'] == 'failed' and state['updates'] == 0
    assert not (tmp_path / 'run/final').exists()
    assert model.training


@pytest.mark.parametrize('kind', ['sft', 'dpo'])
def test_best_selection_keeps_first_best_on_ties_and_saves_final_separately(tmp_path, monkeypatch, kind):
    import src.training as training
    tokenizer = ToyTokenizer()
    model = ToyCausalLM(tokenizer.vocab_size, bias_scale=1)
    reference = copy.deepcopy(model) if kind == 'dpo' else None
    sequence = iter([2.0, 1.0, 1.5, 1.0] if kind == 'sft' else [0.0, 0.7, 0.6, 0.7])
    def validation(*args):
        value = next(sequence)
        return {'loss': value} if kind == 'sft' else {'loss': 0.7, 'reward_accuracy': value, 'reward_margin': 0.1}
    monkeypatch.setattr(training, 'evaluate', validation)
    c = config(kind); c['eval_every'] = 1
    result = train(model, tokenizer, rows(kind), rows(kind, 2, 'val'), c, tmp_path / 'run', reference)
    best = json.loads((tmp_path / 'run/best/state.json').read_text())
    final = json.loads((tmp_path / 'run/final/state.json').read_text())
    assert best['updates'] == 1 and final['updates'] == result['updates'] == 3
    best_weights = torch.load(tmp_path / 'run/best/weights.pt', weights_only=True)
    final_weights = torch.load(tmp_path / 'run/final/weights.pt', weights_only=True)
    assert not torch.equal(best_weights['logit_bias'], final_weights['logit_bias'])
    assert final['status'] == 'completed' and final['files']['weights.pt']['bytes'] > 0
