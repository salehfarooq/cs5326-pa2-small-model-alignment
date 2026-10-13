"""Greedy generation boundaries, mode restoration and repetition diagnostics."""
from types import SimpleNamespace

import pytest
import torch

from src.generation import generate_development, repetition_stats
from tests.test_review_training import ToyTokenizer
from tests.toy_lm import ToyCausalLM


class GenerateToy(ToyCausalLM):
    def __init__(self, tokenizer, result='ababab'):
        super().__init__(tokenizer.vocab_size)
        self.tokenizer = tokenizer
        self.result = result
        self.settings = None
        self.config = SimpleNamespace(max_position_embeddings=1000)
    def generate(self, input_ids, attention_mask, **kwargs):
        assert not self.training and not torch.is_grad_enabled()
        self.settings = kwargs
        new = self.tokenizer.encode(self.result) + [self.tokenizer.eos_token_id]
        return torch.cat([input_ids, torch.tensor([new], device=input_ids.device)], dim=1)


class DecodeToy(ToyTokenizer):
    def decode(self, ids, skip_special_tokens=True):
        return ''.join(self.inv_vocab[i] for i in ids if i > 1)


def test_new_tokens_only_eos_mode_and_shared_settings():
    t = DecodeToy(); m = GenerateToy(t)
    out = generate_development(m, t, ['Question'], repetition_penalty=1.05, no_repeat_ngram_size=3)[0]
    assert out['output'] == 'ababab' and out['response_tokens'] == 6 and out['finish_reason'] == 'eos'
    assert not out['empty_output'] and out['repeated_ngrams'] == 2
    assert m.training
    assert m.settings['do_sample'] is False and m.settings['use_cache'] is True
    assert out['settings']['repetition_penalty'] == 1.05
    m.eval(); generate_development(m, t, ['Question']); assert not m.training


def test_empty_output_is_diagnostic_not_hidden():
    t = DecodeToy(); m = GenerateToy(t, '')
    out = generate_development(m, t, ['Question'])[0]
    assert out['empty_output'] and out['response_tokens'] == 0


def test_long_full_prompt_is_preserved_or_explicitly_rejected():
    t = DecodeToy(); m = GenerateToy(t)
    out = generate_development(m, t, ['a' * 300])[0]
    assert out['prompt_tokens'] > 256
    m.config.max_position_embeddings = 320
    with pytest.raises(ValueError, match='context'):
        generate_development(m, t, ['a' * 300])
    assert m.training


def test_bad_prefix_restores_mode():
    t = DecodeToy(); m = GenerateToy(t)
    m.generate = lambda **kwargs: torch.tensor([[t.vocab['a']]])
    with pytest.raises(ValueError, match='prefix'):
        generate_development(m, t, ['Question'])
    assert m.training


@pytest.mark.parametrize('kwargs', [{'max_new_tokens': True}, {'repetition_penalty': float('nan')},
                                  {'repetition_penalty': 0.9}, {'no_repeat_ngram_size': -1}])
def test_invalid_settings_rejected(kwargs):
    t = DecodeToy()
    with pytest.raises(ValueError): generate_development(GenerateToy(t), t, ['Question'], **kwargs)


def test_short_text_has_zero_repetition():
    assert repetition_stats([1, 2])['repeated_ngram_fraction'] == 0
