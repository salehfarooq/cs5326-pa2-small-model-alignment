"""Independent masking-boundary and frozen-bank checks for preference audit."""
import json
from collections import Counter
from pathlib import Path

import pytest

from scripts.dpo_preflight import inspect_pair, validate_development
from src.data import format_prompt


class Tokenizer:
    eos_token_id = 151643
    def __init__(self, chosen, rejected):
        self.tokens = {format_prompt('question'): list(range(12)), 'chosen': chosen, 'rejected': rejected}
    def encode(self, text, **kwargs):
        return self.tokens[text]


def test_eos_is_added_before_response_budget_boundary():
    row = {'id': 'x', 'prompt': 'question', 'chosen': 'chosen', 'rejected': 'rejected'}
    result = inspect_pair(row, 'train', Tokenizer([4]*499, [5]*500))
    assert result['prompt_tokens'] == 12
    assert result['responses']['chosen']['retained_tokens'] == 500
    assert result['responses']['chosen']['eos_retained']
    assert not result['responses']['chosen']['truncated']
    assert result['responses']['rejected']['retained_tokens'] == 500
    assert not result['responses']['rejected']['eos_retained']
    assert result['responses']['rejected']['truncated']
    assert result['row'] == row


def test_distinction_only_beyond_truncation_is_rejected():
    row = {'prompt': 'question', 'chosen': 'chosen', 'rejected': 'rejected'}
    with pytest.raises(ValueError, match='indistinguishable'):
        inspect_pair(row, 'validation', Tokenizer([4]*500+[7], [4]*500+[8]))


def test_empty_tokenized_answer_is_rejected():
    with pytest.raises(ValueError, match='scoreable'):
        inspect_pair({'prompt':'question','chosen':'chosen','rejected':'rejected'}, 'train', Tokenizer([], [4]))


def valid_bank():
    original = json.loads((Path(__file__).resolve().parents[1]/'data/development/dpo_quality_40.json').read_text())[:15]
    counts = Counter(r['group'] for r in original)
    bank = list(original)
    for group in ['code', 'reasoning', 'safety', 'formatting', 'extraction']:
        for i in range(8-counts[group]):
            bank.append({'id':f'dev-{len(bank)+1}', 'group':group,'instruction':f'Independent {group} probe {i}'})
    return bank


def test_all_original_instructions_and_group_quotas_are_locked():
    bank = valid_bank()
    validate_development(bank)
    bank[0]['instruction'] += ' changed'
    with pytest.raises(ValueError, match='original'):
        validate_development(bank)


def test_canonical_duplicate_and_id_order_are_rejected():
    bank = valid_bank()
    bank[-1]['instruction'] = '\n'+bank[-2]['instruction']+'\n'
    with pytest.raises(ValueError, match='distinct'):
        validate_development(bank)
    bank = valid_bank()
    bank[-1]['id'] = bank[-2]['id']
    with pytest.raises(ValueError, match='order'):
        validate_development(bank)
