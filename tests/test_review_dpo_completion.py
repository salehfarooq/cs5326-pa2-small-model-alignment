"""Incomplete terminal evidence must not pass the experiment completion gate."""
import json

import pytest

from scripts.train_qwen import _verify_finished_diagnostics


def evidence(tmp_path, terminal=200, budget=200):
    prompts = [{'id': 'dev-1', 'instruction': 'Copy exactly: aa aa', 'group': 'code'}]
    decode = {'max_new_tokens': 256, 'repetition_penalty': 1.05, 'no_repeat_ngram_size': 0}
    config = {'updates': {0, 50, 100, 200}, 'prompts': prompts, 'decoding': decode}
    state = {'updates': terminal, 'config': {'max_updates': budget}}
    folder = tmp_path / 'development'
    folder.mkdir()
    for update in config['updates']:
        record = {'status': 'completed', 'update': update, 'decoding': decode,
                  'rows': [{**p, 'output': 'aa aa', 'empty_output': False} for p in prompts]}
        (folder / f'update-{update}.json').write_text(json.dumps(record))
    return state, config, folder


def test_complete_pilot_and_requested_copying_pass(tmp_path):
    state, config, _ = evidence(tmp_path)
    _verify_finished_diagnostics(state, config, tmp_path)


def test_pilot_budget_shortfall_is_rejected(tmp_path):
    state, config, _ = evidence(tmp_path, terminal=199)
    with pytest.raises(ValueError, match='budget'):
        _verify_finished_diagnostics(state, config, tmp_path)


def test_actual_full_terminal_evidence_is_required_after_skips(tmp_path):
    state, config, _ = evidence(tmp_path, terminal=198, budget=None)
    with pytest.raises(ValueError, match='update 198'):
        _verify_finished_diagnostics(state, config, tmp_path)


@pytest.mark.parametrize('change', ['missing', 'blank', 'failed', 'instruction', 'decoding'])
def test_incomplete_or_mismatched_saved_diagnostic_fails(tmp_path, change):
    state, config, folder = evidence(tmp_path)
    path = folder / 'update-200.json'
    record = json.loads(path.read_text())
    if change == 'missing':
        path.unlink()
    else:
        if change == 'blank':
            record['rows'][0]['output'] = ''
        elif change == 'failed':
            record['status'] = 'failed'
        elif change == 'instruction':
            record['rows'][0]['instruction'] = 'Truncated instruction'
        else:
            record['decoding']['repetition_penalty'] = 1.1
        path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        _verify_finished_diagnostics(state, config, tmp_path)
