"""Identity checks before downloading or running any model."""

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('train_cli', Path(__file__).parents[1] / 'scripts/train_qwen.py')
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def checkpoint(tmp_path, scope='full'):
    root = tmp_path / 'best'
    root.mkdir()
    (root / 'model.safetensors').write_bytes(b'fixture weights, not an executable model')
    state = {'checkpoint': 'best', 'config': {'kind': 'sft'}, 'updates': 2,
             'metadata': {'model_id': cli.MODEL_ID, 'model_revision': cli.REVISION, 'run_scope': scope},
             'files': {'model.safetensors': {'bytes': (root / 'model.safetensors').stat().st_size,
                                            'sha256': cli._sha256(root / 'model.safetensors')}}}
    (root / 'state.json').write_text(json.dumps(state))
    return root


def test_sft_checkpoint_origin_and_file_integrity(tmp_path):
    root = checkpoint(tmp_path)
    identity = cli.verify_sft_best(root)
    assert identity['state']['updates'] == 2
    (root / 'model.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash'):
        cli.verify_sft_best(root)


def test_pilot_sft_allowed_only_for_bounded_dpo_pilot(tmp_path):
    root = checkpoint(tmp_path, 'pilot')
    with pytest.raises(ValueError, match='full'):
        cli.verify_sft_best(root)
    assert cli.verify_sft_best(root, allow_pilot=True)['state']['metadata']['run_scope'] == 'pilot'


@pytest.mark.parametrize('mutation', ['wrong_model', 'zero_updates', 'not_sft', 'not_best', 'extra_file', 'traversal'])
def test_invalid_sft_origin_rejected(tmp_path, mutation):
    root = checkpoint(tmp_path)
    state = json.loads((root / 'state.json').read_text())
    if mutation == 'wrong_model': state['metadata']['model_id'] = 'other'
    if mutation == 'zero_updates': state['updates'] = 0
    if mutation == 'not_sft': state['config']['kind'] = 'dpo'
    if mutation == 'not_best': state['checkpoint'] = 'final'
    if mutation == 'extra_file': (root / 'unexpected').write_text('extra')
    if mutation == 'traversal': state['files']['../escape'] = state['files'].pop('model.safetensors')
    (root / 'state.json').write_text(json.dumps(state))
    with pytest.raises(ValueError): cli.verify_sft_best(root)


@pytest.mark.parametrize('extra', [[], ['--resume', 'old'], ['--full-run', '--max-updates', '1'],
                                 ['--max-updates', '0'], ['--full-run', '--sft-best', 'best']])
def test_cli_rejects_ambiguous_restart_or_scope(tmp_path, extra):
    with pytest.raises(SystemExit):
        cli.parse_args(['--track', 'sft', '--data-dir', 'data', '--output-dir', str(tmp_path/'new'), *extra])


def test_prepared_text_only_stage_is_rejected():
    # Tiny early-failure fixture; no 31k-row test allocation or downloads.
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as temp:
        root = Path(temp)
        (root / 'manifest.json').write_text(json.dumps({'version': 1, 'config': {
            'kind': 'sft', 'schema': 'ultrachat_first_turn', 'seed': 42,
            'sizes': {'train': 30000, 'validation': 1000},
            'prompt_template': 'Instruction:\n{instruction}\n\nResponse:\n'},
            'splits': {'train': {}, 'validation': {}}}))
        with pytest.raises(ValueError, match='token-eligible'):
            cli.load_prepared(root, 'sft')
