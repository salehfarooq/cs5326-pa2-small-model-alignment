"""Independent checkpoint identity/failure checks for the DPO handoff."""
import copy
import hashlib
import json

import pytest

from src.checkpoints import verify_checkpoint
from scripts.train_qwen import MODEL_ID, REVISION
from src.training import default_config


def fixture(tmp_path, kind='sft', track='sft', checkpoint='best'):
    root = tmp_path / 'run'
    folder = root / checkpoint
    folder.mkdir(parents=True)
    files = {}
    for name in ['model.safetensors', 'tokenizer.json', 'tokenizer_config.json', 'config.json']:
        content = ('fixture ' + name).encode()
        (folder / name).write_bytes(content)
        files[name] = {'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}
    meta = {'model_id': MODEL_ID, 'model_revision': REVISION, 'run_scope': 'full', 'track': track}
    config = default_config(kind)
    state = {'status': 'running' if checkpoint == 'best' else 'completed', 'updates': 8,
             'checkpoint': checkpoint, 'metadata': meta, 'config': config, 'files': files}
    run = {'status': 'completed', 'updates': 8, 'metadata': copy.deepcopy(meta), 'config': config}
    (folder / 'state.json').write_text(json.dumps(state, sort_keys=True, indent=2) + '\n')
    (root / 'run.json').write_text(json.dumps(run))
    return folder


def alter(path, change):
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))


def test_historical_running_best_is_valid_only_with_completed_parent(tmp_path):
    folder = fixture(tmp_path)
    assert verify_checkpoint(folder)
    alter(folder.parent / 'run.json', lambda r: r.update(status='failed'))
    with pytest.raises(ValueError):
        verify_checkpoint(folder)


@pytest.mark.parametrize('name', ['model.safetensors', 'tokenizer.json', 'config.json'])
def test_tampered_checkpoint_is_rejected(tmp_path, name):
    folder = fixture(tmp_path)
    (folder / name).write_bytes(b'bad')
    with pytest.raises(ValueError):
        verify_checkpoint(folder)


def test_unmanifested_file_is_rejected(tmp_path):
    folder = fixture(tmp_path)
    (folder / 'extra.json').write_text('{}')
    with pytest.raises(ValueError):
        verify_checkpoint(folder)


@pytest.mark.parametrize('value', [0, -1, True, 1.5])
def test_successful_update_is_positive_integer(tmp_path, value):
    folder = fixture(tmp_path)
    alter(folder / 'state.json', lambda r: r.update(updates=value))
    with pytest.raises(ValueError):
        verify_checkpoint(folder)


def test_final_update_must_match_completed_parent(tmp_path):
    folder = fixture(tmp_path, checkpoint='final')
    alter(folder.parent / 'run.json', lambda r: r.update(updates=9))
    with pytest.raises(ValueError):
        verify_checkpoint(folder)


def test_best_may_precede_final_but_not_follow_it(tmp_path):
    folder = fixture(tmp_path)
    alter(folder.parent / 'run.json', lambda r: r.update(updates=9))
    assert verify_checkpoint(folder)
    alter(folder / 'state.json', lambda r: r.update(updates=10))
    with pytest.raises(ValueError):
        verify_checkpoint(folder)


def test_manifest_symlink_rejected_even_with_matching_bytes(tmp_path):
    folder = fixture(tmp_path)
    target = tmp_path / 'outside'
    target.write_bytes((folder / 'tokenizer.json').read_bytes())
    (folder / 'tokenizer.json').unlink()
    (folder / 'tokenizer.json').symlink_to(target)
    with pytest.raises(ValueError):
        verify_checkpoint(folder)


def test_parent_identity_mismatch_rejected(tmp_path):
    folder = fixture(tmp_path)
    alter(folder.parent / 'run.json', lambda r: r['metadata'].update(track='hh'))
    with pytest.raises(ValueError):
        verify_checkpoint(folder)


def test_pilot_checkpoint_not_final_bank_origin(tmp_path):
    folder = fixture(tmp_path)
    alter(folder / 'state.json', lambda r: r['metadata'].update(run_scope='pilot'))
    alter(folder.parent / 'run.json', lambda r: r['metadata'].update(run_scope='pilot'))
    with pytest.raises(ValueError):
        verify_checkpoint(folder)


def test_checkpoint_folder_symlink_rejected(tmp_path):
    folder = fixture(tmp_path)
    alias = tmp_path / 'alias'
    alias.symlink_to(folder, target_is_directory=True)
    with pytest.raises(ValueError):
        verify_checkpoint(alias)


def test_failed_best_state_rejected_even_with_completed_parent(tmp_path):
    folder = fixture(tmp_path)
    alter(folder / 'state.json', lambda r: r.update(status='failed'))
    with pytest.raises(ValueError):
        verify_checkpoint(folder)


@pytest.mark.parametrize('track', ['hh', 'invented'])
def test_sft_checkpoint_track_is_sft(tmp_path, track):
    folder = fixture(tmp_path, track=track)
    with pytest.raises(ValueError):
        verify_checkpoint(folder)


def dpo_fixture(tmp_path):
    from src.checkpoints import _state_json_bytes
    sft=fixture(tmp_path/'sft')
    origin=verify_checkpoint(sft)
    dpo=fixture(tmp_path/'dpo',kind='dpo',track='hh')
    state=json.loads((dpo/'state.json').read_text())
    state['metadata']['sft_origin']={k:copy.deepcopy(origin[k]) for k in ('state','state_sha256','parent_run')}
    (dpo/'state.json').write_bytes(_state_json_bytes(state))
    parent=json.loads((dpo.parent/'run.json').read_text())
    parent['metadata']=copy.deepcopy(state['metadata'])
    (dpo.parent/'run.json').write_text(json.dumps(parent))
    return dpo


def test_dpo_embeds_completed_sft_origin_without_original_mount(tmp_path):
    import shutil
    dpo=dpo_fixture(tmp_path)
    shutil.rmtree(tmp_path/'sft')
    assert verify_checkpoint(dpo)['state']['metadata']['sft_origin']['parent_run']['status']=='completed'


@pytest.mark.parametrize('change',['missing_parent','failed_parent','wrong_parent_updates','state_tamper','wrong_track','malformed_files'])
def test_dpo_origin_cannot_claim_incomplete_or_altered_sft(tmp_path,change):
    from src.checkpoints import _state_json_bytes
    dpo=dpo_fixture(tmp_path)
    state=json.loads((dpo/'state.json').read_text())
    origin=state['metadata']['sft_origin']
    if change=='missing_parent':origin.pop('parent_run')
    if change=='failed_parent':origin['parent_run']['status']='failed'
    if change=='wrong_parent_updates':origin['parent_run']['updates']=7
    if change=='state_tamper':origin['state']['updates']=9
    if change=='wrong_track':origin['state']['metadata']['track']='hh'
    if change=='malformed_files':origin['state']['files']=[]
    (dpo/'state.json').write_bytes(_state_json_bytes(state))
    parent=json.loads((dpo.parent/'run.json').read_text())
    parent['metadata']=copy.deepcopy(state['metadata'])
    (dpo.parent/'run.json').write_text(json.dumps(parent))
    with pytest.raises(ValueError):verify_checkpoint(dpo)


def test_training_rechecks_same_complete_origin_after_model_load(tmp_path):
    from scripts.train_qwen import _dpo_origin
    folder=fixture(tmp_path)
    first=_dpo_origin(folder,allow_pilot=True,require_completed=True)
    assert first==_dpo_origin(folder,allow_pilot=True,require_completed=True)
    assert first['parent_run']['status']=='completed'
    alter(folder.parent/'run.json',lambda r:r.update(status='failed'))
    with pytest.raises(ValueError):_dpo_origin(folder,allow_pilot=True,require_completed=True)
