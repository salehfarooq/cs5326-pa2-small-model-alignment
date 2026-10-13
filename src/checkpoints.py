"""Verified reusable checkpoint generation for SFT/DPO best/final."""

import hashlib
import json
import re
from pathlib import Path

from scripts.train_qwen import MODEL_ID, REVISION


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _state_json_bytes(state):
    return (json.dumps(state, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')


def verify_checkpoint(folder):
    folder = Path(folder).absolute()
    if any(p.is_symlink() for p in [folder, *folder.parents]):
        raise ValueError('symlink in checkpoint path')
    folder = folder.resolve()
    state_path = folder / 'state.json'
    if state_path.is_symlink() or not state_path.is_file():
        raise ValueError('checkpoint state.json missing')
    state = json.loads(state_path.read_text())
    if not isinstance(state, dict):
        raise ValueError('state.json must be an object')

    checkpoint = state.get('checkpoint')
    if checkpoint not in ('best', 'final'):
        raise ValueError('checkpoint must be best or final')
    allowed_status = ('running', 'completed') if checkpoint == 'best' else ('completed',)
    if state.get('status') not in allowed_status:
        raise ValueError('checkpoint status is not usable')
    config = state.get('config', {})
    if not isinstance(config, dict) or config.get('kind') not in ('sft', 'dpo'):
        raise ValueError('config.kind must be sft or dpo')
    kind = config['kind']
    updates = state.get('updates')
    if isinstance(updates, bool) or not isinstance(updates, int) or updates <= 0:
        raise ValueError('updates must be a positive integer')
    metadata = state.get('metadata', {})
    if not isinstance(metadata, dict):
        raise ValueError('metadata must be an object')
    if metadata.get('model_id') != MODEL_ID or metadata.get('model_revision') != REVISION:
        raise ValueError('model ID/revision mismatch')
    if metadata.get('run_scope') != 'full':
        raise ValueError('run_scope must be full')
    if metadata.get('track') not in (('sft',) if kind == 'sft' else ('hh', 'uf')):
        raise ValueError('checkpoint track disagrees with its training kind')

    parent_run_path = folder.parent / 'run.json'
    if parent_run_path.is_symlink() or not parent_run_path.is_file():
        raise ValueError('parent run.json missing')
    parent_run = json.loads(parent_run_path.read_text())
    if not isinstance(parent_run, dict):
        raise ValueError('run.json must be an object')
    if parent_run.get('status') != 'completed':
        raise ValueError('parent run must be completed')
    parent_updates = parent_run.get('updates')
    if isinstance(parent_updates, bool) or not isinstance(parent_updates, int) or parent_updates <= 0:
        raise ValueError('parent updates must be a positive integer')
    if checkpoint == 'best':
        if updates > parent_updates:
            raise ValueError('best checkpoint updates cannot exceed parent run updates')
    elif updates != parent_updates:
        raise ValueError('final checkpoint updates must equal parent run updates')
    if metadata != parent_run.get('metadata', {}):
        raise ValueError('metadata mismatch with parent run')
    if config != parent_run.get('config', {}):
        raise ValueError('config mismatch with parent run')

    if kind == 'dpo':
        sft_origin = metadata.get('sft_origin')
        if not isinstance(sft_origin, dict):
            raise ValueError('DPO checkpoint missing sft_origin')
        origin_state = sft_origin.get('state', {})
        if not isinstance(origin_state, dict):
            raise ValueError('sft_origin state must be an object')
        origin_metadata = origin_state.get('metadata', {})
        if not isinstance(origin_metadata, dict) or not isinstance(origin_state.get('config'), dict):
            raise ValueError('invalid sft_origin metadata/config')
        if origin_state.get('checkpoint') != 'best':
            raise ValueError('sft_origin must be a best checkpoint')
        if origin_state.get('config', {}).get('kind') != 'sft':
            raise ValueError('sft_origin must be SFT')
        if origin_metadata.get('model_id') != MODEL_ID or origin_metadata.get('model_revision') != REVISION:
            raise ValueError('sft_origin model ID/revision mismatch')
        if origin_metadata.get('run_scope') != 'full':
            raise ValueError('sft_origin must be full run')
        if origin_metadata.get('track') != 'sft':
            raise ValueError('sft_origin track must be sft')
        if origin_state.get('status') not in ('running', 'completed'):
            raise ValueError('sft_origin checkpoint status is not usable')
        origin_updates = origin_state.get('updates')
        if isinstance(origin_updates, bool) or not isinstance(origin_updates, int) or origin_updates <= 0:
            raise ValueError('sft_origin has no successful updates')
        origin_files = origin_state.get('files', {})
        if not isinstance(origin_files, dict) or 'model.safetensors' not in origin_files:
            raise ValueError('sft_origin missing weight files')
        _validate_manifest(origin_files)
        origin_state_sha = sft_origin.get('state_sha256')
        if not isinstance(origin_state_sha, str):
            raise ValueError('sft_origin missing state hash')
        if hashlib.sha256(_state_json_bytes(origin_state)).hexdigest() != origin_state_sha:
            raise ValueError('sft_origin state hash mismatch')
        origin_parent = sft_origin.get('parent_run')
        if (not isinstance(origin_parent, dict) or origin_parent.get('status') != 'completed' or
                origin_parent.get('metadata') != origin_metadata or
                origin_parent.get('config') != origin_state['config'] or
                isinstance(origin_parent.get('updates'), bool) or
                not isinstance(origin_parent.get('updates'), int) or
                origin_parent['updates'] < origin_updates):
            raise ValueError('sft_origin lacks matching completed parent provenance')

    files = state.get('files', {})
    if not isinstance(files, dict) or not files:
        raise ValueError('missing files manifest')
    required = {'model.safetensors', 'tokenizer.json', 'config.json'}
    if not required.issubset(files):
        raise ValueError('missing required checkpoint files')
    _validate_manifest(files)
    for name, entry in files.items():
        if not isinstance(entry, dict) or 'bytes' not in entry or 'sha256' not in entry:
            raise ValueError('invalid manifest entry: ' + name)
        path = Path(name)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('invalid manifest path')
        target = folder / path
        if target.is_symlink():
            raise ValueError('symlink in checkpoint')
        if not target.is_file():
            raise ValueError('missing checkpoint file: ' + name)
    actual = set()
    for path in folder.rglob('*'):
        if path.is_symlink():
            raise ValueError('symlink in checkpoint')
        if path.is_file():
            actual.add(str(path.relative_to(folder)))
    if actual != set(files) | {'state.json'}:
        raise ValueError('checkpoint file set differs from manifest')
    for name, entry in files.items():
        target = folder / name
        if target.stat().st_size != entry['bytes']:
            raise ValueError('file size mismatch: ' + name)
        if _sha256(target) != entry['sha256']:
            raise ValueError('file hash mismatch: ' + name)

    return {
        'folder': str(folder),
        'state': state,
        'parent_run': parent_run,
        'state_sha256': _sha256(state_path),
        'model_sha256': _sha256(folder / 'model.safetensors'),
    }


def _validate_manifest(files):
    for name, entry in files.items():
        if (not isinstance(name, str) or name in ('', '.', 'state.json') or
                Path(name).is_absolute() or '..' in Path(name).parts or
                str(Path(name)) != name):
            raise ValueError('invalid manifest path')
        if not isinstance(entry, dict):
            raise ValueError('invalid manifest entry')
        size = entry.get('bytes')
        digest = entry.get('sha256')
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError('invalid manifest size')
        if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
            raise ValueError('invalid manifest SHA256')
        if name == 'model.safetensors' and size == 0:
            raise ValueError('empty checkpoint weights')
