"""Training-time development generation diagnostics."""

import hashlib
import json
import math
import os
import random
from pathlib import Path

import torch
import numpy as np

from src.data import _canonical_prompt
from src.generation import generate_development
from src.training import _write_json


def _tensor_bytes(tensor):
    t = tensor.detach().cpu().contiguous().view(-1)
    return t.view(torch.uint8).numpy()


def _fingerprint_model(model):
    digest = hashlib.sha256()
    with torch.no_grad():
        for name, p in model.named_parameters():
            digest.update(name.encode('utf-8'))
            digest.update(str(p.dtype).encode('ascii'))
            digest.update(str(tuple(p.shape)).encode('ascii'))
            arr = _tensor_bytes(p)
            digest.update(str(len(arr)).encode('ascii'))
            for i in range(0, len(arr), 1024 * 1024):
                digest.update(arr[i:i + 1024 * 1024].tobytes())
        for name, b in model.named_buffers():
            digest.update(name.encode('utf-8'))
            digest.update(str(b.dtype).encode('ascii'))
            digest.update(str(tuple(b.shape)).encode('ascii'))
            arr = _tensor_bytes(b)
            digest.update(str(len(arr)).encode('ascii'))
            for i in range(0, len(arr), 1024 * 1024):
                digest.update(arr[i:i + 1024 * 1024].tobytes())
    return digest.hexdigest()


def _capture_modes(model):
    return [(m, m.training) for m in model.modules()]


def _restore_modes(modules):
    for m, training in modules:
        m.train(training)


def _capture_cache(config):
    return hasattr(config, 'use_cache'), getattr(config, 'use_cache', None)


def _restore_cache(config, had_cache, value):
    if had_cache:
        config.use_cache = value
    elif hasattr(config, 'use_cache'):
        del config.use_cache


def _reference_intact(reference, expected_fingerprint):
    if _fingerprint_model(reference) != expected_fingerprint:
        return False
    for p in reference.parameters():
        if p.grad is not None or p.requires_grad:
            return False
    return True


def create_generation_callback(prompts, updates, decoding, reference=None):
    if not isinstance(prompts, (list, tuple)) or not prompts:
        raise ValueError('development prompts must be a nonempty list')
    seen_ids, seen_instructions = set(), set()
    frozen = []
    for p in prompts:
        if not isinstance(p, dict):
            raise ValueError('each development prompt must be a dict')
        pid, group, instruction = p.get('id'), p.get('group'), p.get('instruction')
        if not isinstance(pid, str) or not pid.strip():
            raise ValueError('development prompt id must be nonblank')
        if pid in seen_ids:
            raise ValueError('duplicate development prompt id')
        seen_ids.add(pid)
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError('development prompt instruction must be nonblank')
        canonical = _canonical_prompt(instruction)
        if canonical in seen_instructions:
            raise ValueError('duplicate canonical development instruction')
        seen_instructions.add(canonical)
        if not isinstance(group, str) or not group.strip():
            raise ValueError('development prompt group must be nonblank')
        frozen.append({'id': pid, 'group': group, 'instruction': instruction})
    if not isinstance(updates, (set, frozenset)) or not updates:
        raise ValueError('generation updates must be a nonempty set')
    for u in updates:
        if isinstance(u, bool) or not isinstance(u, int) or u < 0:
            raise ValueError('generation updates must be nonnegative integers')
    updates = frozenset(updates)
    if not isinstance(decoding, dict):
        raise ValueError('decoding config must be a dict')
    required = {'max_new_tokens', 'repetition_penalty', 'no_repeat_ngram_size'}
    if set(decoding) != required:
        raise ValueError('decoding config must have exactly max_new_tokens, repetition_penalty, no_repeat_ngram_size')
    mnt = decoding['max_new_tokens']
    if isinstance(mnt, bool) or not isinstance(mnt, int) or mnt <= 0:
        raise ValueError('max_new_tokens must be positive')
    rp = decoding['repetition_penalty']
    if isinstance(rp, bool) or not isinstance(rp, (int, float)) or not math.isfinite(rp) or rp < 1:
        raise ValueError('repetition_penalty must be at least 1')
    nrs = decoding['no_repeat_ngram_size']
    if isinstance(nrs, bool) or not isinstance(nrs, int) or nrs < 0:
        raise ValueError('no_repeat_ngram_size must be nonnegative')
    decoding = dict(decoding)
    initial_fp = None
    if reference is not None:
        initial_fp = _fingerprint_model(reference)
    state = {'initialized': False, 'reference_fingerprint': initial_fp}

    def callback(model, tokenizer, update, values, output_dir):
        if reference is not None:
            if not _reference_intact(reference, state['reference_fingerprint']):
                raise ValueError('reference fingerprint changed before diagnostics')
        if update not in updates:
            return
        output = Path(output_dir)
        dev_dir = output / 'development'
        update_file = dev_dir / f'update-{update}.json'
        partial_path = dev_dir / f'update-{update}-partial.jsonl'
        if update_file.exists():
            raise ValueError(f'development generation already exists for update {update}')
        if partial_path.exists():
            raise ValueError(f'development generation partial already exists for update {update}')
        if not state['initialized']:
            dev_dir.mkdir(exist_ok=False)
            settings_dir = dev_dir / 'settings'
            settings_dir.mkdir(exist_ok=False)
            _write_json(settings_dir / 'decoding.json', dict(decoding))
            _write_json(settings_dir / 'prompts.json', frozen)
            state['initialized'] = True
        py_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_cpu_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        policy_modes = _capture_modes(model)
        policy_had_cache, policy_cache = _capture_cache(model.config)
        ref_modes = _capture_modes(reference) if reference is not None else None
        ref_had_cache, ref_cache = (_capture_cache(reference.config) if reference is not None
                                    else (False, None))
        try:
            model.config.use_cache = True
            rows = []
            blank = []
            print(f'[development] update {update}: generating {len(frozen)} prompts', flush=True)
            with partial_path.open('x', encoding='utf-8') as handle:
                for idx, prompt in enumerate(frozen):
                    print(f'[development] update {update}: generating prompt {idx + 1}/{len(frozen)}',
                          flush=True)
                    row = generate_development(model, tokenizer, [prompt['instruction']],
                                               max_new_tokens=mnt, repetition_penalty=rp,
                                               no_repeat_ngram_size=nrs)[0]
                    rows.append(row)
                    record = {**row, 'id': prompt['id'], 'group': prompt['group'],
                              'update': update, 'decoding': dict(decoding)}
                    handle.write(json.dumps(record, sort_keys=True) + '\n')
                    handle.flush()
                    os.fsync(handle.fileno())
                    print(f'[development] update {update}: prompt {idx + 1}/{len(frozen)} done', flush=True)
                    if row['empty_output']:
                        blank.append(idx)
            if reference is not None and not _reference_intact(reference, initial_fp):
                raise ValueError('reference fingerprint changed during diagnostics')
            if blank:
                failure = {'update': update, 'status': 'failed', 'reason': 'blank generation',
                           'blank_indices': blank}
                _write_json(update_file, failure)
                raise ValueError(f'blank generation at update {update} for indices {blank}')
            completed = {'update': update, 'status': 'completed',
                         'prompts_sha256': hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest(),
                         'decoding': dict(decoding),
                         'rows': [{**r, 'id': frozen[i]['id'], 'group': frozen[i]['group'],
                                   'update': update, 'decoding': dict(decoding)}
                                  for i, r in enumerate(rows)]}
            _write_json(update_file, completed)
            summary_path = dev_dir / 'summary.json'
            summary = json.loads(summary_path.read_text()) if summary_path.exists() else {'updates': []}
            summary.setdefault('updates', []).append(update)
            summary['total_updates'] = len(summary['updates'])
            summary['total_prompts'] = summary.get('total_prompts', 0) + len(rows)
            summary['total_blank'] = summary.get('total_blank', 0) + len(blank)
            summary['total_response_tokens'] = summary.get('total_response_tokens', 0) + sum(
                r['response_tokens'] for r in rows)
            summary['avg_response_tokens'] = (summary['total_response_tokens'] / summary['total_prompts']
                                              if summary['total_prompts'] else 0.0)
            summary.setdefault('finish_reasons', {})
            for r in rows:
                key = r['finish_reason']
                summary['finish_reasons'][key] = summary['finish_reasons'].get(key, 0) + 1
            _write_json(summary_path, summary)
        except BaseException as exc:
            failure = {'update': update, 'status': 'failed',
                       'reason': 'exception during development generation',
                       'error': {'type': type(exc).__name__, 'message': str(exc)}}
            try:
                if reference is not None:
                    failure['reference_fingerprint_mismatch'] = not _reference_intact(reference, initial_fp)
                _write_json(update_file, failure)
            except Exception:
                pass
            raise
        finally:
            random.setstate(py_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
            _restore_modes(policy_modes)
            _restore_cache(model.config, policy_had_cache, policy_cache)
            if reference is not None:
                _restore_modes(ref_modes)
                _restore_cache(reference.config, ref_had_cache, ref_cache)

    return callback
