"""Generate fixed banks from verified, completed local SFT/DPO checkpoints."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from scripts.generate_baseline import load_bank as load_alpaca_bank
from src.checkpoints import verify_checkpoint
from src.datafiles import parse_rows
from src.generation import generate_development
from src.training import _write_json

HH_SHA256 = '0731105407d06152e584c722717712be4f418f221a8d5c487bb924c4d91c38c6'
LABELS = {'sft': 'qwen2.5-0.5b-sft', 'hh': 'qwen2.5-0.5b-dpo-hh', 'uf': 'qwen2.5-0.5b-dpo-uf'}
SANITY = [
    'Return only the product of 6 and 7.',
    'Write a Python function that counts each integer in a list using a dictionary.',
    'A stranger asks for my account password. Give a safe response in two sentences.',
]
PROMPT_POLICY = 'preserve full formatted instruction; no truncation'


def _sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def load_bank(path, kind):
    path = Path(path)
    if kind == 'alpaca_eval':
        rows = load_alpaca_bank(path)
    elif kind == 'hh':
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != HH_SHA256:
            raise ValueError('fixed HH evaluation bank changed')
        rows = parse_rows(raw, path)
    else:
        raise ValueError('unknown evaluation bank')
    _validate_rows(rows)
    return {'rows': rows, 'kind': kind, 'sha256': _sha256(path)}


def _validate_rows(rows):
    if not isinstance(rows, list) or len(rows) != 300:
        raise ValueError('fixed bank must contain 300 rows')
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('bank rows must be objects')
        for key in ('id', 'instruction'):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError('bank ID/instruction must be nonblank strings')
        if row['id'] in seen:
            raise ValueError('duplicate bank ID')
        seen.add(row['id'])


def _settings(decoding):
    if not isinstance(decoding, dict) or set(decoding) != {'do_sample', 'max_new_tokens', 'repetition_penalty', 'no_repeat_ngram_size'}:
        raise ValueError('incomplete decoding configuration')
    if decoding['do_sample'] is not False:
        raise ValueError('only greedy decoding is supported')
    # The shared helper validates types without touching a model.
    from src.training_diagnostics import create_generation_callback
    settings = {key: value for key, value in decoding.items() if key != 'do_sample'}
    create_generation_callback([{'id': 'check', 'group': 'check', 'instruction': 'check'}], {0}, settings)
    return settings


def _model_label(identity, generator):
    state = identity['state']
    track = state['metadata'].get('track')
    if track not in LABELS or state['config']['kind'] != ('sft' if track == 'sft' else 'dpo'):
        raise ValueError('invalid checkpoint model track')
    label = LABELS[track]
    if generator not in (track, label):
        raise ValueError('generator label disagrees with verified checkpoint')
    return track, label


def export_checkpoint(model, tokenizer, bank, generator, checkpoint_identity, output_dir,
                      decoding, source_identity):
    """Write every response immediately; completion requires all fixed IDs."""
    rows = bank['rows']
    _validate_rows(rows)
    settings = _settings(decoding)
    track, label = _model_label(checkpoint_identity, generator)
    if bank['kind'] not in ('alpaca_eval', 'hh'):
        raise ValueError('unknown evaluation bank')
    if tokenizer.eos_token_id != 151643:
        raise ValueError('unexpected Qwen tokenizer EOS')
    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError('output directory must be new')
    state = {'status': 'running', 'completed': 0, 'total': 300, 'bank_kind': bank['kind'],
             'bank_sha256': bank['sha256'], 'generator': label, 'checkpoint': checkpoint_identity,
             'decoding': dict(decoding), 'parameter_dtype': str(next(model.parameters()).dtype),
             'tokenizer_fingerprint': checkpoint_identity['state']['files']['tokenizer.json']['sha256'],
             'prompt_policy': PROMPT_POLICY, 'source': source_identity}
    if track == 'sft':
        state['origin_sha256'] = checkpoint_identity['model_sha256']
    else:
        state['frozen_sft_origin_sha256'] = checkpoint_identity['state']['metadata']['sft_origin']['state']['files']['model.safetensors']['sha256']
    output.mkdir(parents=False)
    results, diagnostics = [], []
    try:
        _write_json(output / 'run.json', state)
        sanity = generate_development(model, tokenizer, SANITY, **settings)
        _write_json(output / 'sanity.json', sanity)
        if any(row['empty_output'] for row in sanity):
            raise ValueError('blank sanity output; inspect before full generation')
        with (output / 'partial.jsonl').open('x', encoding='utf-8') as handle:
            for i, original in enumerate(rows):
                generated = generate_development(model, tokenizer, [original['instruction']], **settings)[0]
                result = {**original, 'source_dataset': original.get('dataset'),
                          'dataset': 'alpaca_eval' if bank['kind'] == 'alpaca_eval' else 'hh_rlhf_eval',
                          'output': generated['output'], 'generator': label}
                handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
                handle.flush()
                os.fsync(handle.fileno())
                results.append(result)
                diagnostics.append({'id': original['id'], **generated})
                state['completed'] = i + 1
                _write_json(output / 'diagnostics_partial.json', diagnostics)
                _write_json(output / 'run.json', state)
                if i < 3 or (i + 1) % 10 == 0:
                    print(f"[{track.upper()} {i+1}/300] tokens={generated['response_tokens']} "
                          f"finish={generated['finish_reason']} empty={generated['empty_output']}", flush=True)
        _write_json(output / 'outputs.json', results)
        _write_json(output / 'diagnostics.json', diagnostics)
        state.update(status='completed', outputs_sha256=_sha256(output / 'outputs.json'),
                     empty_outputs=sum(row['empty_output'] for row in diagnostics),
                     length_limited=sum(row['finish_reason'] == 'length' for row in diagnostics))
        _write_json(output / 'run.json', state)
        return state
    except BaseException as exc:
        state.update(status='failed', error={'type': type(exc).__name__, 'message': str(exc)})
        _write_json(output / 'run.json', state)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--bank-kind', choices=['alpaca_eval', 'hh'], required=True)
    parser.add_argument('--generator', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--repetition-penalty', type=float, default=1.05)
    parser.add_argument('--no-repeat-ngram-size', type=int, default=0)
    args = parser.parse_args(argv)
    identity = verify_checkpoint(args.checkpoint)
    _model_label(identity, args.generator)
    bank = load_bank(args.bank, args.bank_kind)
    decoding = {'do_sample': False, 'max_new_tokens': args.max_new_tokens,
                'repetition_penalty': args.repetition_penalty, 'no_repeat_ngram_size': args.no_repeat_ngram_size}
    _settings(decoding)
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError('output directory must be new')
    device = torch.device(args.device)
    if device.type != 'cuda' or device.index is None or device.index >= torch.cuda.device_count():
        raise ValueError('generation requires an available explicit CUDA device')
    from transformers import AutoModelForCausalLM, AutoTokenizer
    common = {'local_files_only': True, 'trust_remote_code': False}
    tokenizer = AutoTokenizer.from_pretrained(identity['folder'], **common)
    if tokenizer.eos_token_id != 151643:
        raise ValueError('unexpected tokenizer EOS')
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(identity['folder'], torch_dtype=torch.float32, **common).to(device)
    if verify_checkpoint(args.checkpoint) != identity:
        raise ValueError('checkpoint changed during model loading')
    repo = Path(__file__).resolve().parents[1]
    source = {'source_hashes': {str(p.relative_to(repo)): _sha256(p) for p in sorted((repo/'src').glob('*.py'))},
              'script_sha256': _sha256(Path(__file__)), 'seed': 42}
    export_checkpoint(model, tokenizer, bank, args.generator, identity, args.output_dir, decoding, source)


if __name__ == '__main__':
    main()
