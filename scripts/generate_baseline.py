"""Pinned base-model greedy outputs on the unchanged 300-prompt Alpaca bank."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from src.datafiles import parse_rows
from src.generation import MAIN_DECODING, generate_development
from src.training import _write_json

MODEL_ID = 'Qwen/Qwen2.5-0.5B'
REVISION = '060db6499f32faf8b98477b0a26969ef7d8b9987'
BANK_SHA256 = '3989bb58dae97eb5fded3b21de17f8d76605488d0ba3ab7d17fc77e38fdfd6c1'


def load_bank(path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != BANK_SHA256:
        raise ValueError('fixed Alpaca evaluation bank changed')
    rows = parse_rows(path.read_bytes(), path)
    if len(rows) != 300 or len({r['id'] for r in rows}) != 300:
        raise ValueError('expected exactly 300 distinct fixed IDs')
    for row in rows:
        if not isinstance(row['instruction'], str) or not row['instruction'].strip():
            raise ValueError('blank instruction')
    return rows


def export(model, tokenizer, rows, output):
    """Persist every finished response; final JSON exists only after all rows finish."""
    output.mkdir(exist_ok=False)
    state = {'status': 'running', 'completed': 0, 'total': len(rows),
             'model_id': MODEL_ID, 'model_revision': REVISION, 'bank_sha256': BANK_SHA256,
             'decoding': {'do_sample': False, **MAIN_DECODING},
             'parameter_dtype': str(next(model.parameters()).dtype),
             'prompt_policy': 'preserve full formatted instruction; no truncation'}
    results, diagnostics = [], []
    _write_json(output / 'run.json', state)
    try:
        with (output / 'partial.jsonl').open('x', encoding='utf-8') as handle:
            for i, row in enumerate(rows):
                generated = generate_development(model, tokenizer, [row['instruction']], **MAIN_DECODING)[0]
                result = {**row, 'source_dataset': row.get('dataset'), 'dataset': 'alpaca_eval',
                          'output': generated['output'], 'generator': 'qwen2.5-0.5b-base'}
                handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
                handle.flush()
                os.fsync(handle.fileno())
                results.append(result)
                diagnostics.append({'id': row['id'], **generated})
                state['completed'] = i + 1
                _write_json(output / 'run.json', state)
                if i < 3 or (i + 1) % 10 == 0:
                    print(f"[BASE {i+1}/{len(rows)}] tokens={generated['response_tokens']} "
                          f"finish={generated['finish_reason']} empty={generated['empty_output']}", flush=True)
        _write_json(output / 'outputs.json', results)
        _write_json(output / 'diagnostics.json', diagnostics)
        state.update(status='completed', outputs_sha256=hashlib.sha256((output / 'outputs.json').read_bytes()).hexdigest(),
                     empty_outputs=sum(r['empty_output'] for r in diagnostics),
                     length_limited=sum(r['finish_reason'] == 'length' for r in diagnostics))
        _write_json(output / 'run.json', state)
        print(json.dumps(state, indent=2), flush=True)
        return state
    except BaseException as exc:
        state.update(status='failed', error={'type': type(exc).__name__, 'message': str(exc)})
        _write_json(output / 'run.json', state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--sanity-only', action='store_true')
    args = parser.parse_args()
    rows = load_bank(args.bank)
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise ValueError('output directory must be new')
    from transformers import AutoModelForCausalLM, AutoTokenizer
    common = {'revision': REVISION, 'token': False, 'trust_remote_code': False}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **common)
    if tokenizer.eos_token_id != 151643:
        raise ValueError('unexpected tokenizer EOS')
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32, **common).to(args.device)
    # Independent development prompts were inspected during the CUDA pilot. Recheck
    # decoding at the fixed 256-token budget before touching the full evaluation bank.
    fixture = json.loads((Path(__file__).parent / 'pilot_inputs.json').read_text())
    prompts = fixture['development_prompts']
    if isinstance(prompts[0], dict):
        prompts = [row['instruction'] for row in prompts]
    sanity = generate_development(model, tokenizer, prompts[:3], **MAIN_DECODING)
    for row in sanity:
        print('[BASE sanity] ' + json.dumps(row, ensure_ascii=False), flush=True)
    if args.sanity_only:
        _write_json(args.output_dir, sanity)
        return
    export(model, tokenizer, rows, args.output_dir)
    _write_json(args.output_dir / 'sanity.json', sanity)


if __name__ == '__main__':
    main()
