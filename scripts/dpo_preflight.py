"""Offline-only frozen development/data inspection. Never modifies training data."""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from scripts.train_qwen import load_prepared, MODEL_ID, REVISION
from src.data import _canonical_prompt, format_prompt
from src.datafiles import parse_rows
from src.training import _write_json

ORIGINAL_SHA = '5fe9cb7e79f875999813b0a360ca7fbc457c744dd52af71b41cff83f0dff9d30'
GROUPS = {'code', 'reasoning', 'safety', 'formatting', 'extraction'}


def read_bank(path):
    path = Path(path)
    raw = path.read_bytes()
    rows = parse_rows(raw, path)
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('instruction'), str) or not row['instruction'].strip():
            raise ValueError('bank rows require complete nonblank instructions')
    return rows, hashlib.sha256(raw).hexdigest()


def validate_development(rows):
    if not isinstance(rows, list) or len(rows) != 40:
        raise ValueError('development bank requires exactly 40 rows')
    for i, row in enumerate(rows):
        if (set(row) != {'id', 'group', 'instruction'} or row['id'] != f'dev-{i+1}' or
                row['group'] not in GROUPS or not isinstance(row['instruction'], str) or not row['instruction'].strip()):
            raise ValueError('development bank schema/order is invalid')
    if Counter(r['group'] for r in rows) != Counter({g: 8 for g in GROUPS}):
        raise ValueError('development groups must each have eight rows')
    if len({_canonical_prompt(r['instruction']) for r in rows}) != 40:
        raise ValueError('development instructions must be canonically distinct')
    original = json.dumps(rows[:15], sort_keys=True, ensure_ascii=False).encode()
    if hashlib.sha256(original).hexdigest() != ORIGINAL_SHA:
        raise ValueError('original 15 development probes changed')


def inspect_pair(row, split, tokenizer):
    prompt = tokenizer.encode(format_prompt(row['prompt']), add_special_tokens=False)
    if not prompt or len(prompt) > 256:
        raise ValueError('prepared pair prompt eligibility changed')
    result = {'split': split, 'row': row, 'prompt_tokens': len(prompt), 'responses': {}}
    retained = {}
    for side in ['chosen', 'rejected']:
        tokens = tokenizer.encode(row[side], add_special_tokens=False)
        budget = 512 - len(prompt)
        kept = (tokens + [tokenizer.eos_token_id])[:budget]
        retained[side] = kept
        result['responses'][side] = {'full_tokens': len(tokens), 'retained_tokens': len(kept),
            'truncated': len(tokens) + 1 > budget, 'eos_retained': len(tokens) + 1 <= budget,
            'retained_sha256': hashlib.sha256(json.dumps(kept).encode()).hexdigest()}
        if not tokens or not kept:
            raise ValueError('prepared response has no scoreable target')
    result['identical_retained_pair'] = retained['chosen'] == retained['rejected']
    if result['identical_retained_pair']:
        raise ValueError('prepared chosen/rejected responses are indistinguishable')
    result['chosen_rejected_length_ratio'] = result['responses']['chosen']['full_tokens'] / result['responses']['rejected']['full_tokens']
    return result


def audit(data_root, alpaca_full, hh_bank, development, tokenizer_path, output_dir):
    output_dir = Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError('output directory must be new')
    rows, dev_sha = read_bank(development)
    validate_development(rows)
    alpaca, alpaca_sha = read_bank(alpaca_full)
    hh, hh_sha = read_bank(hh_bank)
    if len(alpaca) != 805 or len(hh) != 300:
        raise ValueError('full fixed exclusion banks require 805/300 rows')
    data, identities, seen = {}, {}, set()
    for track in ['sft', 'hh', 'uf']:
        data[track], identities[track] = load_prepared(Path(data_root)/track, track)
        excluded = {b['name']: b['sha256'] for b in identities[track]['config']['exclusions']['banks']}
        if excluded != {'alpaca_eval.jsonl': alpaca_sha, 'hh_rlhf_eval_300_seed42.json': hh_sha}:
            raise ValueError('exclusion bank identity changed')
        for split_rows in data[track].values():
            seen.update(_canonical_prompt(r['prompt']) for r in split_rows)
    seen.update(_canonical_prompt(r['instruction']) for r in alpaca + hh)
    overlap = [r['id'] for r in rows if _canonical_prompt(r['instruction']) in seen]
    if overlap:
        raise ValueError('development overlap: ' + ', '.join(overlap))
    from transformers import AutoTokenizer
    tokenizer_path = Path(tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, trust_remote_code=False)
    if tokenizer.eos_token_id != 151643:
        raise ValueError('unexpected Qwen EOS')
    tokens = {r['id']: len(tokenizer.encode(format_prompt(r['instruction']), add_special_tokens=False)) for r in rows}
    if max(tokens.values()) > 8192 or sum(r['group'] == 'extraction' and int(r['id'][4:]) > 15 and tokens[r['id']] > 256 for r in rows) < 2:
        raise ValueError('requires two new full long extraction probes within context budget')
    tokenizer_files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(tokenizer_path.iterdir()) if p.is_file()}
    inspections = {}
    for track in ['hh', 'uf']:
        samples = []
        for split, count in [('train', 50), ('validation', 20)]:
            indices = random.Random(42).sample(range(len(data[track][split])), count)
            samples.extend(inspect_pair(data[track][split][i], split, tokenizer) for i in indices)
        inspections[track] = samples
    summary = {'status': 'completed', 'development_sha256': dev_sha, 'development_count': 40,
               'group_counts': dict(Counter(r['group'] for r in rows)), 'development_prompt_tokens': tokens,
               'overlap_ids': [], 'prepared_rows': sum(len(v) for d in data.values() for v in d.values()),
               'banks': {'alpaca_full_sha256': alpaca_sha, 'hh_sha256': hh_sha},
               'prepared': identities, 'tokenizer': {'model_id': MODEL_ID, 'revision': REVISION,
                   'files': tokenizer_files, 'vocab_sha256': hashlib.sha256(json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()).hexdigest()},
               'sampling': {'seed': 42, 'train_per_track': 50, 'validation_per_track': 20},
               'quality_review': 'pending; diagnostics do not establish preference correctness'}
    output_dir.mkdir(exist_ok=False)
    _write_json(output_dir/'development.json', rows)
    summary['development_export_sha256'] = hashlib.sha256((output_dir/'development.json').read_bytes()).hexdigest()
    for track, samples in inspections.items():
        _write_json(output_dir/(track+'_inspection.json'), samples)
    _write_json(output_dir/'summary.json', summary)
    print(json.dumps({k: summary[k] for k in ['status', 'development_count', 'group_counts', 'prepared_rows', 'quality_review']}, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['data-root', 'alpaca-full', 'hh-bank', 'development', 'tokenizer', 'output-dir']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    audit(args.data_root, args.alpaca_full, args.hh_bank, args.development, args.tokenizer, args.output_dir)


if __name__ == '__main__':
    main()
