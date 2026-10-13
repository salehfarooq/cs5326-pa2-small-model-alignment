"""Verified-input custom Qwen SFT/HH/UF training. Requires explicit run scope."""

import argparse
import hashlib
import json
import math
import platform
import re
import subprocess
from importlib.metadata import version
from pathlib import Path

import torch

from src.data import _canonical_prompt, format_prompt
from src.training_diagnostics import _fingerprint_model, _reference_intact, create_generation_callback
from src.datafiles import parse_rows
from src.data_tokens import filter_for_context
from src.training import default_config, train, _write_json

MODEL_ID = 'Qwen/Qwen2.5-0.5B'
REVISION = '060db6499f32faf8b98477b0a26969ef7d8b9987'

EXCLUSION_BANK_ROWS = {'alpaca_eval.jsonl': 805, 'hh_rlhf_eval_300_seed42.json': 300}
DEV_CONTEXT_BUDGET = 8192


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_prepared(folder, track):
    """Verify completed split manifests and provenance before any model download."""
    folder = Path(folder)
    payload = (folder / 'manifest.json').read_bytes()
    manifest = json.loads(payload)
    kind = 'sft' if track == 'sft' else 'dpo'
    quotas = {'train': 30000 if kind == 'sft' else 15000, 'validation': 1000}
    config = manifest['config']
    schema = {'sft': 'ultrachat_first_turn', 'hh': 'hh', 'uf': 'ultrafeedback'}[track]
    if (manifest['version'] != 1 or config['kind'] != kind or config['schema'] != schema or
            config['seed'] != 42 or config['sizes'] != quotas or set(manifest['splits']) != set(quotas) or
            config['prompt_template'] != format_prompt('{instruction}')):
        raise ValueError('prepared manifest does not match the selected guide track')
    token = config.get('token_filter', {})
    if any(token.get(k) != value for k, value in {'model_id': MODEL_ID, 'revision': REVISION,
           'max_length': 512, 'max_prompt_tokens': 256, 'policy': 'preserve_full_prompt_filter'}.items()):
        raise ValueError('requires final pinned-token-eligible data, not text-only splits')
    exclusion = config.get('exclusions', {})
    if {bank.get('name') for bank in exclusion.get('banks', [])} != {'alpaca_eval.jsonl', 'hh_rlhf_eval_300_seed42.json'}:
        raise ValueError('missing full evaluation-bank exclusion provenance')
    source_id = {'sft': 'HuggingFaceH4/ultrachat_200k', 'hh': 'Anthropic/hh-rlhf',
                 'uf': 'HuggingFaceH4/ultrafeedback_binarized'}[track]
    if manifest['source']['id'] != source_id or not re.fullmatch('[0-9a-f]{40}', manifest['source']['revision']):
        raise ValueError('unexpected source identity/revision')
    data, ids, indices = {}, set(), set()
    for split, quota in quotas.items():
        entry = manifest['splits'][split]
        if entry['file'] != split + '.jsonl':
            raise ValueError('invalid split filename')
        raw = (folder / entry['file']).read_bytes()
        rows = parse_rows(raw, entry['file'])
        if hashlib.sha256(raw).hexdigest() != entry['sha256'] or len(rows) != entry['rows'] or len(rows) != quota:
            raise ValueError('split hash/row quota mismatch')
        if [row['id'] for row in rows] != entry['ids'] or [row['source_index'] for row in rows] != entry['source_indices']:
            raise ValueError('split ID/source-index mismatch')
        for row in rows:
            if row['id'] in ids or row['source_index'] in indices:
                raise ValueError('duplicate ID/source index across training and validation')
            ids.add(row['id']); indices.add(row['source_index'])
        data[split] = rows
    from src.training import _validate_rows
    train_prompts = _validate_rows(data['train'], kind)
    val_prompts = _validate_rows(data['validation'], kind)
    if train_prompts & val_prompts:
        raise ValueError('training and validation prompts overlap')
    identity = {'manifest_sha256': hashlib.sha256(payload).hexdigest(),
                'source': manifest['source'], 'config': config,
                'splits': {split: {key: entry[key] for key in ['file', 'rows', 'sha256']}
                           for split, entry in manifest['splits'].items()}}
    return data, identity


def verify_sft_best(folder, allow_pilot=False):
    folder = Path(folder).resolve()
    state = json.loads((folder / 'state.json').read_text())
    if state.get('checkpoint') != 'best' or state.get('config', {}).get('kind') != 'sft':
        raise ValueError('DPO must initialize from a saved SFT best checkpoint')
    metadata = state.get('metadata', {})
    if metadata.get('model_id') != MODEL_ID or metadata.get('model_revision') != REVISION:
        raise ValueError('SFT checkpoint model/tokenizer origin mismatch')
    if state.get('updates', 0) <= 0:
        raise ValueError('SFT best has no successful training updates; do not use untrained/pilot baseline')
    if metadata.get('run_scope') not in (['full', 'pilot'] if allow_pilot else ['full']):
        raise ValueError('DPO requires full SFT best; pilots must not become production initialization')
    files = state.get('files', {})
    if not files or not any(name.endswith('.safetensors') for name in files):
        raise ValueError('missing checkpoint weight-file identity')
    actual = {str(path.relative_to(folder)) for path in folder.rglob('*') if path.is_file()}
    if actual != set(files) | {'state.json'}:
        raise ValueError('checkpoint file set differs from its manifest')
    for name, entry in files.items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('invalid checkpoint manifest path')
        target = folder / path
        if target.stat().st_size != entry['bytes'] or _sha256(target) != entry['sha256']:
            raise ValueError('SFT checkpoint file hash mismatch')
    return {'path': str(folder), 'state_sha256': _sha256(folder / 'state.json'), 'state': state}


def _dpo_origin(folder, allow_pilot=False, require_completed=False):
    checked = None
    if require_completed:
        from src.checkpoints import verify_checkpoint
        checked = verify_checkpoint(folder)
        if checked['state']['checkpoint'] != 'best' or checked['state']['config']['kind'] != 'sft':
            raise ValueError('reviewed DPO diagnostics require a completed full SFT best origin')
    origin = verify_sft_best(folder, allow_pilot=allow_pilot)
    if checked is not None:
        origin['parent_run'] = checked['parent_run']
    return origin


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--track', required=True, choices=['sft', 'hh', 'uf'])
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument('--max-updates', type=int, help='bounded optimizer pilot; does not produce a full SFT origin')
    scope.add_argument('--full-run', action='store_true')
    parser.add_argument('--sft-best', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--reference-device', default='cuda:1')
    parser.add_argument('--eval-every', type=int, default=50)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--resume', help='unsupported; explicitly rejected')
    parser.add_argument('--development-prompts', type=Path)
    parser.add_argument('--generation-updates', type=str)
    parser.add_argument('--decoding-config', type=Path)
    parser.add_argument('--exclusion-bank', type=Path, action='append')
    args = parser.parse_args(argv)
    if args.resume is not None:
        parser.error('resume is unsupported; preserve prior outputs and use a fresh run directory')
    if (args.track == 'sft' and args.sft_best is not None) or (args.track != 'sft' and args.sft_best is None):
        parser.error('HH/UF require --sft-best; SFT starts from the pinned base')
    if args.max_updates is not None and args.max_updates <= 0:
        parser.error('--max-updates must be positive')
    if args.eval_every <= 0:
        parser.error('--eval-every must be positive')
    if args.output_dir.exists() or args.output_dir.is_symlink():
        parser.error('output directory must be new; previous runs are preserved')
    dev_args = [args.development_prompts, args.generation_updates, args.decoding_config]
    banks = args.exclusion_bank or []
    if any(a is not None for a in dev_args) or banks:
        if not all(a is not None for a in dev_args) or len(banks) != 2:
            parser.error('--development-prompts, --generation-updates, --decoding-config, and exactly two '
                         '--exclusion-bank paths must be supplied together')
    return args


def validate_development_inputs(args, data, identity, c):
    """Validate every development-diagnostic input before any model allocation."""
    if args.development_prompts is None:
        return None
    from scripts.dpo_preflight import validate_development
    raw = args.development_prompts.read_bytes()
    prompts_rows = json.loads(raw)
    if not isinstance(prompts_rows, list) or not prompts_rows:
        raise ValueError('development prompts must be a nonempty JSON list')
    for p in prompts_rows:
        if not isinstance(p, dict):
            raise ValueError('each development prompt must be a dict')
        for key in ('id', 'group', 'instruction'):
            if not isinstance(p.get(key), str) or not p[key].strip():
                raise ValueError(f'development prompt {key} must be nonblank')
    validate_development(prompts_rows)
    prompts_sha = hashlib.sha256(raw).hexdigest()
    updates_raw = []
    for item in args.generation_updates.split(','):
        item = item.strip()
        if item == 'terminal':
            updates_raw.append('terminal')
        else:
            try:
                updates_raw.append(int(item))
            except ValueError:
                raise ValueError(f'invalid generation update: {item}')
    decoding_bytes = args.decoding_config.read_bytes()
    decoding_raw = json.loads(decoding_bytes)
    if not isinstance(decoding_raw, dict):
        raise ValueError('decoding config must be a JSON object')
    if decoding_raw.get('do_sample') is not False:
        raise ValueError('decoding config do_sample must be false')
    required_keys = {'do_sample', 'max_new_tokens', 'repetition_penalty', 'no_repeat_ngram_size'}
    if set(decoding_raw) != required_keys:
        raise ValueError('decoding config must have exactly do_sample, max_new_tokens, repetition_penalty, no_repeat_ngram_size')
    decoding = {k: decoding_raw[k] for k in ('max_new_tokens', 'repetition_penalty', 'no_repeat_ngram_size')}
    # Validate the shared generation contract before loading either model.
    create_generation_callback(prompts_rows, {0}, decoding)
    if decoding['max_new_tokens'] > 256:
        raise ValueError('max_new_tokens must be at most 256')
    decoding_sha = hashlib.sha256(decoding_bytes).hexdigest()
    dev_canonical = {_canonical_prompt(p['instruction']) for p in prompts_rows}
    for split in ('train', 'validation'):
        for row in data[split]:
            if _canonical_prompt(row['prompt']) in dev_canonical:
                raise ValueError(f'development prompts overlap with {split} prompts')
    for track in ('sft', 'hh', 'uf'):
        if track == args.track:
            continue
        sibling_dir = args.data_dir.parent / track
        if not (sibling_dir / 'manifest.json').is_file():
            raise ValueError(f'prepared sibling track missing: {track}')
        sibling_data, _ = load_prepared(sibling_dir, track)
        for split_rows in sibling_data.values():
            for row in split_rows:
                if _canonical_prompt(row['prompt']) in dev_canonical:
                    raise ValueError(f'development prompts overlap with prepared track {track}')
    exclusion = identity['config'].get('exclusions', {})
    recorded = {bank.get('name'): bank for bank in exclusion.get('banks', [])}
    if set(recorded) != set(EXCLUSION_BANK_ROWS):
        raise ValueError('missing full evaluation-bank exclusion provenance')
    provided = {}
    for bank_path in args.exclusion_bank or []:
        name = bank_path.name
        if name not in recorded:
            raise ValueError(f'unexpected exclusion bank: {name}')
        if name in provided:
            raise ValueError(f'duplicate exclusion bank: {name}')
        bank_raw = bank_path.read_bytes()
        bank_rows = parse_rows(bank_raw, name)
        if hashlib.sha256(bank_raw).hexdigest() != recorded[name].get('sha256'):
            raise ValueError(f'exclusion bank {name} content hash mismatch')
        if len(bank_rows) != EXCLUSION_BANK_ROWS[name]:
            raise ValueError(f'exclusion bank {name} row count mismatch')
        provided[name] = bank_rows
    if set(provided) != set(recorded):
        raise ValueError('missing exclusion banks: ' + ', '.join(sorted(set(recorded) - set(provided))))
    for name, bank_rows in provided.items():
        for row in bank_rows:
            if not isinstance(row, dict):
                raise ValueError(f'exclusion bank {name} contains a non-object row')
            text = row.get('instruction') or row.get('prompt') or ''
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f'exclusion bank {name} has no complete instruction')
            if _canonical_prompt(text) in dev_canonical:
                raise ValueError(f'development prompt overlaps with exclusion bank {name}')
    window_size = c['batch_size'] * c['accumulation']
    planned = math.ceil(len(data['train']) / window_size) * c['epochs']
    terminal = min(args.max_updates, planned) if args.max_updates is not None else planned
    updates_set = set()
    for u in updates_raw:
        if u == 'terminal':
            updates_set.add(terminal)
            continue
        if u < 0 or u > terminal:
            raise ValueError(f'generation update {u} exceeds run budget {terminal}')
        if u != 0 and u % c['eval_every'] != 0 and u != terminal:
            raise ValueError(f'generation update {u} is not a validation position')
        updates_set.add(u)
    return {'prompts': prompts_rows, 'updates': updates_set, 'decoding': decoding,
            'prompts_sha256': prompts_sha, 'decoding_sha256': decoding_sha, 'terminal': terminal}


def _validate_development_context(tokenizer, dev_config):
    """Full formatted prompt plus response must fit the development context budget."""
    long_probes = 0
    for p in dev_config['prompts']:
        tokens = len(tokenizer.encode(format_prompt(p['instruction']), add_special_tokens=False))
        if tokens + dev_config['decoding']['max_new_tokens'] > DEV_CONTEXT_BUDGET:
            raise ValueError('development prompt plus response exceeds context budget')
        if p['group'] == 'extraction' and int(p['id'][4:]) > 15 and tokens > 256:
            long_probes += 1
    if long_probes < 2:
        raise ValueError('requires two new full long extraction probes within context budget')


def _verify_finished_diagnostics(state, dev_config, output_dir):
    """A completed loop is usable only with its required diagnostic evidence."""
    if dev_config is None:
        return
    expected = set(dev_config['updates']) | {state['updates']}
    if state['config']['max_updates'] is not None and state['updates'] != state['config']['max_updates']:
        raise ValueError('pilot ended before its successful-update budget')
    ids = [p['id'] for p in dev_config['prompts']]
    instructions = [p['instruction'] for p in dev_config['prompts']]
    for update in sorted(expected):
        path = Path(output_dir) / 'development' / f'update-{update}.json'
        if not path.is_file():
            raise ValueError(f'missing required development generation at update {update}')
        record = json.loads(path.read_text())
        rows = record.get('rows', [])
        if (record.get('status') != 'completed' or record.get('update') != update or
                record.get('decoding') != dev_config['decoding'] or
                [r.get('id') for r in rows] != ids or
                [r.get('instruction') for r in rows] != instructions or
                any(r.get('empty_output') or not isinstance(r.get('output'), str) or
                    not r['output'].strip() for r in rows)):
            raise ValueError(f'invalid required development generation at update {update}')


def main(argv=None):
    args = parse_args(argv)
    kind = 'sft' if args.track == 'sft' else 'dpo'
    c = default_config(kind)
    c.update(max_updates=args.max_updates, eval_every=args.eval_every, beta=args.beta, amp=True)
    if args.lr is not None:
        c['lr'] = args.lr
    from src.training import validate_config
    validate_config(c)
    for device in [args.device] + ([args.reference_device] if kind == 'dpo' else []):
        parsed = torch.device(device)
        if parsed.type != 'cuda' or parsed.index is None or parsed.index >= torch.cuda.device_count():
            raise ValueError('training CLI requires an available explicit CUDA device')
    if kind == 'dpo' and torch.device(args.device) == torch.device(args.reference_device):
        raise ValueError('use separate policy/reference GPUs for the reviewed memory layout')
    data, identity = load_prepared(args.data_dir, args.track)
    dev_config = validate_development_inputs(args, data, identity, c)
    origin = _dpo_origin(args.sft_best, allow_pilot=args.max_updates is not None,
                         require_completed=dev_config is not None) if kind == 'dpo' else None
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer_path = str(args.sft_best) if origin else MODEL_ID
    common = {'revision': REVISION, 'token': False, 'trust_remote_code': False} if not origin else {'local_files_only': True, 'trust_remote_code': False}
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **common)
    if tokenizer.eos_token_id != 151643:
        raise ValueError('unexpected Qwen tokenizer EOS')
    for split, rows in data.items():
        kept, counts = filter_for_context(rows, kind, tokenizer)
        if len(kept) != len(rows):
            raise ValueError('prepared token eligibility does not reproduce: ' + split + ' ' + str(counts))
    if dev_config is not None:
        _validate_development_context(tokenizer, dev_config)
    torch.manual_seed(c['seed'])
    torch.cuda.manual_seed_all(c['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    source = str(args.sft_best) if origin else MODEL_ID
    model = AutoModelForCausalLM.from_pretrained(source, torch_dtype=torch.float32, **common).to(args.device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    reference = None
    if origin:
        reference = AutoModelForCausalLM.from_pretrained(source, torch_dtype=torch.float32, **common).to(args.reference_device)
        reference.config.use_cache = False
        if _dpo_origin(args.sft_best, allow_pilot=args.max_updates is not None,
                       require_completed=dev_config is not None) != origin:
            raise ValueError('SFT checkpoint changed during model initialization')
    repo = Path(__file__).resolve().parents[1]
    metadata = {'track': args.track, 'model_id': MODEL_ID, 'model_revision': REVISION,
                'run_scope': 'pilot' if args.max_updates is not None else 'full',
                'data': identity, 'sft_origin': origin, 'policy_device': args.device,
                'reference_device': args.reference_device if origin else None,
                'parameter_dtype': 'float32', 'autocast_dtype': 'float16',
                'reference_dtype': 'float32' if origin else None, 'gradient_checkpointing': True,
                'python': platform.python_version(), 'versions': {name: version(name) for name in ['torch', 'transformers', 'safetensors']},
                'git_head': (json.loads((repo / 'source_manifest.json').read_text())['source_git_head']
                             if (repo / 'source_manifest.json').is_file() else
                             subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()),
                'source_hashes': {str(p.relative_to(repo)): _sha256(p) for p in sorted((repo / 'src').glob('*.py'))},
                'script_sha256': _sha256(Path(__file__)), 'seeded_not_claimed_bitwise_deterministic': True}
    if dev_config is not None:
        metadata['development'] = {'prompts_sha256': dev_config['prompts_sha256'],
                                   'decoding_sha256': dev_config['decoding_sha256'],
                                   'generation_updates': sorted(dev_config['updates']),
                                   'terminal': dev_config['terminal'],
                                   'diagnostic_identity': 'training_diagnostics'}
    torch.cuda.reset_peak_memory_stats(args.device)
    if origin:
        torch.cuda.reset_peak_memory_stats(args.reference_device)
    callback = None
    initial_fp = None
    if dev_config is not None:
        if reference is not None:
            initial_fp = _fingerprint_model(reference)
        callback = create_generation_callback(dev_config['prompts'], dev_config['updates'],
                                              dev_config['decoding'], reference)
    try:
        state = train(model, tokenizer, data['train'], data['validation'], c, args.output_dir,
                      reference, metadata, validation_callback=callback)
    except Exception:
        if reference is not None and initial_fp is not None:
            try:
                if not _reference_intact(reference, initial_fp):
                    run_path = args.output_dir / 'run.json'
                    record = json.loads(run_path.read_text()) if run_path.exists() else {}
                    record['reference_fingerprint_mismatch'] = True
                    _write_json(run_path, record)
            except Exception:
                pass
        raise
    final_fp = None
    try:
        _verify_finished_diagnostics(state, dev_config, args.output_dir)
        if reference is not None and initial_fp is not None:
            final_fp = _fingerprint_model(reference)
            if final_fp != initial_fp or not _reference_intact(reference, initial_fp):
                raise ValueError('reference fingerprint changed after training')
            _write_json(args.output_dir / 'reference_integrity.json',
                        {'status': 'verified', 'initial_fingerprint': initial_fp,
                         'final_fingerprint': final_fp, 'all_parameters_frozen': True,
                         'all_gradients_absent': True, 'sft_origin': origin})
    except BaseException as exc:
        run_path = args.output_dir / 'run.json'
        record = json.loads(run_path.read_text())
        record['status'] = 'failed'
        record['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        if final_fp is not None and final_fp != initial_fp:
            record['reference_fingerprint_mismatch'] = True
        _write_json(run_path, record)
        raise
    memory = {'policy_peak_bytes': torch.cuda.max_memory_allocated(args.device),
              'reference_peak_bytes': torch.cuda.max_memory_allocated(args.reference_device) if origin else None}
    _write_json(args.output_dir / 'cuda_memory.json', memory)
    print(json.dumps({'status': state['status'], 'updates': state['updates'],
                      'skipped_updates': state['skipped_updates'], 'best_metrics': state['best_metrics'],
                      'memory': memory}, indent=2))


if __name__ == '__main__':
    main()
