"""Direct SFT/DPO loops. No Trainer; resumable optimizer state is not claimed."""

import contextlib
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
from pathlib import Path

import torch

from src.batching import build_sft_batch
from src.data import _canonical_prompt, format_prompt
from src.losses import (_model_device, compute_sft_loss, compute_response_logprobs,
                        compute_dpo_loss_from_logps)


def values_for_display(record):
    return ' '.join(f'{name}={record[name]:.6f}' for name in
                    ['loss', 'reward_accuracy', 'reward_margin'] if name in record)


def default_config(kind):
    if kind not in ('sft', 'dpo'):
        raise ValueError('kind must be sft or dpo')
    return {'kind': kind, 'batch_size': 2 if kind == 'sft' else 1,
            'accumulation': 16 if kind == 'sft' else 32, 'epochs': 1,
            'lr': 2e-5 if kind == 'sft' else 1e-6,
            'weight_decay': 0.1 if kind == 'sft' else 0.0,
            'warmup_ratio': 0.03 if kind == 'sft' else 0.0,
            'schedule': 'cosine' if kind == 'sft' else 'fixed',
            'beta': 0.1, 'clip_norm': 1.0, 'max_length': 512,
            'eval_every': 50, 'eval_batch_size': 1, 'seed': 42,
            'max_updates': None, 'amp': False}


def validate_config(config):
    defaults = default_config(config.get('kind'))
    if set(config) != set(defaults):
        raise ValueError('config must contain exactly the default_config keys')
    for name in ['batch_size', 'accumulation', 'epochs', 'max_length', 'eval_every', 'eval_batch_size']:
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(name + ' must be a positive integer')
    if isinstance(config['seed'], bool) or not isinstance(config['seed'], int) or config['seed'] < 0:
        raise ValueError('seed must be a nonnegative integer')
    for name in ['lr', 'beta', 'clip_norm', 'weight_decay', 'warmup_ratio']:
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(name + ' must be finite')
        if value < 0 or (name in ['lr', 'beta', 'clip_norm'] and value == 0):
            raise ValueError(name + ' has an invalid range')
    if not 0 <= config['warmup_ratio'] < 1:
        raise ValueError('warmup_ratio must be in [0,1)')
    if config['schedule'] not in ('fixed', 'cosine') or type(config['amp']) is not bool:
        raise ValueError('invalid schedule or amp')
    if config['schedule'] == 'fixed' and config['warmup_ratio'] != 0:
        raise ValueError('fixed schedule requires zero warmup')
    limit = config['max_updates']
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
        raise ValueError('max_updates must be a positive integer or None')


def _validate_rows(rows, kind):
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError('training and validation rows must be nonempty lists')
    keys = ['prompt', 'response'] if kind == 'sft' else ['prompt', 'chosen', 'rejected']
    prompts = set()
    for row in rows:
        if not isinstance(row, dict) or any(not isinstance(row.get(k), str) or not row[k].strip() for k in keys):
            raise ValueError('rows require nonblank training fields')
        key = _canonical_prompt(row['prompt'])
        if key in prompts:
            raise ValueError('duplicate prompt within a split')
        prompts.add(key)
    return prompts


def _autocast(model, amp):
    if amp:
        return torch.autocast(device_type=_model_device(model).type, dtype=torch.float16)
    return contextlib.nullcontext()


def _microbatch(model, reference, tokenizer, rows, config):
    prompts = [format_prompt(row['prompt']) for row in rows]
    if config['kind'] == 'sft':
        responses = [row['response'] for row in rows]
        labels = build_sft_batch(tokenizer, prompts, responses, config['max_length'])['labels']
        count = int(labels[:, 1:].ne(-100).sum())
        with _autocast(model, config['amp']):
            loss = compute_sft_loss(model, tokenizer, prompts, responses, config['max_length'])
        return loss, count, {}
    chosen = [row['chosen'] for row in rows]
    rejected = [row['rejected'] for row in rows]
    with _autocast(model, config['amp']):
        pc = compute_response_logprobs(model, tokenizer, prompts, chosen, config['max_length'])
        pr = compute_response_logprobs(model, tokenizer, prompts, rejected, config['max_length'])
    with torch.no_grad(), _autocast(reference, config['amp']):
        rc = compute_response_logprobs(reference, tokenizer, prompts, chosen, config['max_length']).to(pc.device)
        rr = compute_response_logprobs(reference, tokenizer, prompts, rejected, config['max_length']).to(pc.device)
    loss = compute_dpo_loss_from_logps(pc, pr, rc, rr, config['beta'])
    rw = config['beta'] * (pc.detach() - rc)
    rl = config['beta'] * (pr.detach() - rr)
    values = {'reward_accuracy': (rw > rl).float().mean().item(),
              'reward_margin': (rw - rl).mean().item(),
              'chosen_logp': pc.detach().mean().item(), 'rejected_logp': pr.detach().mean().item()}
    return loss, len(rows), values


def evaluate(model, reference, tokenizer, rows, config):
    """Token-weighted SFT and pair-weighted DPO metrics; restore caller modes."""
    modes = [(model, model.training)]
    if reference is not None:
        modes.append((reference, reference.training))
    totals, count = {}, 0
    try:
        for item, _ in modes:
            item.eval()
        with torch.no_grad():
            for start in range(0, len(rows), config['eval_batch_size']):
                loss, size, values = _microbatch(model, reference, tokenizer,
                                               rows[start:start + config['eval_batch_size']], config)
                values['loss'] = loss.item()
                if not all(math.isfinite(value) for value in values.values()):
                    raise FloatingPointError('nonfinite validation metric')
                for name, value in values.items():
                    totals[name] = totals.get(name, 0.0) + value * size
                count += size
        if count == 0:
            raise ValueError('validation requires scoreable rows')
        return {name: value / count for name, value in totals.items()}
    finally:
        for item, training in modes:
            item.train(training)


def _write_json(path, value):
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n'
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, encoding='utf-8', delete=False) as handle:
        temp = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _checkpoint(model, tokenizer, output, name, state):
    """Write fully before replacing a previous best; preserve it on save failure."""
    stage = Path(tempfile.mkdtemp(prefix='.' + name + '-', dir=output))
    target, backup = output / name, output / ('.' + name + '-previous')
    try:
        if hasattr(model, 'save_pretrained'):
            model.save_pretrained(stage, safe_serialization=True)
            tokenizer.save_pretrained(stage)
        else:
            torch.save(model.state_dict(), stage / 'weights.pt')
        files = {}
        for path in sorted(stage.rglob('*')):
            if path.is_file():
                digest = hashlib.sha256()
                with path.open('rb') as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                        digest.update(chunk)
                files[str(path.relative_to(stage))] = {'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}
        _write_json(stage / 'state.json', {**state, 'files': files})
        if target.exists():
            os.rename(target, backup)
        try:
            os.rename(stage, target)
        except BaseException:
            if backup.exists():
                os.rename(backup, target)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def _scheduler(optimizer, config, planned_updates):
    warmup = min(planned_updates - 1, math.ceil(planned_updates * config['warmup_ratio']))
    def factor(completed):
        if config['schedule'] == 'fixed':
            return 1.0
        if completed < warmup:
            return (completed + 1) / warmup
        progress = min(1.0, (completed - warmup) / max(1, planned_updates - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def train(model, tokenizer, train_rows, validation_rows, config, output_dir, reference=None,
          metadata=None, scaler=None, validation_callback=None):
    """Train a bounded pilot or epochs with exact accumulation and best/final saves.

    max_updates limits successful updates; skipped AMP steps consume their data
    window and are recorded. No resume: output_dir must be new. RNGs are seeded;
    callers must seed separately before creating a randomly initialized model.
    """
    validate_config(config)
    if validation_callback is not None and not callable(validation_callback):
        raise ValueError('validation_callback must be callable')
    train_prompts = _validate_rows(train_rows, config['kind'])
    val_prompts = _validate_rows(validation_rows, config['kind'])
    if train_prompts & val_prompts:
        raise ValueError('training and validation prompts overlap')
    if config['kind'] == 'dpo':
        if reference is None or reference is model:
            raise ValueError('DPO requires an independent reference model')
        policy_storage = {(p.device, p.untyped_storage().data_ptr()) for p in model.parameters()}
        if any((p.device, p.untyped_storage().data_ptr()) in policy_storage for p in reference.parameters()):
            raise ValueError('policy and reference share parameter storage')
    elif reference is not None:
        raise ValueError('SFT does not use a reference')
    params = [p for p in model.parameters() if p.requires_grad]
    if not params or any(p.dtype != torch.float32 for p in params):
        raise ValueError('trainable policy parameters must be FP32')
    if config['amp'] and (_model_device(model).type != 'cuda' or
                          (reference is not None and _model_device(reference).type != 'cuda')):
        raise ValueError('FP16 AMP requires CUDA models')
    if reference is not None:
        reference.requires_grad_(False)
        reference.zero_grad(set_to_none=True)
        reference.eval()
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
    window_size = config['batch_size'] * config['accumulation']
    planned = math.ceil(len(train_rows) / window_size) * config['epochs']
    if config['max_updates'] is not None and config['max_updates'] > planned:
        raise ValueError('max_updates exceeds the available epoch updates')
    output = Path(output_dir)
    # Validate all metadata serialization before creating the owned output.
    state = {'config': dict(config), 'metadata': metadata or {}, 'planned_updates': planned,
             'updates': 0, 'attempts': 0, 'skipped_updates': 0, 'status': 'running',
             'resume_supported': False, 'best_metrics': None}
    json.dumps(state, allow_nan=False)
    output.mkdir(exist_ok=False)
    optimizer = torch.optim.AdamW(params, lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = _scheduler(optimizer, config, planned)
    if scaler is None:
        scaler = torch.amp.GradScaler('cuda', enabled=config['amp'], init_scale=128.0)
    records = []
    model_mode = model.training
    def log(record):
        records.append(record)
        payload = ''.join(json.dumps(row, sort_keys=True, allow_nan=False) + '\n' for row in records)
        path = output / 'metrics.jsonl'
        temp = output / '.metrics.tmp'
        with temp.open('w', encoding='utf-8') as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        _write_json(output / 'run.json', state)
        if record['event'] == 'validation':
            print(f"[{config['kind'].upper()} validation {state['updates']}/{planned}] "
                  f"{values_for_display(record)} best={record['best']}", flush=True)
        elif state['attempts'] == 1 or state['attempts'] % 10 == 0 or not record['optimizer_step']:
            print(f"[{config['kind'].upper()} update {state['updates']}/{planned}] "
                  f"loss={record['loss']:.6f} lr={record['lr']:.3g} "
                  f"skipped={state['skipped_updates']}", flush=True)
    def validation():
        print(f"[{config['kind'].upper()}] validating {len(validation_rows)} examples "
              f"at update {state['updates']}; saving best checkpoint if improved", flush=True)
        values = evaluate(model, reference, tokenizer, validation_rows, config)
        metric = 'loss' if config['kind'] == 'sft' else 'reward_accuracy'
        previous = state['best_metrics']
        better = previous is None or (values[metric] < previous[metric] if metric == 'loss'
                                     else values[metric] > previous[metric])
        if better:
            checkpoint_state = {**state, 'best_metrics': values, 'checkpoint': 'best'}
            _checkpoint(model, tokenizer, output, 'best', checkpoint_state)
            state['best_metrics'] = values
        log({'event': 'validation', 'updates': state['updates'], 'attempts': state['attempts'],
             'best': better, **values})
        if validation_callback is not None:
            validation_callback(model, tokenizer, state['updates'], values, output)
    try:
        _write_json(output / 'run.json', state)
        print(f"[{config['kind'].upper()}] starting {len(train_rows)} training examples, "
              f"{planned} planned updates; progress: {output / 'metrics.jsonl'}", flush=True)
        validation()
        rng = random.Random(config['seed'])
        stop = False
        for epoch in range(config['epochs']):
            indices = list(range(len(train_rows)))
            rng.shuffle(indices)
            for start in range(0, len(indices), window_size):
                window = [train_rows[index] for index in indices[start:start + window_size]]
                batches = [window[i:i + config['batch_size']] for i in range(0, len(window), config['batch_size'])]
                if config['kind'] == 'sft':
                    denominator = 0
                    for batch in batches:
                        labels = build_sft_batch(tokenizer, [format_prompt(r['prompt']) for r in batch],
                                                 [r['response'] for r in batch], config['max_length'])['labels']
                        denominator += int(labels[:, 1:].ne(-100).sum())
                else:
                    denominator = len(window)
                if denominator == 0:
                    raise ValueError('accumulation window has no targets')
                model.train()
                optimizer.zero_grad(set_to_none=True)
                totals = {}
                for batch in batches:
                    loss, size, values = _microbatch(model, reference, tokenizer, batch, config)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('nonfinite training loss')
                    weighted_loss = loss * (size / denominator)
                    scaler.scale(weighted_loss).backward()
                    values['loss'] = loss.detach().item()
                    for name, value in values.items():
                        if not math.isfinite(value):
                            raise FloatingPointError('nonfinite training metric')
                        totals[name] = totals.get(name, 0.0) + value * size / denominator
                scaler.unscale_(optimizer)
                finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in params)
                norm = None
                if finite:
                    norm = float(torch.nn.utils.clip_grad_norm_(params, config['clip_norm'], error_if_nonfinite=True))
                elif not config['amp']:
                    raise FloatingPointError('nonfinite training gradients')
                old_scale = scaler.get_scale()
                used_lr = optimizer.param_groups[0]['lr']
                scaler.step(optimizer)
                scaler.update()
                successful = scaler.get_scale() >= old_scale
                state['attempts'] += 1
                if successful:
                    if not all(torch.isfinite(p).all() for p in params):
                        raise FloatingPointError('nonfinite parameters after optimizer update')
                    state['updates'] += 1
                    scheduler.step()
                else:
                    state['skipped_updates'] += 1
                optimizer.zero_grad(set_to_none=True)
                log({'event': 'train', 'epoch': epoch + 1, 'updates': state['updates'],
                     'attempts': state['attempts'], 'examples': len(window), 'targets': denominator,
                     'lr': used_lr, 'grad_norm': norm, 'finite_gradients': bool(finite),
                     'scale': scaler.get_scale(), 'optimizer_step': successful, **totals})
                if successful and state['updates'] % config['eval_every'] == 0:
                    validation()
                if config['max_updates'] is not None and state['updates'] >= config['max_updates']:
                    stop = True
                    break
            if stop:
                break
        if records[-1]['event'] != 'validation' or records[-1]['attempts'] != state['attempts']:
            validation()
        state['status'] = 'completed'
        _checkpoint(model, tokenizer, output, 'final', {**state, 'checkpoint': 'final'})
        _write_json(output / 'run.json', state)
        return state
    except BaseException as exc:
        state['status'] = 'failed'
        state['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        _write_json(output / 'run.json', state)
        raise
    finally:
        optimizer.zero_grad(set_to_none=True)
        model.train(model_mode)
