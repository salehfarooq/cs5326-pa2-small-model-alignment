"""Bounded real-data CUDA optimizer pilots, independent HH/UF and development outputs."""
import argparse
import gc
import hashlib
import json
import random
from pathlib import Path

import torch

from src.data_tokens import filter_for_context
from src.generation import generate_development
from src.training import default_config, train, _write_json

MODEL_ID = 'Qwen/Qwen2.5-0.5B'
REVISION = '060db6499f32faf8b98477b0a26969ef7d8b9987'


def fingerprint(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        data = tensor.detach().cpu().contiguous().numpy()
        digest.update(memoryview(data).cast('B'))
    return digest.hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args(argv)
    payload = args.inputs.read_bytes()
    inputs = json.loads(payload)
    if inputs['scope'] != 'real-data optimizer plumbing pilot; not full training':
        raise ValueError('unexpected fixture scope')
    if inputs['model_revision'] != REVISION:
        raise ValueError('tokenizer pin mismatch')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise ValueError('two CUDA devices required')
    args.output_dir.mkdir(exist_ok=False)
    report = {'status': 'running', 'scope': inputs['scope'], 'model_id': MODEL_ID,
              'model_revision': REVISION, 'input_sha256': hashlib.sha256(payload).hexdigest(),
              'full_training': False, 'quality_improvement_claimed': False, 'tracks': {}}
    _write_json(args.output_dir / 'pilot_report.json', report)
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        common = {'revision': REVISION, 'token': False, 'trust_remote_code': False}
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **common)
        for track, data in inputs['tracks'].items():
            for split in ['train', 'validation']:
                kept, counts = filter_for_context(data[split], 'sft' if track == 'sft' else 'dpo', tokenizer)
                if len(kept) != len(data[split]):
                    raise ValueError('pilot token eligibility failed: ' + track + '/' + split)
        torch.manual_seed(42); torch.cuda.manual_seed_all(42); random.seed(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32, **common).to('cuda:0')
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        base_outputs = generate_development(model, tokenizer, inputs['development_prompts'], max_new_tokens=64)
        _write_json(args.output_dir / 'base_development.json', base_outputs)
        print('Base development outputs saved; starting two SFT updates.', flush=True)
        before = fingerprint(model)
        c = default_config('sft')
        c.update(amp=True, max_updates=2, eval_every=1)
        torch.cuda.reset_peak_memory_stats(0)
        data = inputs['tracks']['sft']
        state = train(model, tokenizer, data['train'], data['validation'], c, args.output_dir / 'sft',
                      metadata={'run_scope': 'pilot', 'fixture': inputs['tracks']['sft']['identity'],
                                'schedule_horizon': 'two fixture windows; differs from full 30k run'})
        after = fingerprint(model)
        if state['updates'] != 2 or state['skipped_updates'] or before == after:
            raise AssertionError('SFT pilot requires two finite, effective updates without overflows')
        sft_outputs = generate_development(model, tokenizer, inputs['development_prompts'], max_new_tokens=64)
        _write_json(args.output_dir / 'sft_development.json', sft_outputs)
        report['tracks']['sft'] = {'updates': 2, 'changed_parameters': True,
                                 'policy_peak_bytes': torch.cuda.max_memory_allocated(0),
                                 'best_metrics': state['best_metrics'], 'final_fingerprint': after}
        _write_json(args.output_dir / 'pilot_report.json', report)
        print('SFT pilot passed; starting independent DPO pilots.', flush=True)
        # Each DPO pilot loads this SAME SFT pilot final. These are plumbing
        # checks; production runs must start from full SFT best instead.
        origin = args.output_dir / 'sft/final'
        del model
        gc.collect(); torch.cuda.empty_cache()
        for track in ['hh', 'uf']:
            local = {'local_files_only': True, 'trust_remote_code': False}
            policy = AutoModelForCausalLM.from_pretrained(origin, torch_dtype=torch.float32, **local).to('cuda:0')
            reference = AutoModelForCausalLM.from_pretrained(origin, torch_dtype=torch.float32, **local).to('cuda:1')
            policy.config.use_cache = False
            policy.gradient_checkpointing_enable()
            reference.config.use_cache = False
            reference.requires_grad_(False); reference.eval()
            initial = fingerprint(policy)
            ref_before = fingerprint(reference)
            if initial != after or ref_before != after:
                raise AssertionError('DPO initialization differs from shared SFT origin')
            c = default_config('dpo'); c.update(amp=True, max_updates=1, eval_every=1)
            torch.cuda.reset_peak_memory_stats(0); torch.cuda.reset_peak_memory_stats(1)
            data = inputs['tracks'][track]
            state = train(policy, tokenizer, data['train'], data['validation'], c,
                          args.output_dir / track, reference,
                          metadata={'run_scope': 'pilot', 'fixture': data['identity'],
                                    'shared_sft_pilot_fingerprint': after})
            if state['updates'] != 1 or state['skipped_updates'] or fingerprint(policy) == initial:
                raise AssertionError(track + ' pilot did not make a finite effective update')
            if fingerprint(reference) != ref_before or any(p.grad is not None for p in reference.parameters()):
                raise AssertionError('reference changed or received gradients')
            outputs = generate_development(policy, tokenizer, inputs['development_prompts'], max_new_tokens=64)
            _write_json(args.output_dir / (track + '_development.json'), outputs)
            report['tracks'][track] = {'updates': 1, 'changed_parameters': True, 'reference_unchanged': True,
                                      'initial_policy_fingerprint': initial, 'best_metrics': state['best_metrics'],
                                      'policy_peak_bytes': torch.cuda.max_memory_allocated(0),
                                      'reference_peak_bytes': torch.cuda.max_memory_allocated(1)}
            _write_json(args.output_dir / 'pilot_report.json', report)
            print(track, json.dumps(report['tracks'][track]), flush=True)
            del policy, reference
            gc.collect(); torch.cuda.empty_cache()
        report['status'] = 'success'
    except BaseException as exc:
        report['status'] = 'failed'
        report['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        _write_json(args.output_dir / 'pilot_report.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
