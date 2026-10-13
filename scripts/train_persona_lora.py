"""Manual Task 5: a small, separately evaluated TRL/PEFT persona experiment."""
import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.persona_data import prepare_data

MODEL_ID = 'HuggingFaceTB/SmolLM2-360M-Instruct'
VERSIONS = {'transformers':'4.51.3', 'trl':'0.19.1', 'peft':'0.15.2', 'datasets':'3.6.0', 'accelerate':'1.7.0'}


def write_json(path, value):
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n'
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def frozen_hash(model):
    result = hashlib.sha256()
    for name, value in model.named_parameters():
        if 'lora_' not in name:
            if value.requires_grad:
                raise ValueError('Base parameter is trainable: '+name)
            result.update(name.encode())
            result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def checkpoint_manifest(directory):
    files = {}
    for path in sorted(directory.rglob('*')):
        if path.is_file():
            files[str(path.relative_to(directory))] = {'bytes':path.stat().st_size, 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    if 'adapter_model.safetensors' not in files or 'adapter_config.json' not in files:
        raise ValueError('Incomplete adapter checkpoint')
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--revision', required=True)
    args = parser.parse_args()
    if len(args.revision)!=40 or any(c not in '0123456789abcdef' for c in args.revision):
        raise ValueError('Revision must be a pinned full commit SHA')
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError('Preserve existing experiments: '+str(output))
    output.mkdir(parents=True)
    run = {'status':'initializing', 'model_id':MODEL_ID, 'revision':args.revision, 'seed':42}
    write_json(output/'run.json', run)
    trainer = None
    try:
        from importlib.metadata import version
        for package, expected in VERSIONS.items():
            if version(package)!=expected:
                raise ValueError(package+' version mismatch: '+version(package))
        import torch
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
        from peft import LoraConfig
        from trl import DPOConfig, DPOTrainer
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
            raise ValueError('Expose exactly GPU0 with CUDA_VISIBLE_DEVICES=0')
        set_seed(42)
        data_path = Path(args.data_path)
        rows = json.loads(data_path.read_text())
        if len(rows)!=220:
            raise ValueError('Expected complete supplied 220-pair dataset')
        train, validation, split = prepare_data(rows, seed=42)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=args.revision)
        if not tokenizer.chat_template or tokenizer.eos_token_id is None:
            raise ValueError('Native chat template and EOS required')
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'
        lengths = []
        for row in train+validation:
            for key in ['chosen','rejected']:
                tokens = tokenizer.apply_chat_template(row['prompt']+row[key], tokenize=True)
                lengths.append(len(tokens))
                if len(tokens)+1>1024:
                    raise ValueError('Context would truncate a preference')
        write_json(output/'split.json', split)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=args.revision, torch_dtype=torch.float32).to('cuda:0')
        modules = [name for name, module in model.named_modules() if name.endswith(('q_proj','v_proj'))]
        if not modules or not any(name.endswith('q_proj') for name in modules) or not any(name.endswith('v_proj') for name in modules):
            raise ValueError('Required query/value projection targets missing')
        model.config.use_cache = False
        lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, target_modules=['q_proj','v_proj'], bias='none', task_type='CAUSAL_LM')
        config = DPOConfig(output_dir=str(output/'checkpoints'), learning_rate=1e-5, beta=0.1,
                           num_train_epochs=1, max_steps=math.ceil(math.ceil(len(train)/2)/8),
                           per_device_train_batch_size=2, per_device_eval_batch_size=2,
                           gradient_accumulation_steps=8, optim='adamw_torch', lr_scheduler_type='constant',
                           warmup_steps=0, weight_decay=0.0, max_grad_norm=1.0, fp16=False, bf16=False,
                           max_length=1024, max_prompt_length=None, max_completion_length=None,
                           eval_strategy='steps', eval_steps=5, save_strategy='steps', save_steps=5,
                           logging_steps=1, logging_nan_inf_filter=False, eval_on_start=False,
                           load_best_model_at_end=True, metric_for_best_model='rewards/accuracies',
                           greater_is_better=True, save_total_limit=3, gradient_checkpointing=False,
                           disable_dropout=True, remove_unused_columns=False, report_to='none', seed=42,
                           data_seed=42, dataloader_num_workers=0)
        run.update(status='configured', versions={name:version(name) for name in VERSIONS}, torch_version=version('torch'),
                   source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   data_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(), train_count=len(train), validation_count=len(validation),
                   configuration=config.to_dict(), lora_configuration={'rank':8,'alpha':16,'dropout':0,'targets':['q_proj','v_proj']},
                   reference={'model_id':MODEL_ID,'revision':args.revision,'strategy':'same_frozen_base_with_adapter_disabled'},
                   max_observed_sequence_length=max(lengths), total_pairs=len(rows))
        write_json(output/'run.json', run)

        class Progress(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                for key, value in (logs or {}).items():
                    if isinstance(value, (int,float)) and not math.isfinite(value):
                        raise FloatingPointError('Nonfinite metric: '+key)
                record = {'step':state.global_step, **(logs or {})}
                with (output/'metrics.jsonl').open('a') as handle:
                    handle.write(json.dumps(record, allow_nan=False)+'\n')
                    handle.flush()
                    os.fsync(handle.fileno())
                run['step'] = state.global_step
                write_json(output/'run.json', run)
                print('[PERSONA TRAIN] '+json.dumps(record), flush=True)

            def on_step_end(self, args, state, control, model=None, **kwargs):
                for name, value in model.named_parameters():
                    if value.requires_grad and not torch.isfinite(value).all():
                        raise FloatingPointError('Nonfinite adapter: '+name)
                if state.global_step==state.max_steps:
                    model.save_pretrained(output/'final_adapter', safe_serialization=True)
                    tokenizer.save_pretrained(output/'final_adapter')
                    # Include the terminal update in checkpoint selection.
                    control.should_evaluate = True
                    control.should_save = True
                return control

        trainer = DPOTrainer(model=model, ref_model=None, args=config, processing_class=tokenizer,
                             train_dataset=Dataset.from_list(train), eval_dataset=Dataset.from_list(validation),
                             peft_config=lora, callbacks=[Progress()])
        if trainer.ref_model is not None:
            raise ValueError('Expected adapter-disabled reference')
        names = [name for name, value in trainer.model.named_parameters() if value.requires_grad]
        if not names or any('lora_' not in name for name in names):
            raise ValueError('Only LoRA parameters may be trainable')
        before = frozen_hash(trainer.model)
        run.update(status='ready', frozen_base_sha256_before=before, trainable_names=names,
                   trainable_parameters=sum(value.numel() for value in trainer.model.parameters() if value.requires_grad),
                   total_parameters=sum(value.numel() for value in trainer.model.parameters()))
        initial = trainer.evaluate()
        write_json(output/'initial_validation.json', initial)
        if initial.get('eval_rewards/accuracies')!=0.0:
            raise ValueError('Identical initial policy/reference must have strict reward accuracy zero')
        run.update(status='running', initial_validation=initial)
        write_json(output/'run.json', run)
        print('PERSONA TRAINING START '+json.dumps({'pairs':len(train),'validation':len(validation),'trainable':run['trainable_parameters']}),flush=True)
        trainer.train()
        if trainer.state.global_step!=13 or not math.isclose(trainer.state.epoch, 1.0, abs_tol=1e-8):
            raise ValueError('Expected one complete epoch, including the partial final batch')
        after = frozen_hash(trainer.model)
        if before!=after:
            raise ValueError('Frozen base/reference weights changed')
        if not (output/'final_adapter/adapter_model.safetensors').exists():
            raise ValueError('Final adapter was not captured before best reload')
        trainer.model.save_pretrained(output/'best_adapter', safe_serialization=True)
        tokenizer.save_pretrained(output/'best_adapter')
        selected_validation = trainer.evaluate()
        write_json(output/'best_validation.json', selected_validation)
        trainer.state.save_to_json(str(output/'trainer_state.json'))
        run.update(status='completed', step=trainer.state.global_step, frozen_base_sha256_after=after,
                   best_checkpoint=trainer.state.best_model_checkpoint, best_reward_accuracy=trainer.state.best_metric,
                   best_validation=selected_validation,
                   checkpoints={'best':checkpoint_manifest(output/'best_adapter'),'final':checkpoint_manifest(output/'final_adapter')})
        write_json(output/'run.json', run)
        print('PERSONA TRAINING COMPLETED '+json.dumps({'step':run['step'],'best_accuracy':run['best_reward_accuracy'],'reference_unchanged':True}),flush=True)
    except BaseException as exc:
        run.update(status='failed', error={'type':type(exc).__name__,'message':str(exc)})
        if trainer is not None:run['step']=trainer.state.global_step
        write_json(output/'run.json', run)
        raise


if __name__=='__main__':
    main()
