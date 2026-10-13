# Persona LoRA DPO recipe and rationale

Manual Task5 and section6.4: SmolLM2-360M-Instruct, conversational supplied preferences, TRL DPOTrainer, PEFT LoRA; separately evaluate cooking, coding, general help and safety. This is a student-selected recipe, not a manual-specified hyperparameter table.

Model revision: a10cc1512eabd3dde888204e902eca88bddb4951, verified via Hugging Face model metadata. Original220 pairs, deterministic198/22 prompt-disjoint split, seed42; no filtering. Native chat template; no persona system prompt. Maximum observed full prompt+response length123 tokens, context1024, no truncation needed. Frozen40 authored evaluation instructions, ten per manual category, disjoint from the persona preference bank and both fixed general evaluation banks.

Rank8, alpha16, query/value projections only: small adapter capacity for220 pairs; standard2x scaling; avoids changing all transformer projections. Dropout0 so preference/reference comparisons remain deterministic. One epoch and1e-5 fixed LR: conservative first adapter experiment rather than prolonged optimization on a small dataset. AdamW, weight decay0, clip1, per-device2 accumulation8 effective16, beta0.1, FP32, GPU0. No quantization; model fits T4. Validation every5 updates plus terminal validation, select best reward accuracy, retain final separately. Baseline/reference is the original SmolLM2 instruction checkpoint with adapters disabled. Hash all non-LoRA weights before/after to verify invariance. No extra SFT.

Decoding: native chat prompt, greedy256 new tokens, repetition penalty1.1, ngram block0. Identical baseline and adapter settings. Evaluations do not use a Ramsay system instruction; style should come from the adapter. No tuning based on previous final Qwen judge scores.

Sources consulted: https://huggingface.co/docs/trl/v0.19.1/en/dpo_trainer and https://huggingface.co/docs/peft/package_reference/lora. Pinned reproducibility stack: TRL0.19.1, transformers4.51.3, PEFT0.15.2, datasets3.6.0, accelerate1.7.0. Runtime must verify actual versions.

Pinned Transformers4.51.3 floors epoch-derived accumulation steps. Explicit max_steps13 retains the partial batch for198 rows / batch2 / accumulation8. Require runtime epoch1.0 and13 updates; otherwise mark failed. This is a coverage correction, not added epochs. Source: https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/trainer.py

Native-template clarification: no explicit system message is supplied, but the pinned SmolLM2 template automatically inserts its default system text: You are a helpful AI assistant named SmolLM, trained by Hugging Face. No Ramsay/persona system instruction is injected. The generation field native_chat_template_no_system means no caller-supplied system message, not absence of the template default. Preserve that interpretation in reporting.
