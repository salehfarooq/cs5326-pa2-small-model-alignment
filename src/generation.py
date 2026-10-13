"""Small greedy development checks; shared settings for model comparisons."""

import math
from collections import Counter

import torch

from src.data import format_prompt
from src.losses import _model_device

MAIN_DECODING = {'max_new_tokens': 256, 'repetition_penalty': 1.05, 'no_repeat_ngram_size': 4}


def repetition_stats(tokens, n=3):
    grams = [tuple(tokens[i:i+n]) for i in range(max(0, len(tokens)-n+1))]
    counts = Counter(grams)
    repeated = sum(count - 1 for count in counts.values())
    return {'ngram_size': n, 'ngram_count': len(grams), 'repeated_ngrams': repeated,
            'repeated_ngram_fraction': repeated / len(grams) if grams else 0.0}


def generate_development(model, tokenizer, prompts, max_new_tokens=128, repetition_penalty=1.0,
                         no_repeat_ngram_size=0):
    """Full instructions, greedy decoding, new tokens only; restore model mode.

    Repetition metrics are diagnostics, not automatic judgments of quality.
    Penalties must be selected on development prompts and shared with SFT/DPO.
    """
    if (not isinstance(prompts, (list, tuple)) or not prompts or
            any(not isinstance(p, str) or not p.strip() for p in prompts)):
        raise ValueError('development prompts must be nonblank strings')
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
        raise ValueError('max_new_tokens must be positive')
    if (isinstance(repetition_penalty, bool) or not isinstance(repetition_penalty, (float, int)) or
            not math.isfinite(repetition_penalty) or repetition_penalty < 1):
        raise ValueError('repetition_penalty must be finite and at least 1')
    if isinstance(no_repeat_ngram_size, bool) or not isinstance(no_repeat_ngram_size, int) or no_repeat_ngram_size < 0:
        raise ValueError('no_repeat_ngram_size must be nonnegative')
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    settings = {'do_sample': False, 'max_new_tokens': max_new_tokens, 'repetition_penalty': repetition_penalty,
                'no_repeat_ngram_size': no_repeat_ngram_size, 'eos_token_id': eos, 'pad_token_id': pad}
    mode, rows = model.training, []
    try:
        model.eval()
        with torch.no_grad():
            for prompt in prompts:
                ids = tokenizer.encode(format_prompt(prompt), add_special_tokens=False)
                if not ids:
                    raise ValueError('formatted prompt is empty after tokenization')
                context_limit = getattr(getattr(model, 'config', None), 'max_position_embeddings', None)
                if context_limit is not None and len(ids) + max_new_tokens > context_limit:
                    raise ValueError('full development prompt plus response exceeds model context')
                inputs = torch.tensor([ids], dtype=torch.long, device=_model_device(model))
                outputs = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                         use_cache=True, **settings)
                if outputs.ndim != 2 or outputs.shape[0] != 1 or not torch.equal(outputs[0, :len(ids)], inputs[0]):
                    raise ValueError('generation did not preserve its input prefix')
                new = outputs[0, len(ids):].tolist()
                hit_eos = eos is not None and eos in new
                if hit_eos:
                    new = new[:new.index(eos)]
                text = tokenizer.decode(new, skip_special_tokens=True)
                rows.append({'instruction': prompt, 'output': text, 'prompt_tokens': len(ids),
                             'response_tokens': len(new), 'empty_output': not text.strip(),
                             'finish_reason': 'eos' if hit_eos else 'length' if len(new) >= max_new_tokens else 'other',
                             'settings': dict(settings), **repetition_stats(new)})
        return rows
    finally:
        model.train(mode)
