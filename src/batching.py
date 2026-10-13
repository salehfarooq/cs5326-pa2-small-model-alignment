"""Batch construction for SFT and DPO.

Boundary policy: prompt and response are tokenized separately with
add_special_tokens=False and concatenated directly with no separator token.
EOS is appended to the response when the tokenizer defines one, then the
concatenated sequence is right-truncated to max_length (keep the left part,
so EOS can be truncated out). Rows are right-padded to the longest retained
sequence. Prompt and padding positions get labels=-100; real prompt tokens get
attention_mask=1 while padding gets attention_mask=0. Masking is by position,
never by token ID equality, so EOS==PAD is safe. Row containers must be a list
or tuple of strings; mappings, None, generators, and bare str/bytes are rejected.
"""

from __future__ import annotations

import torch


def _check_max_length(max_length):
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length <= 0:
        raise ValueError(f"max_length must be a positive integer, got {max_length!r}")


def _check_rows(*row_groups):
    for rows in row_groups:
        if not isinstance(rows, (list, tuple)):
            raise ValueError(
                "input rows must be a list or tuple of strings, got "
                f"{type(rows).__name__}"
            )
    lengths = [len(rows) for rows in row_groups]
    if lengths[0] == 0:
        raise ValueError("input row lists must be non-empty")
    if any(length != lengths[0] for length in lengths[1:]):
        raise ValueError(f"input row lists must have equal lengths, got {lengths}")
    for rows in row_groups:
        for row in rows:
            if not isinstance(row, str):
                raise ValueError(f"every row must be a string, got {type(row).__name__}")


def _pad_id_for(tokenizer):
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is None:
        pad_id = 0
    return pad_id


def _encode_pair(tokenizer, prompt, response, max_length):
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        response_ids = response_ids + [eos_id]
    token_ids = (prompt_ids + response_ids)[:max_length]
    prompt_length = min(len(prompt_ids), len(token_ids))
    return token_ids, prompt_length


def build_sft_batch(tokenizer, prompts, responses, max_length):
    """Return input_ids, attention_mask, and response-only labels."""
    _check_max_length(max_length)
    _check_rows(prompts, responses)

    token_ids_list = []
    prompt_lengths = []
    for prompt, response in zip(prompts, responses):
        token_ids, prompt_length = _encode_pair(tokenizer, prompt, response, max_length)
        token_ids_list.append(token_ids)
        prompt_lengths.append(prompt_length)

    width = max(len(token_ids) for token_ids in token_ids_list)
    pad_id = _pad_id_for(tokenizer)
    batch_size = len(token_ids_list)

    input_ids = torch.full((batch_size, width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, width), dtype=torch.long)
    labels = torch.full((batch_size, width), -100, dtype=torch.long)

    for row in range(batch_size):
        token_ids = token_ids_list[row]
        length = len(token_ids)
        ids_tensor = torch.tensor(token_ids, dtype=torch.long)
        input_ids[row, :length] = ids_tensor
        attention_mask[row, :length] = 1
        labels[row, :length] = ids_tensor
        labels[row, : prompt_lengths[row]] = -100

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    """Return separate response-masked LM batches for chosen and rejected rows."""
    _check_max_length(max_length)
    _check_rows(prompts, chosen_responses, rejected_responses)

    chosen_batch = build_sft_batch(tokenizer, prompts, chosen_responses, max_length)
    rejected_batch = build_sft_batch(tokenizer, prompts, rejected_responses, max_length)
    return {"chosen": chosen_batch, "rejected": rejected_batch}
