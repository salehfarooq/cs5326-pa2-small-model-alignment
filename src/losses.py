"""Response-only log-probability scoring and SFT/DPO losses.

Scoring uses the causal shift: logits[:, :-1] predict labels[:, 1:], so the
first token of the sequence (e.g. the first response token when the prompt is
empty) has no predecessor and is never scored. Response log-probabilities are
summed per row. SFT uses a token-mean reduction: the sum of all response
log-probs divided by the number of scoreable response labels. DPO uses
-logsigmoid(beta * (policy_gap - reference_gap)).mean().
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from src.batching import build_sft_batch


def _model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        pass
    try:
        return next(model.buffers()).device
    except StopIteration:
        return torch.device("cpu")


def _forward_logits(model, input_ids, attention_mask):
    device = _model_device(model)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    if hasattr(output, "logits"):
        logits = output.logits
    else:
        logits = output["logits"]
    if not torch.isfinite(logits).all():
        raise ValueError("model logits must be finite; got non-finite values from the model")
    return logits


def _promote_scores(logps):
    if logps.dtype in (torch.float16, torch.bfloat16):
        return logps.float()
    return logps


def _shift_logprobs(logits, labels):
    labels = labels.to(logits.device)
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    mask = shift_labels.ne(-100)
    safe_labels = shift_labels.clamp_min(0)
    token_logps = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logps = token_logps.masked_fill(~mask, 0.0)
    return token_logps, mask


def _score_rows(model, tokenizer, prompts, responses, max_length):
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    labels = batch["labels"]
    scoreable_per_row = labels[:, 1:].ne(-100).sum(dim=-1)
    if int(scoreable_per_row.min()) == 0:
        raise ValueError(
            "each row must have at least one scoreable response token; "
            "a row with an empty prompt and empty response (or a response "
            "fully truncated away) has none"
        )
    logits = _forward_logits(model, batch["input_ids"], batch["attention_mask"])
    return _shift_logprobs(logits, labels)


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    """Return one summed response-token log-probability per input row."""
    token_logps, mask = _score_rows(model, tokenizer, prompts, responses, max_length)
    return token_logps.sum(dim=-1)


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    """Return a scalar response-only SFT loss using token-mean reduction."""
    token_logps, mask = _score_rows(model, tokenizer, prompts, responses, max_length)
    total_tokens = int(mask.sum())
    return -token_logps.sum() / total_tokens


def compute_dpo_loss_from_logps(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta,
):
    """Return the scalar mean DPO objective from per-example log probabilities."""
    logp_groups = [
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
    ]
    for logps in logp_groups:
        if not isinstance(logps, torch.Tensor):
            raise ValueError("all log-probability inputs must be torch tensors")
        if logps.ndim != 1:
            raise ValueError(f"log-probability inputs must be one-dimensional, got shape {tuple(logps.shape)}")
        if logps.numel() == 0:
            raise ValueError("log-probability inputs must be non-empty")
        if not torch.isfinite(logps).all():
            raise ValueError("log-probability inputs must be finite")
    shape = tuple(policy_chosen_logps.shape)
    for logps in logp_groups[1:]:
        if tuple(logps.shape) != shape:
            raise ValueError(f"log-probability inputs must share one shape, got {shape} and {tuple(logps.shape)}")
    if isinstance(beta, bool) or not isinstance(beta, (int, float)) or not math.isfinite(beta) or beta <= 0:
        raise ValueError(f"beta must be a finite positive number, got {beta!r}")

    policy_chosen_logps = _promote_scores(policy_chosen_logps)
    policy_rejected_logps = _promote_scores(policy_rejected_logps)
    reference_chosen_logps = _promote_scores(reference_chosen_logps)
    reference_rejected_logps = _promote_scores(reference_rejected_logps)

    policy_gap = policy_chosen_logps - policy_rejected_logps
    reference_gap = reference_chosen_logps - reference_rejected_logps
    return -F.logsigmoid(beta * (policy_gap - reference_gap)).mean()


def compute_dpo_loss(
    policy_model,
    reference_model,
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    beta,
    max_length,
):
    """Return a scalar mean DPO loss with a frozen reference computation."""
    policy_chosen_logps = compute_response_logprobs(policy_model, tokenizer, prompts, chosen_responses, max_length)
    policy_rejected_logps = compute_response_logprobs(policy_model, tokenizer, prompts, rejected_responses, max_length)

    with torch.no_grad():
        reference_chosen_logps = compute_response_logprobs(reference_model, tokenizer, prompts, chosen_responses, max_length)
        reference_rejected_logps = compute_response_logprobs(reference_model, tokenizer, prompts, rejected_responses, max_length)

    policy_device = policy_chosen_logps.device
    reference_chosen_logps = reference_chosen_logps.to(policy_device)
    reference_rejected_logps = reference_rejected_logps.to(policy_device)

    return compute_dpo_loss_from_logps(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
        beta,
    )
