"""Fresh adversarial checks against numerical and input assumptions."""

import random
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import submission_adapter as adapter
from tests.toy_lm import ToyTokenizer


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_preference_arithmetic_is_promoted(dtype):
    chosen = torch.tensor([40000.0], dtype=dtype, requires_grad=True)
    rejected = torch.tensor([-40000.0], dtype=dtype, requires_grad=True)
    loss = adapter.compute_dpo_loss_from_logps(chosen, rejected, chosen.detach(), rejected.detach(), 0.1)
    assert loss.dtype == torch.float32
    torch.testing.assert_close(loss, torch.log(torch.tensor(2.0)))
    loss.backward()
    assert torch.isfinite(chosen.grad).all() and chosen.grad.lt(0).all()
    assert torch.isfinite(rejected.grad).all() and rejected.grad.gt(0).all()


class ContextLM(nn.Module):
    def __init__(self, vocab_size, dtype=torch.float64):
        super().__init__()
        generator = torch.Generator().manual_seed(419)
        self.weight = nn.Parameter(torch.randn(vocab_size, vocab_size, generator=generator, dtype=dtype))

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(logits=self.weight[input_ids])


def test_scoring_preserves_float64_precision():
    tokenizer = ToyTokenizer()
    model = ContextLM(tokenizer.vocab_size)
    scores = adapter.compute_response_logprobs(model, tokenizer, ["P:"], ["abc"], 20)
    ids = torch.tensor(tokenizer.encode("P:abc") + [tokenizer.eos_token_id])
    expected = -F.cross_entropy(model.weight[ids[:-1]][1:], ids[2:], reduction="sum")
    assert scores.dtype == torch.float64
    torch.testing.assert_close(scores[0], expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("bad_rows", [None, {"a": "b"}, (x for x in ["a"])])
def test_non_sequence_row_containers_fail_cleanly(bad_rows):
    with pytest.raises(ValueError):
        adapter.build_sft_batch(ToyTokenizer(), bad_rows, ["x"], 10)


def test_nan_model_output_is_rejected_at_scoring_boundary():
    tokenizer = ToyTokenizer()
    model = ContextLM(tokenizer.vocab_size, torch.float32)
    with torch.no_grad():
        model.weight.fill_(float("nan"))
    with pytest.raises(ValueError, match="finite"):
        adapter.compute_sft_loss(model, tokenizer, ["P:"], ["a"], 10)


class BufferLM(nn.Module):
    def __init__(self, vocab_size, device):
        super().__init__()
        self.register_buffer("table", torch.zeros(vocab_size, vocab_size, device=device))

    def forward(self, input_ids, attention_mask=None):
        assert input_ids.device == self.table.device
        return {"logits": self.table[input_ids]}


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_parameterless_model_uses_buffer_device():
    tokenizer = ToyTokenizer()
    model = BufferLM(tokenizer.vocab_size, "mps")
    score = adapter.compute_response_logprobs(model, tokenizer, ["P:"], ["a"], 10)
    assert score.device.type == "mps"
    torch.testing.assert_close(score.cpu(), torch.tensor([-2 * torch.log(torch.tensor(float(tokenizer.vocab_size)))]))


def test_randomized_response_scores_match_independent_cross_entropy():
    tokenizer = ToyTokenizer()
    model = ContextLM(tokenizer.vocab_size, torch.float32)
    rng = random.Random(749)
    for _ in range(30):
        prompts, responses = [], []
        for _ in range(rng.randrange(1, 5)):
            prompts.append("".join(rng.choices("PQ:", k=rng.randrange(1, 5))))
            responses.append("".join(rng.choices("abc", k=rng.randrange(0, 6))))
        scores = adapter.compute_response_logprobs(model, tokenizer, prompts, responses, 20)
        expected = []
        for prompt, response in zip(prompts, responses):
            prompt_ids = tokenizer.encode(prompt)
            ids = prompt_ids + tokenizer.encode(response) + [tokenizer.eos_token_id]
            inputs = torch.tensor(ids[:-1])
            targets = torch.tensor(ids[1:])
            start = len(prompt_ids) - 1
            expected.append(-F.cross_entropy(model.weight[inputs][start:], targets[start:], reduction="sum"))
        torch.testing.assert_close(scores, torch.stack(expected), rtol=1e-5, atol=1e-6)


def test_empty_response_can_train_eos_when_prompt_is_present():
    tokenizer = ToyTokenizer()
    model = ContextLM(tokenizer.vocab_size, torch.float32)
    score = adapter.compute_response_logprobs(model, tokenizer, ["P:"], [""], 10)
    expected = torch.log_softmax(model.weight[tokenizer.vocab[":"]], -1)[tokenizer.eos_token_id]
    torch.testing.assert_close(score[0], expected)
