"""Independent review checks, beyond the supplied public smoke tests."""

import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import submission_adapter as adapter
from tests.toy_lm import ToyCausalLM, ToyTokenizer


def test_eos_equal_to_pad_is_still_a_response_target():
    tokenizer = ToyTokenizer()
    tokenizer.pad_token_id = tokenizer.eos_token_id
    batch = adapter.build_sft_batch(tokenizer, ["P:", "Long:"], ["a", "bc"], 20)
    assert batch["labels"][0, 3].item() == tokenizer.eos_token_id
    assert batch["attention_mask"][0, 3].item() == 1
    assert batch["labels"][0, 4:].eq(-100).all()
    assert batch["attention_mask"][0, 4:].eq(0).all()


def test_eos_is_appended_before_truncation_and_rows_stay_in_order():
    tokenizer = ToyTokenizer()
    batch = adapter.build_sft_batch(tokenizer, ["P:", "Q:"], ["abcd", "z"], 5)
    assert batch["input_ids"][0].tolist() == tokenizer.encode("P:abc")
    assert batch["labels"][0].tolist() == [-100, -100] + tokenizer.encode("abc")
    assert batch["input_ids"][1, :4].tolist() == tokenizer.encode("Q:z") + [1]
    assert batch["labels"][1, 3].item() == 1
    assert batch["labels"][1, 4].item() == -100


def test_prompt_tokens_are_context_not_padding():
    tokenizer = ToyTokenizer()
    batch = adapter.build_sft_batch(tokenizer, ["abc:"], ["x"], 10)
    assert batch["labels"][0, :4].eq(-100).all()
    assert batch["attention_mask"][0, :4].eq(1).all()


def test_tokenizer_without_special_tokens_is_not_mutated():
    tokenizer = ToyTokenizer()
    tokenizer.eos_token_id = None
    tokenizer.pad_token_id = None
    batch = adapter.build_sft_batch(tokenizer, ["P:", "Q:"], ["a", "bc"], 10)
    assert batch["labels"][0, 2].item() == tokenizer.vocab["a"]
    assert batch["labels"][0, 3].item() == -100
    assert tokenizer.eos_token_id is None and tokenizer.pad_token_id is None


class PositionLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.bias = nn.Parameter(torch.linspace(-0.7, 0.9, vocab_size))

    def forward(self, input_ids, attention_mask=None):
        # Each position has distinct logits, exposing causal-shift errors.
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        factors = positions.float().add(1).view(1, -1, 1)
        logits = self.bias.view(1, 1, -1) * factors
        return SimpleNamespace(logits=logits.expand(input_ids.shape[0], -1, -1))


def test_response_scores_match_hand_selected_next_token_distributions():
    tokenizer = ToyTokenizer()
    model = PositionLM(tokenizer.vocab_size)
    scores = adapter.compute_response_logprobs(model, tokenizer, ["P:"], ["ab"], 8)
    targets = tokenizer.encode("ab") + [tokenizer.eos_token_id]
    expected = torch.tensor(0.0)
    for position, target in zip([1, 2, 3], targets):
        probabilities = torch.softmax(model.bias * (position + 1), dim=0)
        expected = expected + probabilities[target].log()
    torch.testing.assert_close(scores[0], expected)
    scores.sum().backward()
    assert model.bias.grad is not None and model.bias.grad.abs().sum() > 0


def test_sft_token_mean_weights_different_response_lengths():
    tokenizer = ToyTokenizer()
    model = PositionLM(tokenizer.vocab_size)
    scores = adapter.compute_response_logprobs(model, tokenizer, ["P:", "Q:"], ["a", "bcd"], 12)
    loss = adapter.compute_sft_loss(model, tokenizer, ["P:", "Q:"], ["a", "bcd"], 12)
    # Two targets in first row and four in second, including EOS.
    torch.testing.assert_close(loss, -scores.sum() / 6)


def test_scores_do_not_change_when_extra_padding_is_added():
    tokenizer = ToyTokenizer()
    model = PositionLM(tokenizer.vocab_size)
    alone = adapter.compute_response_logprobs(model, tokenizer, ["P:"], ["a"], 20)
    mixed = adapter.compute_response_logprobs(model, tokenizer, ["P:", "Q:"], ["a", "bcdef"], 20)
    torch.testing.assert_close(alone[0], mixed[0])


def test_empty_prompt_first_response_token_is_unscored():
    tokenizer = ToyTokenizer()
    model = PositionLM(tokenizer.vocab_size)
    score = adapter.compute_response_logprobs(model, tokenizer, [""], ["ab"], 8)
    targets = [tokenizer.vocab["b"], tokenizer.eos_token_id]
    expected = sum(torch.softmax(model.bias * (i + 1), dim=0)[target].log()
                   for i, target in enumerate(targets))
    torch.testing.assert_close(score[0], expected)


def test_zero_scored_response_is_rejected_instead_of_silent_zero_loss():
    tokenizer = ToyTokenizer()
    model = ToyCausalLM(tokenizer.vocab_size)
    batch = adapter.build_sft_batch(tokenizer, ["LongPrompt"], ["a"], 3)
    assert batch["labels"].eq(-100).all()
    with pytest.raises(ValueError):
        adapter.compute_response_logprobs(model, tokenizer, ["LongPrompt"], ["a"], 3)


@pytest.mark.parametrize("length", [0, -1, 1.5, True])
def test_invalid_context_length_rejected(length):
    with pytest.raises(ValueError):
        adapter.build_sft_batch(ToyTokenizer(), ["P"], ["a"], length)


def test_mismatched_pair_lengths_rejected():
    with pytest.raises(ValueError):
        adapter.build_dpo_batch(ToyTokenizer(), ["P"], ["a", "b"], ["c"], 10)


def test_bare_string_is_not_treated_as_a_batch_of_characters():
    with pytest.raises(ValueError):
        adapter.build_sft_batch(ToyTokenizer(), "P:", "ab", 10)


def test_empty_input_batch_rejected():
    with pytest.raises(ValueError):
        adapter.build_sft_batch(ToyTokenizer(), [], [], 10)


def test_non_string_row_rejected():
    with pytest.raises(ValueError):
        adapter.build_sft_batch(ToyTokenizer(), ["P:"], [42], 10)


def test_dpo_numerical_values_and_gradient_direction():
    chosen = torch.tensor([-1.0, -2.0], requires_grad=True)
    rejected = torch.tensor([-2.0, -1.0], requires_grad=True)
    ref_chosen = torch.tensor([-2.0, -2.0])
    ref_rejected = torch.tensor([-2.0, -2.0])
    loss = adapter.compute_dpo_loss_from_logps(chosen, rejected, ref_chosen, ref_rejected, 0.1)
    expected = (math.log1p(math.exp(-0.1)) + math.log1p(math.exp(0.1))) / 2
    assert loss.item() == pytest.approx(expected, abs=1e-6)
    loss.backward()
    assert chosen.grad.lt(0).all()
    assert rejected.grad.gt(0).all()


def test_extreme_dpo_logits_are_finite():
    loss = adapter.compute_dpo_loss_from_logps(
        torch.tensor([10000.0, -10000.0]), torch.zeros(2), torch.zeros(2), torch.zeros(2), 1.0)
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(5000.0)


@pytest.mark.parametrize("beta", [0, -0.1, float("nan"), float("inf")])
def test_invalid_beta_rejected(beta):
    with pytest.raises(ValueError):
        adapter.compute_dpo_loss_from_logps(*(torch.zeros(2) for _ in range(4)), beta)


def test_dpo_tensor_shapes_do_not_broadcast_silently():
    with pytest.raises(ValueError):
        adapter.compute_dpo_loss_from_logps(torch.zeros(2), torch.zeros(1), torch.zeros(2), torch.zeros(2), 0.1)


def test_nonfinite_log_probabilities_rejected():
    with pytest.raises(ValueError):
        adapter.compute_dpo_loss_from_logps(torch.tensor([float("nan")]), torch.zeros(1), torch.zeros(1), torch.zeros(1), 0.1)


def test_dpo_scoring_preserves_caller_model_modes():
    tokenizer = ToyTokenizer()
    policy = ToyCausalLM(tokenizer.vocab_size)
    reference = ToyCausalLM(tokenizer.vocab_size)
    policy.eval()
    reference.train()
    adapter.compute_dpo_loss(policy, reference, tokenizer, ["P:"], ["a"], ["b"], 0.1, 10)
    assert policy.training is False
    assert reference.training is True


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_accelerator_policy_with_cpu_reference():
    tokenizer = ToyTokenizer()
    policy = PositionLM(tokenizer.vocab_size).to("mps")
    reference = PositionLM(tokenizer.vocab_size)
    loss = adapter.compute_dpo_loss(policy, reference, tokenizer, ["P:"], ["a"], ["b"], 0.1, 10)
    assert loss.device.type == "mps"
    assert torch.isfinite(loss)
    loss.backward()
    assert policy.bias.grad is not None and policy.bias.grad.abs().sum() > 0
    assert reference.bias.grad is None


def test_policy_step_changes_policy_and_preserves_reference():
    tokenizer = ToyTokenizer()
    policy = ToyCausalLM(tokenizer.vocab_size)
    reference = ToyCausalLM(tokenizer.vocab_size)
    before_policy = policy.logit_bias.detach().clone()
    before_reference = reference.logit_bias.detach().clone()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    loss = adapter.compute_dpo_loss(policy, reference, tokenizer, ["P:"], ["a"], ["b"], 0.1, 10)
    loss.backward()
    optimizer.step()
    assert not torch.equal(policy.logit_bias, before_policy)
    assert torch.equal(reference.logit_bias, before_reference)
    assert reference.logit_bias.grad is None
