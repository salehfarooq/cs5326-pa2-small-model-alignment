"""Independent schema, leakage and reproducibility checks for data preparation."""

import copy
import random

import pytest

from src.data import format_prompt, normalize_rows


def test_prompt_template_preserves_instruction_text():
    assert format_prompt("a {literal}\n    code") == "Instruction:\na {literal}\n    code\n\nResponse:\n"
    for bad in ["", "  ", None, 3]:
        with pytest.raises(ValueError):
            format_prompt(bad)


def test_sft_identity_and_nonmutating_normalization():
    raw = [{"id": "original", "prompt": "P", "response": " answer "},
           {"prompt": "Q", "response": "b"}]
    before = copy.deepcopy(raw)
    clean, counts = normalize_rows(raw, "sft")
    again, again_counts = normalize_rows(raw, "sft")
    assert clean == again and counts == again_counts
    assert raw == before
    assert clean[0]["id"] == "original" and clean[0]["source_index"] == 0
    assert clean[0]["response"] == " answer "
    assert isinstance(clean[1]["id"], str) and clean[1]["id"]
    assert counts["input"] == counts["kept"] == 2


def test_filter_counts_account_for_every_input():
    raw = [{"prompt": "P", "response": "a"},
           {"prompt": " P ", "response": "other"},
           {"prompt": "", "response": "b"},
           {"prompt": "Q", "response": 2}, None]
    clean, counts = normalize_rows(raw, "sft")
    assert len(clean) == counts["kept"] == 1
    assert counts["duplicate_prompt"] == 1
    assert counts["empty_text"] == 1
    assert counts["invalid_record"] == 2
    rejected = sum(counts[key] for key in ["invalid_record", "empty_text", "multi_turn",
                                         "prompt_mismatch", "identical_responses", "duplicate_prompt"])
    assert counts["input"] == counts["kept"] + rejected


def test_prompt_deduplication_normalizes_unicode_and_newlines_only():
    clean, counts = normalize_rows([
        {"prompt": "cafe\u0301\r\ncode", "response": "one"},
        {"prompt": "caf\u00e9\ncode", "response": "two"},
        {"prompt": "caf\u00e9\n    code", "response": "three"}], "sft")
    assert len(clean) == 2 and counts["duplicate_prompt"] == 1
    assert clean[0]["prompt"] == "cafe\u0301\r\ncode"
    assert clean[1]["source_index"] == 2


def test_id_collision_is_fatal_even_when_prompts_differ():
    with pytest.raises(ValueError, match="[Ii][Dd]"):
        normalize_rows([{"id": "same", "prompt": "P", "response": "a"},
                        {"id": "same", "prompt": "Q", "response": "b"}], "sft")


def test_duplicate_id_with_different_response_is_not_hidden_by_prompt_dedup():
    with pytest.raises(ValueError, match="[Ii][Dd]"):
        normalize_rows([{"id": "same", "prompt": "P", "response": "a"},
                        {"id": "same", "prompt": "P", "response": "b"}], "sft")


def test_dpo_filters_identical_normalized_answers():
    clean, counts = normalize_rows([
        {"prompt": "P", "chosen": "caf\u00e9", "rejected": " cafe\u0301 "},
        {"prompt": "Q", "chosen": "yes", "rejected": "no"}], "dpo")
    assert len(clean) == 1 and counts["identical_responses"] == 1
    assert clean[0]["chosen"] == "yes" and clean[0]["rejected"] == "no"


def test_ultrachat_accepts_single_turn_and_counts_multiturn():
    messages = [{"role": "user", "content": "Question"}, {"role": "assistant", "content": "Answer"}]
    raw = [{"prompt_id": "chat-original", "prompt": "Question", "messages": messages},
           {"messages": messages + messages}]
    clean, counts = normalize_rows(raw, "sft", "ultrachat")
    assert len(clean) == 1 and clean[0]["id"] == "chat-original"
    assert clean[0]["prompt"] == "Question" and clean[0]["response"] == "Answer"
    assert counts["multi_turn"] == 1


def test_ultrachat_mismatched_prompt_does_not_override_conversation():
    clean, counts = normalize_rows([{"prompt": "Different", "messages": [
        {"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]}], "sft", "ultrachat")
    assert not clean and counts["prompt_mismatch"] == 1


def test_ultrafeedback_both_sides_share_exact_prompt():
    def messages(prompt, answer):
        return [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
    clean, counts = normalize_rows([
        {"prompt": "Q", "chosen": messages("Q", "yes"), "rejected": messages("Q", "no")},
        {"prompt": "Q2", "chosen": messages("Q2", "yes"), "rejected": messages("wrong", "no")}],
        "dpo", "ultrafeedback")
    assert len(clean) == 1 and clean[0]["chosen"] == "yes"
    assert counts["prompt_mismatch"] == 1


def test_hh_single_turn_preserves_answer_and_inline_marker_text():
    clean, counts = normalize_rows([{
        "chosen": "\n\nHuman: Explain Human: as a label\n\nAssistant: useful\n  content ",
        "rejected": "\n\nHuman: Explain Human: as a label\n\nAssistant: wrong"}], "dpo", "hh")
    assert len(clean) == 1 and counts["kept"] == 1
    assert clean[0]["prompt"] == "Explain Human: as a label"
    assert clean[0]["chosen"] == "useful\n  content "


def test_hh_rejects_multiturn_and_mismatched_prefixes():
    clean, counts = normalize_rows([
        {"chosen": "\n\nHuman: Q\n\nAssistant: A\n\nHuman: Again\n\nAssistant: B",
         "rejected": "\n\nHuman: Q\n\nAssistant: no"},
        {"chosen": "\n\nHuman: Q\n\nAssistant: yes", "rejected": "\n\nHuman: Wrong\n\nAssistant: no"}],
        "dpo", "hh")
    assert not clean and counts["multi_turn"] == counts["prompt_mismatch"] == 1


def test_hh_does_not_accept_unanchored_transcript():
    clean, counts = normalize_rows([{"chosen": "prefix\n\nHuman: Q\n\nAssistant: yes",
                                    "rejected": "\n\nHuman: Q\n\nAssistant: no"}], "dpo", "hh")
    assert not clean and counts["invalid_record"] == 1


def test_eval_keeps_safety_metadata_without_response_leakage():
    clean, counts = normalize_rows([{"id": "e1", "instruction": "Q", "category": "benign",
                                    "expected_behavior": "Help", "output": "Do not copy baseline answer"}], "eval")
    assert clean[0]["instruction"] == "Q"
    assert clean[0]["category"] == "benign" and clean[0]["expected_behavior"] == "Help"
    assert "output" not in clean[0]


@pytest.mark.parametrize("kind,schema", [("bad", "normalized"), ("dpo", "ultrachat"), ("sft", "hh"), ("sft", "bad")])
def test_invalid_schema_configuration_rejected(kind, schema):
    with pytest.raises(ValueError):
        normalize_rows([], kind, schema)


def test_normalization_does_not_touch_global_random_state():
    state = random.getstate()
    normalize_rows([{"prompt": "P", "response": "a"}], "sft")
    assert random.getstate() == state


def test_generated_ids_cannot_collide_at_field_delimiters():
    clean, counts = normalize_rows([
        {"prompt": "a\nresponse=b", "response": "c"},
        {"prompt": "a", "response": "b\nresponse=c"}], "sft")
    assert len(clean) == 2
    assert clean[0]["id"] != clean[1]["id"]


def test_eval_metadata_with_invalid_type_is_not_silently_discarded():
    clean, counts = normalize_rows([{"instruction": "Q", "category": 5}], "eval")
    assert not clean and counts["invalid_record"] == 1


@pytest.mark.parametrize("kind,schema", [([], "normalized"), ("sft", []), (None, "normalized")])
def test_configuration_types_fail_cleanly(kind, schema):
    with pytest.raises(ValueError):
        normalize_rows([], kind, schema)


def prepared_rows(count):
    return [{"id": f"row-{i}", "source_index": i, "prompt": f"Q{i}", "response": "A"}
            for i in range(count)]


def test_splits_are_exact_disjoint_reproducible_and_nonmutating():
    from src.data import split_rows
    rows = prepared_rows(12)
    before = copy.deepcopy(rows)
    state = random.getstate()
    train, validation = split_rows(rows, 7, 3, 42)
    assert len(train) == 7 and len(validation) == 3
    assert not ({r["id"] for r in train} & {r["id"] for r in validation})
    assert (train, validation) == split_rows(rows, 7, 3, 42)
    assert (train, validation) != split_rows(rows, 7, 3, 43)
    assert rows == before and random.getstate() == state


@pytest.mark.parametrize("train,val,seed", [(0, 1, 42), (2, 0, 42), (-1, 1, 42),
                                            (True, 1, 42), (1.5, 1, 42), (2, 1, True), (2, 1, 0.5)])
def test_invalid_split_configuration_rejected(train, val, seed):
    from src.data import split_rows
    with pytest.raises(ValueError):
        split_rows(prepared_rows(5), train, val, seed)


def test_insufficient_data_fails_without_repeating_or_shrinking():
    from src.data import split_rows, select_eval
    with pytest.raises(ValueError):
        split_rows(prepared_rows(4), 3, 2)
    with pytest.raises(ValueError):
        select_eval([{"id": "e", "instruction": "Q"}], 2)


def test_split_revalidates_prompt_and_id_uniqueness():
    from src.data import split_rows
    rows = prepared_rows(3)
    rows[1]["prompt"] = " Q0 "
    with pytest.raises(ValueError):
        split_rows(rows, 1, 1)
    rows = prepared_rows(3)
    rows[1]["id"] = rows[0]["id"]
    with pytest.raises(ValueError):
        split_rows(rows, 1, 1)


def test_fixed_eval_sampling_retains_metadata_and_global_rng():
    from src.data import select_eval
    rows = [{"id": f"e-{i}", "instruction": f"Q{i}", "category": "benign"} for i in range(10)]
    before = copy.deepcopy(rows)
    state = random.getstate()
    selected = select_eval(rows, 4, 42)
    assert len(selected) == 4 and len({r["id"] for r in selected}) == 4
    assert selected == select_eval(rows, 4, 42)
    assert all(r["category"] == "benign" for r in selected)
    assert rows == before and random.getstate() == state


def test_eval_sampler_validates_instruction_even_when_prompt_exists():
    from src.data import select_eval
    with pytest.raises(ValueError):
        select_eval([{"id": "a", "prompt": "P"}], 1)
    with pytest.raises(ValueError):
        select_eval([{"id": "a", "prompt": "P", "instruction": "Q"},
                     {"id": "b", "prompt": "Other", "instruction": " Q "}], 1)


def test_training_split_requires_prompt_not_eval_instruction():
    from src.data import split_rows
    with pytest.raises(ValueError):
        split_rows([{"id": "a", "instruction": "Q"},
                    {"id": "b", "instruction": "Other"}], 1, 1)


@pytest.mark.parametrize("kind,schema", [("sft", "ultrachat"), ("dpo", "ultrafeedback")])
def test_present_null_prompt_is_mismatch(kind, schema):
    user = {"role": "user", "content": "Q"}
    good = {"role": "assistant", "content": "yes"}
    bad = {"role": "assistant", "content": "no"}
    row = {"prompt": None, "messages": [user, good],
           "chosen": [user, good], "rejected": [user, bad]}
    clean, counts = normalize_rows([row], kind, schema)
    assert not clean and counts["prompt_mismatch"] == 1


def test_official_fixed_eval_bank_is_preserved_exactly():
    import json
    from pathlib import Path
    from src.data import validate_fixed_eval
    filename = Path(__file__).resolve().parents[1] / "eval_sets/hh_rlhf_eval_300_seed42.json"
    rows = json.loads(filename.read_text())
    before = copy.deepcopy(rows)
    state = random.getstate()
    result = validate_fixed_eval(rows)
    assert len(result) == 300 and result == rows == before
    assert len({r["id"] for r in result}) == 300
    assert result is not rows and all(a is not b for a, b in zip(result, rows))
    assert random.getstate() == state


def test_fixed_eval_repeated_instructions_with_distinct_ids_are_retained():
    from src.data import validate_fixed_eval
    rows = [{"id": "a", "instruction": "Q", "dataset": "fixed"},
            {"id": "b", "instruction": "Q", "category": "safety"}]
    assert validate_fixed_eval(rows) == rows


@pytest.mark.parametrize("rows", [[], None, [{"instruction": "Q"}],
    [{"id": "a", "instruction": " "}], [{"id": "a", "instruction": "Q", "category": None}],
    [{"id": "a", "instruction": "Q"}, {"id": "a", "instruction": "Other"}]])
def test_fixed_eval_malformed_rows_fail_instead_of_silent_filtering(rows):
    from src.data import validate_fixed_eval
    with pytest.raises(ValueError):
        validate_fixed_eval(rows)
