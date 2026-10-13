"""Manual 6.2 conversion, explicit opt-in; DPO remains strict single-turn."""

import copy

import pytest


def chat():
    return {"prompt_id": "raw-id", "prompt": "Original\n    code", "messages": [
        {"role": "user", "content": "Original\n    code"},
        {"role": "assistant", "content": "First answer\r\n    code"},
        {"role": "user", "content": "Follow-up"},
        {"role": "assistant", "content": "Later answer"}]}


def test_first_turn_conversion_is_explicit_preserves_text_id_and_order():
    from src.data import normalize_rows
    row = chat()
    before = copy.deepcopy(row)
    strict, strict_counts = normalize_rows([row], "sft", "ultrachat")
    assert strict == [] and strict_counts["multi_turn"] == 1
    rows, counts = normalize_rows([row], "sft", "ultrachat_first_turn")
    assert rows == [{"id": "raw-id", "source_index": 0, "prompt": row["prompt"],
                     "response": row["messages"][1]["content"]}]
    assert counts["kept"] == 1 and counts["multi_turn"] == 0 and row == before
    assert "Later answer" not in rows[0]["response"]


@pytest.mark.parametrize("case", ["system_first", "first_wrong_role", "odd_length",
    "later_wrong_role", "later_missing_content", "later_nonstr_content", "later_blank_content",
    "prompt_mismatch", "first_blank", "missing_pair"])
def test_first_turn_conversion_still_rejects_invalid_conversations(case):
    from src.data import normalize_rows
    row = chat()
    if case == "system_first": row["messages"][0]["role"] = "system"
    elif case == "first_wrong_role": row["messages"][1]["role"] = "user"
    elif case == "odd_length": row["messages"].pop()
    elif case == "later_wrong_role": row["messages"][2]["role"] = "assistant"
    elif case == "later_missing_content": del row["messages"][3]["content"]
    elif case == "later_nonstr_content": row["messages"][3]["content"] = None
    elif case == "later_blank_content": row["messages"][3]["content"] = " "
    elif case == "prompt_mismatch": row["prompt"] = "Different"
    elif case == "first_blank": row["messages"][1]["content"] = " "
    elif case == "missing_pair": row["messages"] = []
    rows, counts = normalize_rows([row], "sft", "ultrachat_first_turn")
    assert rows == [] and counts["input"] == 1 and counts["kept"] == 0
    reason = "empty_text" if case in {"later_blank_content", "first_blank"} else "prompt_mismatch" if case == "prompt_mismatch" else "invalid_record"
    assert counts[reason] == 1
    assert sum(counts[key] for key in counts if key not in {"input", "kept"}) == 1


def test_conversion_schema_is_not_available_for_dpo():
    from src.data import normalize_rows
    with pytest.raises(ValueError):
        normalize_rows([chat()], "dpo", "ultrachat_first_turn")


def test_converted_prompt_dedup_and_conflicting_id_remain_enforced():
    from src.data import normalize_rows
    first, second = chat(), chat()
    second["prompt_id"] = "second-id"
    rows, counts = normalize_rows([first, second], "sft", "ultrachat_first_turn")
    assert len(rows) == 1 and counts["duplicate_prompt"] == 1
    second["prompt_id"] = first["prompt_id"]
    second["messages"][1]["content"] = "Different first answer"
    with pytest.raises(ValueError, match="duplicate id"):
        normalize_rows([first, second], "sft", "ultrachat_first_turn")


def test_ultrafeedback_source_prompt_groups_allow_different_pairs_but_not_different_prompts():
    from src.data import normalize_rows
    first = {"prompt_id": "group", "prompt": "Q", "chosen": [
        {"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}],
        "rejected": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "B"}]}
    second = copy.deepcopy(first)
    second["chosen"][1]["content"] = "Different answer"
    rows, counts = normalize_rows([first, second], "dpo", "ultrafeedback")
    assert rows == [{"id": "group", "source_index": 0, "prompt": "Q", "chosen": "A", "rejected": "B"}]
    assert counts["duplicate_prompt"] == 1
    first["id"] = second["id"] = "explicit-row-id"
    with pytest.raises(ValueError, match="duplicate id"):
        normalize_rows([first, second], "dpo", "ultrafeedback")
    del first["id"], second["id"]
    second["prompt"] = second["chosen"][0]["content"] = second["rejected"][0]["content"] = "Other prompt"
    with pytest.raises(ValueError, match="duplicate id"):
        normalize_rows([first, second], "dpo", "ultrafeedback")
