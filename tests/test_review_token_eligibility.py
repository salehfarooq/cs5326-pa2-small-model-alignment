"""Offline context eligibility, invariant to batch size and tied to real adapter masks."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from src.data import format_prompt
from src.data_tokens import filter_for_context
from submission_adapter import build_sft_batch, build_dpo_batch


class CharTokenizer:
    eos_token_id = 997
    pad_token_id = 998
    def __call__(self, texts, add_special_tokens=False, truncation=False, max_length=None):
        assert add_special_tokens is False
        rows = [[ord(char) for char in text] if text != "unencodable" else [] for text in texts]
        if truncation:
            rows = [row[:max_length] for row in rows]
        return {"input_ids": rows}
    def encode(self, text, add_special_tokens=False):
        return self([text], add_special_tokens=add_special_tokens)["input_ids"][0]


def test_context_filter_removes_full_long_prompt_and_keeps_original_rows():
    rows = [{"id": "keep", "source_index": 4, "prompt": "Q", "response": "A"},
            {"id": "long", "source_index": 5, "prompt": "long prompt", "response": "A"},
            {"id": "empty", "source_index": 6, "prompt": "Q", "response": "unencodable"}]
    before = copy.deepcopy(rows)
    clean, counts = filter_for_context(rows, "sft", CharTokenizer(), max_length=40, max_prompt_tokens=30, batch_size=2)
    assert clean == [rows[0]] and clean[0] is rows[0] and rows == before
    assert counts == {"input": 3, "kept": 1, "prompt_too_long": 1,
                      "empty_tokenized_response": 1, "identical_retained_pair": 0}
    batch = build_sft_batch(CharTokenizer(), [format_prompt(clean[0]["prompt"])], [clean[0]["response"]], 40)
    assert (batch["labels"][:, 1:] != -100).sum().item() > 0


def test_dpo_filter_detects_identical_actual_retained_tokens_and_eos_order():
    capacity = 40 - len(format_prompt("Q"))
    rows = [{"id": "signal", "prompt": "Q", "chosen": "A", "rejected": "B"},
            {"id": "lost", "prompt": "Q", "chosen": "x" * capacity + "A", "rejected": "x" * capacity + "B"},
            {"id": "eos", "prompt": "Q", "chosen": "x" * (capacity - 1), "rejected": "x" * capacity}]
    clean, counts = filter_for_context(rows, "dpo", CharTokenizer(), max_length=40, max_prompt_tokens=30)
    assert [row["id"] for row in clean] == ["signal", "eos"] and counts["identical_retained_pair"] == 1
    batch = build_dpo_batch(CharTokenizer(), [format_prompt(row["prompt"]) for row in clean],
                            [row["chosen"] for row in clean], [row["rejected"] for row in clean], 40)
    for index in range(len(clean)):
        chosen = batch["chosen"]
        rejected = batch["rejected"]
        chosen_ids = chosen["input_ids"][index][chosen["labels"][index] != -100].tolist()
        rejected_ids = rejected["input_ids"][index][rejected["labels"][index] != -100].tolist()
        assert chosen_ids != rejected_ids


def test_token_filter_batch_size_does_not_change_rows_or_counts():
    rows = [{"id": str(i), "prompt": "Q", "chosen": f"A{i}", "rejected": f"B{i}"} for i in range(11)]
    expected = filter_for_context(rows, "dpo", CharTokenizer(), max_length=40, max_prompt_tokens=30, batch_size=1)
    assert filter_for_context(rows, "dpo", CharTokenizer(), max_length=40, max_prompt_tokens=30, batch_size=7) == expected


@pytest.mark.parametrize("kwargs", [{"max_length": True}, {"max_prompt_tokens": 0}, {"batch_size": -1},
                                     {"max_length": 30, "max_prompt_tokens": 30}, {"kind": "eval"}])
def test_invalid_token_budget_or_track_raises(kwargs):
    args = {"rows": [], "kind": "sft", "tokenizer": CharTokenizer(), **kwargs}
    with pytest.raises(ValueError):
        filter_for_context(**args)


def test_bad_row_fails_without_mutation():
    with pytest.raises(ValueError):
        filter_for_context([{"prompt": "Q", "response": None}], "sft", CharTokenizer())


def cli_module():
    path = Path(__file__).resolve().parents[1] / "scripts/prepare_data.py"
    spec = importlib.util.spec_from_file_location("review_token_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_uses_pinned_local_tokenizer_before_split_and_records_stage(tmp_path, monkeypatch):
    from transformers import AutoTokenizer
    calls = []
    def cached(model_id, **kwargs):
        calls.append((model_id, kwargs))
        return CharTokenizer()
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", cached)
    path = tmp_path / "rows.json"
    path.write_text(json.dumps([{"prompt": "Q", "response": "A"},
        {"prompt": "B", "response": "C"}, {"prompt": "Too long for this prompt budget", "response": "A"}]))
    dest = tmp_path / "out"
    cli_module().main(["--input", str(path), "--kind", "sft", "--source-id", "fixture", "--revision", "test",
        "--source-split", "train", "--train-size", "1", "--validation-size", "1", "--output-dir", str(dest),
        "--tokenizer-id", "fixture-tokenizer", "--tokenizer-revision", "a" * 40, "--tokenizer-cache", str(tmp_path),
        "--max-length", "40", "--max-prompt-tokens", "30"])
    assert calls == [("fixture-tokenizer", {"revision": "a" * 40, "cache_dir": str(tmp_path),
                                          "local_files_only": True, "token": False})]
    config = json.loads((dest / "manifest.json").read_text())["config"]
    assert config["filter_counts"]["kept"] == 3
    assert config["token_filter"]["counts"] == {"input": 3, "kept": 2, "prompt_too_long": 1,
        "empty_tokenized_response": 0, "identical_retained_pair": 0}


@pytest.mark.parametrize("flags", [["--max-length", "512"], ["--tokenizer-id", "x"],
    ["--tokenizer-id", "x", "--tokenizer-revision", "main"], ["--tokenizer-cache", "unused"]])
def test_cli_rejects_incomplete_tokenizer_configuration_before_input_read(flags):
    with pytest.raises(ValueError):
        cli_module().main(["--input", "does-not-exist.json", "--kind", "sft", "--source-id", "fixture",
            "--revision", "test", "--source-split", "train", "--output-dir", "unused", *flags])


@pytest.mark.parametrize("case", ["bad_sum", "bad_input", "bool_budget", "unreserved_response", "unpinned_revision"])
def test_writer_rejects_invalid_token_stage_before_mutation(tmp_path, case):
    from tests.test_review_datafiles import sample_data
    from src.datafiles import write_prepared
    splits, metadata = sample_data()
    stage = {"model_id": "fixture-tokenizer", "revision": "a" * 40, "max_length": 40, "max_prompt_tokens": 30,
             "policy": "preserve_full_prompt_filter", "counts": {"input": 4, "kept": 4,
              "prompt_too_long": 0, "empty_tokenized_response": 0, "identical_retained_pair": 0}}
    metadata["config"]["token_filter"] = stage
    if case == "bad_sum": stage["counts"]["prompt_too_long"] = 1
    elif case == "bad_input": stage["counts"].update(input=5, kept=5)
    elif case == "bool_budget": stage["max_length"] = True
    elif case == "unreserved_response": stage["max_prompt_tokens"] = 40
    elif case == "unpinned_revision": stage["revision"] = "main"
    with pytest.raises(ValueError):
        write_prepared(tmp_path / "out", splits, metadata)
    assert list(tmp_path.iterdir()) == []
