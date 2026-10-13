"""Independent checks for single-pass, bounded-input data preparation."""

import copy
import hashlib
import io
import json
import weakref
from pathlib import Path

import pytest


def test_streaming_matches_batch_schema_filtering_ids_and_indices():
    from src.data import normalize_rows, normalize_records
    examples = [
        ("sft", "normalized", [{"prompt": " Q\r\n", "response": "A"}, None,
                                {"prompt": "Q\n", "response": "B"},
                                {"prompt": "Second", "response": " "}]),
        ("sft", "ultrachat", [{"prompt_id": "one", "messages": [
            {"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]},
            {"messages": [{}, {}, {}]}, {"messages": []}]),
        ("dpo", "hh", [{"chosen": "\n\nHuman: Q\n\nAssistant: A",
                          "rejected": "\n\nHuman: Q\n\nAssistant: B"},
                         {"chosen": "bad", "rejected": "bad"}]),
        ("dpo", "ultrafeedback", [{"chosen": [{"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"}], "rejected": [
            {"role": "user", "content": "Q"}, {"role": "assistant", "content": "B"}]}]),
        ("eval", "normalized", [{"instruction": "Q", "dataset": "vicuna"},
                                 {"instruction": "Q", "dataset": "other"},
                                 {"instruction": "New", "dataset": None}]),
    ]
    before = copy.deepcopy(examples)
    for kind, schema, rows in examples:
        assert normalize_records((row for row in rows), kind, schema) == normalize_rows(rows, kind, schema)
    assert examples == before


def test_streaming_consumes_each_record_once_and_retains_global_source_index():
    from src.data import normalize_records
    class Once:
        def __init__(self):
            self.calls = 0
        def __iter__(self):
            self.calls += 1
            assert self.calls == 1
            for index in range(1000):
                yield {"prompt": f"Q{index % 5}", "response": f"A{index}"}
    records = Once()
    rows, counts = normalize_records(records, "sft")
    assert [row["source_index"] for row in rows] == list(range(5))
    assert counts["input"] == 1000 and counts["kept"] == 5 and counts["duplicate_prompt"] == 995
    assert records.calls == 1


def test_streaming_releases_raw_records_as_source_advances():
    from src.data import normalize_records
    class RawRow(dict):
        pass
    references = []
    def records():
        for index in range(100):
            row = RawRow(prompt=f"Q{index % 2}", response="A", unused_large_metadata="x" * 100000)
            references.append(weakref.ref(row))
            yield row
            assert sum(reference() is not None for reference in references) <= 2
    rows, counts = normalize_records(records(), "sft")
    assert len(rows) == 2 and counts["duplicate_prompt"] == 98
    assert all(reference() is None for reference in references)


def test_streaming_does_not_accumulate_normalized_fields_of_discarded_duplicates():
    from src.data import normalize_records
    class Response(str):
        pass
    references = []
    def records():
        for index in range(100):
            response = Response(f"A{index}")
            references.append(weakref.ref(response))
            yield {"id": f"id-{index}", "prompt": f"Q{index % 2}", "response": response}
            assert sum(reference() is not None for reference in references) <= 3
    rows, counts = normalize_records(records(), "sft")
    assert len(rows) == 2 and counts["duplicate_prompt"] == 98
    assert sum(reference() is not None for reference in references) == 2


def test_streaming_checks_id_collisions_after_prompt_dedup_and_at_eof():
    from src.data import normalize_records
    rows = [{"id": "a", "prompt": "Q", "response": "A"},
            {"id": "b", "prompt": "Q", "response": "B"},
            {"id": "b", "prompt": "different", "response": "B"}]
    with pytest.raises(ValueError, match="duplicate id"):
        normalize_records(iter(rows), "sft")


@pytest.mark.parametrize("records", [None, 1, "abc", b"abc", {"prompt": "Q"}])
def test_streaming_rejects_ambiguous_outer_containers(records):
    from src.data import normalize_records
    with pytest.raises(ValueError):
        normalize_records(records, "sft")


def test_empty_iterator_matches_empty_batch_and_preserves_error_propagation():
    from src.data import normalize_rows, normalize_records
    assert normalize_records(iter(()), "sft") == normalize_rows([], "sft")
    def interrupted():
        yield {"prompt": "Q", "response": "A"}
        raise OSError("source interrupted")
    with pytest.raises(OSError, match="source interrupted"):
        normalize_records(interrupted(), "sft")


def test_invalid_stream_configuration_does_not_consume_generator():
    from src.data import normalize_records
    def rows():
        raise AssertionError("must validate configuration before consuming")
        yield None
    with pytest.raises(ValueError):
        normalize_records(rows(), "dpo", "ultrachat")


def test_jsonl_reader_is_single_open_iterative_and_hashes_all_original_bytes(tmp_path, monkeypatch):
    from src import datafiles
    row = {"prompt": "café\u2028code", "response": "A\r\nB"}
    payload = b"\r\n" + json.dumps(row, ensure_ascii=False).encode() + b"\r\n \r\nnull\n"
    path = tmp_path / "in.JSONL"
    path.write_bytes(payload)
    calls, handles = [], []
    class IterOnly(io.BytesIO):
        def read(self, *args, **kwargs):
            raise AssertionError("JSONL must not preload the entire file")
        def readlines(self, *args, **kwargs):
            raise AssertionError("JSONL must not preload raw lines")
    def single_open(self, *args, **kwargs):
        assert self == path and args[0] == "rb"
        calls.append(self)
        handle = IterOnly(payload if len(calls) == 1 else b"changed")
        handles.append(handle)
        return handle
    monkeypatch.setattr(Path, "open", single_open)
    monkeypatch.setattr("builtins.open", single_open)
    rows, counts, metadata = datafiles.read_normalized(path, "sft")
    assert len(calls) == 1 and handles[0].closed
    assert counts["input"] == 2 and counts["invalid_record"] == 1 and counts["kept"] == 1
    assert rows[0]["prompt"] == row["prompt"] and rows[0]["response"] == row["response"]
    assert rows[0]["source_index"] == 0
    assert metadata == {"name": path.name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


@pytest.mark.parametrize("payload,physical_line", [(b"\n{bad}\n", 2),
    (b'{"prompt":"Q", "response":NaN}\n', 1), (b"\n\xff\n", 2)])
def test_stream_reader_errors_have_path_and_physical_line_and_close_handle(tmp_path, monkeypatch, payload, physical_line):
    from src import datafiles
    path = tmp_path / "broken.jsonl"
    handle = io.BytesIO(payload)
    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: handle)
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: handle)
    with pytest.raises(ValueError) as error:
        datafiles.read_normalized(path, "sft")
    assert "broken.jsonl" in str(error.value) and str(physical_line) in str(error.value)
    assert handle.closed


@pytest.mark.parametrize("suffix,payload", [(".json", b"[]"), (".jsonl", b"\n \n"), (".txt", b"[]")])
def test_read_normalized_rejects_empty_or_unsupported_bank(tmp_path, suffix, payload):
    from src import datafiles
    path = tmp_path / ("empty" + suffix)
    path.write_bytes(payload)
    with pytest.raises(ValueError):
        datafiles.read_normalized(path, "sft")


def test_jsonl_handle_closes_when_normalization_raises_before_exhaustion(tmp_path, monkeypatch):
    from src import datafiles
    path = tmp_path / "conflict.jsonl"
    payload = b'{"id":"one","prompt":"Q","response":"A"}\n' + b'{"id":"one","prompt":"Q","response":"B"}\n' + b'null\n'
    handle = io.BytesIO(payload)
    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: handle)
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: handle)
    with pytest.raises(ValueError, match="duplicate id") as error:
        datafiles.read_normalized(path, "sft")
    assert handle.closed, str(error.value)
