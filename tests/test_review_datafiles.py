"""Independent artifact integrity, failure containment and real CLI checks."""

import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]


def datafiles():
    from src import datafiles as module
    return module


def sample_data():
    rows = [{"id": f"id-{i}", "source_index": i, "prompt": f"Question {i}",
             "response": "café\n    code"} for i in range(4)]
    splits = {"train": rows[:2], "validation": rows[2:]}
    metadata = {
        "source": {"id": "synthetic-fixture", "revision": "test-revision", "split": "train"},
        "input": {"name": "fixture.json", "sha256": hashlib.sha256(b"raw").hexdigest(), "bytes": 3},
        "config": {"kind": "sft", "schema": "normalized", "seed": 42,
                   "sizes": {"train": 2, "validation": 2},
                   "filter_counts": {"input": 4, "kept": 4, "invalid_record": 0,
                                     "empty_text": 0, "multi_turn": 0, "prompt_mismatch": 0,
                                     "identical_responses": 0, "duplicate_prompt": 0},
                   "prompt_template": "Instruction:\n{instruction}\n\nResponse:\n"}}
    return splits, metadata


def test_artifacts_hash_identity_and_deterministic_bytes(tmp_path):
    module = datafiles()
    splits, metadata = sample_data()
    before = copy.deepcopy((splits, metadata))
    first, second = tmp_path / "one", tmp_path / "two"
    module.write_prepared(first, splits, metadata)
    module.write_prepared(second, dict(reversed(list(splits.items()))), metadata)
    assert (splits, metadata) == before
    assert {p.name for p in first.iterdir()} == {"train.jsonl", "validation.jsonl", "manifest.json"}
    for path in first.iterdir():
        assert path.read_bytes() == (second / path.name).read_bytes()
    manifest = json.loads((first / "manifest.json").read_text())
    assert manifest["version"] == 1
    assert manifest["source"] == metadata["source"] and manifest["config"] == metadata["config"]
    assert manifest["input"] == metadata["input"]
    for name, expected_rows in splits.items():
        record = manifest["splits"][name]
        payload = (first / record["file"]).read_bytes()
        assert payload.endswith(b"\n") and b"caf\xc3\xa9" in payload
        assert record["sha256"] == hashlib.sha256(payload).hexdigest()
        assert record["rows"] == len(expected_rows)
        assert record["ids"] == [r["id"] for r in expected_rows]
        assert record["source_indices"] == [r["source_index"] for r in expected_rows]
        assert [json.loads(line) for line in payload.splitlines()] == expected_rows


def test_reads_json_and_jsonl_with_blank_lines(tmp_path):
    module = datafiles()
    rows = [{"prompt": "café", "response": "A"}, None]
    array = tmp_path / "input.JSON"
    array.write_text(json.dumps(rows, ensure_ascii=False))
    lines = tmp_path / "input.jsonl"
    lines.write_text('\n' + json.dumps(rows[0], ensure_ascii=False) + '\n \nnull\n')
    assert module.read_rows(array) == module.read_rows(lines) == rows


@pytest.mark.parametrize("filename,payload", [("a.json", "{}"), ("b.json", "[]"),
    ("c.json", ""), ("d.jsonl", "\n \n"), ("e.json", "[{]"),
    ("f.json", "[{\"value\": NaN}]"), ("g.jsonl", "{\"value\": Infinity}\n"),
    ("h.txt", "[]")])
def test_bad_export_fails_with_file_context(tmp_path, filename, payload):
    path = tmp_path / filename
    path.write_text(payload)
    with pytest.raises(ValueError) as error:
        datafiles().read_rows(path)
    assert filename in str(error.value)


def test_jsonl_parse_error_has_physical_line_number(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text('{"a": 1}\n\n{bad}\n')
    with pytest.raises(ValueError) as error:
        datafiles().read_rows(path)
    assert "broken.jsonl" in str(error.value) and "3" in str(error.value)


@pytest.mark.parametrize("existing", ["empty_dir", "file", "dangling_symlink"])
def test_existing_destination_never_overwritten(tmp_path, existing):
    splits, metadata = sample_data()
    dest = tmp_path / "out"
    if existing == "empty_dir":
        dest.mkdir()
    elif existing == "file":
        dest.write_text("preexisting")
    else:
        dest.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        datafiles().write_prepared(dest, splits, metadata)
    assert os.path.lexists(dest)
    if existing == "empty_dir":
        assert list(dest.iterdir()) == []
    elif existing == "file":
        assert dest.read_text() == "preexisting"
    else:
        assert dest.is_symlink()
    assert list(tmp_path.iterdir()) == [dest]


@pytest.mark.parametrize("defect", ["empty_splits", "unknown_split", "id_overlap", "prompt_overlap",
    "index_overlap", "blank_id", "bool_index", "missing_response", "nan_extra",
    "wrong_sizes", "wrong_counts", "blank_source", "bad_sha", "bad_template", "forged_top_level"])
def test_validation_is_complete_before_any_mutation(tmp_path, defect):
    splits, metadata = sample_data()
    if defect == "empty_splits":
        splits = {}
    elif defect == "unknown_split":
        splits["../outside"] = splits.pop("validation")
    elif defect == "id_overlap":
        splits["validation"][0]["id"] = splits["train"][0]["id"]
    elif defect == "prompt_overlap":
        splits["validation"][0]["prompt"] = " Question 0 "
    elif defect == "index_overlap":
        splits["validation"][0]["source_index"] = 0
    elif defect == "blank_id":
        splits["train"][0]["id"] = " "
    elif defect == "bool_index":
        splits["train"][0]["source_index"] = True
    elif defect == "missing_response":
        del splits["train"][0]["response"]
    elif defect == "nan_extra":
        splits["train"][0]["extra"] = float("nan")
    elif defect == "wrong_sizes":
        metadata["config"]["sizes"]["train"] = 3
    elif defect == "wrong_counts":
        metadata["config"]["filter_counts"]["input"] = 20
    elif defect == "blank_source":
        metadata["source"]["revision"] = " "
    elif defect == "bad_sha":
        metadata["input"]["sha256"] = "not-a-sha"
    elif defect == "bad_template":
        metadata["config"]["prompt_template"] = "something else"
    else:
        metadata["splits"] = {"train": {"sha256": "forged"}}
    with pytest.raises(ValueError):
        datafiles().write_prepared(tmp_path / "output", splits, metadata)
    assert list(tmp_path.iterdir()) == []


def test_manifest_is_published_last(tmp_path, monkeypatch):
    module = datafiles()
    splits, metadata = sample_data()
    original = module.os.link
    order = []
    def track(source, target, *args, **kwargs):
        order.append(Path(target).name)
        return original(source, target, *args, **kwargs)
    monkeypatch.setattr(module.os, "link", track)
    module.write_prepared(tmp_path / "out", splits, metadata)
    assert order[-1] == "manifest.json" and set(order[:-1]) == {"train.jsonl", "validation.jsonl"}


@pytest.mark.parametrize("intrusion", ["none", "sentinel", "replacement"])
def test_rollback_removes_only_owned_files(tmp_path, monkeypatch, intrusion):
    module = datafiles()
    splits, metadata = sample_data()
    dest = tmp_path / "out"
    original = module.os.link
    linked = []
    def fail(source, target, *args, **kwargs):
        if linked:
            if intrusion == "sentinel":
                (dest / "unrelated.txt").write_text("keep me")
            elif intrusion == "replacement":
                linked[0].unlink()
                linked[0].write_text("replacement, not ours")
            raise OSError("injected publication failure")
        original(source, target, *args, **kwargs)
        linked.append(Path(target))
    monkeypatch.setattr(module.os, "link", fail)
    with pytest.raises(OSError, match="injected"):
        module.write_prepared(dest, splits, metadata)
    if intrusion == "none":
        assert list(tmp_path.iterdir()) == []
    elif intrusion == "sentinel":
        assert (dest / "unrelated.txt").read_text() == "keep me"
        assert {p.name for p in dest.iterdir()} == {"unrelated.txt"}
    else:
        assert linked[0].read_text() == "replacement, not ours"
        assert list(dest.iterdir()) == [linked[0]]
    assert not (dest / "manifest.json").exists()
    assert all(p == dest for p in tmp_path.iterdir())


def cli(input_path, output_path, *extra):
    return subprocess.run([sys.executable, str(REPO / "scripts/prepare_data.py"),
        "--input", str(input_path), "--output-dir", str(output_path),
        "--source-id", "synthetic-fixture", "--revision", "test-revision", "--source-split", "train",
        *extra], cwd=REPO, capture_output=True, text=True)


def test_cli_real_process_with_spaces_provenance_and_disjoint_splits(tmp_path):
    path = tmp_path / "input bank.jsonl"
    rows = [{"prompt": f"Question {i}", "response": "Answer"} for i in range(8)]
    rows += [{"prompt": "Question 0", "response": "other"}, {"prompt": "", "response": "empty"}]
    raw = ''.join(json.dumps(row) + '\n' for row in rows).encode()
    path.write_bytes(raw)
    dest = tmp_path / "prepared data"
    result = cli(path, dest, "--kind", "sft", "--train-size", "4", "--validation-size", "2")
    assert result.returncode == 0, result.stderr + result.stdout
    manifest = json.loads((dest / "manifest.json").read_text())
    assert manifest["input"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert manifest["input"]["bytes"] == len(raw) and manifest["input"]["name"] == path.name
    assert manifest["config"]["filter_counts"]["input"] == 10
    assert manifest["config"]["filter_counts"]["kept"] == 8
    train = [json.loads(line) for line in (dest / "train.jsonl").read_text().splitlines()]
    val = [json.loads(line) for line in (dest / "validation.jsonl").read_text().splitlines()]
    assert len(train) == 4 and len(val) == 2
    assert not ({r["prompt"] for r in train} & {r["prompt"] for r in val})
    assert all(not r["prompt"].startswith("Instruction:") for r in train + val)
    before = {p.name: p.read_bytes() for p in dest.iterdir()}
    again = cli(path, dest, "--kind", "sft", "--train-size", "4", "--validation-size", "2")
    assert again.returncode != 0
    assert {p.name: p.read_bytes() for p in dest.iterdir()} == before


@pytest.mark.parametrize("args", [("--kind", "sft"),
    ("--kind", "sft", "--eval-size", "2"),
    ("--kind", "eval", "--train-size", "2"),
    ("--kind", "sft", "--schema", "hh"),
    ("--kind", "sft", "--train-size", "0", "--validation-size", "1")])
def test_cli_bad_quota_or_schema_creates_no_output(tmp_path, args):
    path = tmp_path / "tiny.json"
    path.write_text(json.dumps([{"prompt": "Q", "response": "A", "instruction": "Q"}]))
    dest = tmp_path / "out"
    result = cli(path, dest, *args)
    assert result.returncode != 0 and not dest.exists()
    assert set(tmp_path.iterdir()) == {path}


def test_cli_generic_eval_keeps_metadata(tmp_path):
    path = tmp_path / "eval.json"
    path.write_text(json.dumps([{"instruction": f"Q{i}", "category": "benign"} for i in range(5)]))
    dest = tmp_path / "out"
    result = cli(path, dest, "--kind", "eval", "--eval-size", "3")
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in (dest / "eval.jsonl").read_text().splitlines()]
    assert len(rows) == 3 and all(row["category"] == "benign" for row in rows)
    assert set(json.loads((dest / "manifest.json").read_text())["splits"]) == {"eval"}


def test_cli_rejects_copy_of_authoritative_hh_bank(tmp_path):
    path = tmp_path / "renamed.json"
    path.write_bytes((REPO / "eval_sets/hh_rlhf_eval_300_seed42.json").read_bytes())
    result = cli(path, tmp_path / "out", "--kind", "eval", "--eval-size", "100")
    assert result.returncode != 0 and not (tmp_path / "out").exists()
    assert "fixed" in result.stderr.lower()


def test_cli_provenance_hashes_the_same_single_read(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("review_prepare", REPO / "scripts/prepare_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "input.json"
    payload = json.dumps([{"prompt": f"Q{i}", "response": "A"} for i in range(3)]).encode()
    path.write_bytes(payload)
    original = Path.read_bytes
    calls = []
    def read_once(self):
        if self == path:
            calls.append(self)
            return payload if len(calls) == 1 else b"changed after read"
        return original(self)
    monkeypatch.setattr(Path, "read_bytes", read_once)
    dest = tmp_path / "out"
    module.main(["--input", str(path), "--output-dir", str(dest), "--kind", "sft",
                 "--train-size", "1", "--validation-size", "1", "--source-id", "fixture",
                 "--revision", "rev", "--source-split", "train"])
    manifest = json.loads((dest / "manifest.json").read_text())
    assert len(calls) == 1 and manifest["input"]["sha256"] == hashlib.sha256(payload).hexdigest()


def test_cli_hh_preference_fields_survive_end_to_end(tmp_path):
    path = tmp_path / "hh.jsonl"
    rows = [{"chosen": f"\n\nHuman: Q{i}\n\nAssistant: yes {i}",
             "rejected": f"\n\nHuman: Q{i}\n\nAssistant: no {i}"} for i in range(5)]
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    dest = tmp_path / "out"
    result = cli(path, dest, "--kind", "dpo", "--schema", "hh", "--train-size", "3", "--validation-size", "1")
    assert result.returncode == 0, result.stderr
    for split in ["train", "validation"]:
        for line in (dest / (split + ".jsonl")).read_text().splitlines():
            row = json.loads(line)
            idx = row["source_index"]
            assert row["prompt"] == f"Q{idx}"
            assert row["chosen"] == f"yes {idx}" and row["rejected"] == f"no {idx}"


def test_output_parent_requirement_never_creates_unrequested_ancestors(tmp_path):
    splits, metadata = sample_data()
    with pytest.raises((ValueError, FileNotFoundError)):
        datafiles().write_prepared(tmp_path / "missing" / "out", splits, metadata)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("filename,payload", [("empty.json", b"[]"), ("empty.jsonl", b"\n \n")])
def test_parser_itself_rejects_empty_bank(filename, payload):
    with pytest.raises(ValueError, match="empty"):
        datafiles().parse_rows(payload, filename)


def test_manifest_serialized_before_any_filesystem_mutation(tmp_path, monkeypatch):
    module = datafiles()
    splits, metadata = sample_data()
    metadata["source"]["extra"] = float("nan")
    calls = []
    original = module.tempfile.mkdtemp
    def track(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(module.tempfile, "mkdtemp", track)
    with pytest.raises(ValueError):
        module.write_prepared(tmp_path / "out", splits, metadata)
    assert not calls and list(tmp_path.iterdir()) == []


def test_rollback_never_removes_destination_created_by_competing_writer(tmp_path, monkeypatch):
    module = datafiles()
    splits, metadata = sample_data()
    dest = tmp_path / "out"
    original = module.os.mkdir
    competing_inode = []
    def create_competing(path, *args, **kwargs):
        if Path(path) == dest:
            original(path, *args, **kwargs)
            competing_inode.append(dest.stat().st_ino)
            raise FileExistsError("another writer created destination")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(module.os, "mkdir", create_competing)
    with pytest.raises(FileExistsError):
        module.write_prepared(dest, splits, metadata)
    assert dest.is_dir() and dest.stat().st_ino == competing_inode[0]
    assert set(tmp_path.iterdir()) == {dest}


def test_rollback_does_not_remove_replaced_destination_directory(tmp_path, monkeypatch):
    module = datafiles()
    splits, metadata = sample_data()
    dest, moved = tmp_path / "out", tmp_path / "moved"
    original = module.os.link
    calls = []
    replacement_inode = []
    def replace_dir(src, dst, *args, **kwargs):
        calls.append(dst)
        if len(calls) == 2:
            dest.rename(moved)
            dest.mkdir()
            replacement_inode.append(dest.stat().st_ino)
            raise OSError("injected second-link failure")
        return original(src, dst, *args, **kwargs)
    monkeypatch.setattr(module.os, "link", replace_dir)
    with pytest.raises(OSError):
        module.write_prepared(dest, splits, metadata)
    assert dest.is_dir() and dest.stat().st_ino == replacement_inode[0]
    assert list(dest.iterdir()) == []
    assert set(tmp_path.iterdir()) == {dest, moved}


def test_rollback_preserves_replacement_symlink_even_to_owned_inode(tmp_path, monkeypatch):
    module = datafiles()
    splits, metadata = sample_data()
    dest = tmp_path / "out"
    original = module.os.link
    calls = []
    replacement = []
    def replace_file(src, dst, *args, **kwargs):
        calls.append((src, dst))
        if len(calls) == 2:
            first_src, first_dst = calls[0]
            Path(first_dst).unlink()
            Path(first_dst).symlink_to(first_src)
            replacement.append(Path(first_dst))
            raise OSError("injected failure after symlink replacement")
        return original(src, dst, *args, **kwargs)
    monkeypatch.setattr(module.os, "link", replace_file)
    with pytest.raises(OSError):
        module.write_prepared(dest, splits, metadata)
    assert replacement[0].is_symlink()
    assert set(dest.iterdir()) == {replacement[0]}
    assert set(tmp_path.iterdir()) == {dest}


@pytest.mark.parametrize("field", ["dataset", "category", "expected_behavior"])
def test_writer_validates_optional_eval_metadata_before_mutation(tmp_path, field):
    _, metadata = sample_data()
    metadata["config"].update(kind="eval", sizes={"eval": 1})
    row = {"id": "one", "source_index": 0, "instruction": "Q", field: ["wrong type"]}
    with pytest.raises(ValueError):
        datafiles().write_prepared(tmp_path / "out", {"eval": [row]}, metadata)
    assert list(tmp_path.iterdir()) == []


def test_generic_eval_preserves_dataset_and_drops_baseline_output():
    from src.data import normalize_rows
    rows, counts = normalize_rows([{"instruction": "Q", "dataset": "vicuna",
                                    "output": "baseline response", "generator": "baseline"}], "eval")
    assert counts["kept"] == 1 and rows[0]["dataset"] == "vicuna"
    assert "output" not in rows[0] and "generator" not in rows[0]
    bad, counts = normalize_rows([{"instruction": "Q", "dataset": None}], "eval")
    assert bad == [] and counts["invalid_record"] == 1


@pytest.mark.parametrize("separator", ["\u2028", "\u2029", "\u0085"])
def test_jsonl_unicode_separators_inside_strings_are_not_record_boundaries(separator):
    row = {"prompt": "Q", "response": "before" + separator + "after"}
    payload = (json.dumps(row, ensure_ascii=False) + "\r\n").encode()
    assert datafiles().parse_rows(payload, "unicode.jsonl") == [row]


def test_rollback_skips_links_in_replaced_directory_even_when_inode_matches(tmp_path, monkeypatch):
    module = datafiles()
    splits, metadata = sample_data()
    dest, moved = tmp_path / "out", tmp_path / "moved"
    original = module.os.link
    calls = []
    replacement = []
    def replace_with_shared_file(src, dst, *args, **kwargs):
        calls.append(dst)
        if len(calls) == 2:
            first_name = Path(calls[0]).name
            dest.rename(moved)
            dest.mkdir()
            introduced = dest / first_name
            original(moved / first_name, introduced)
            replacement.append(introduced)
            raise OSError("directory replaced with independently introduced hard link")
        return original(src, dst, *args, **kwargs)
    monkeypatch.setattr(module.os, "link", replace_with_shared_file)
    with pytest.raises(OSError):
        module.write_prepared(dest, splits, metadata)
    assert replacement[0].exists() and set(dest.iterdir()) == {replacement[0]}
    assert set(tmp_path.iterdir()) == {dest, moved}


def test_known_fixed_bank_path_is_independent_of_current_directory(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("review_fixed_path", REPO / "scripts/prepare_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.chdir(tmp_path)
    assert module._is_fixed_eval_bank(str(REPO / "eval_sets/hh_rlhf_eval_300_seed42.json"), "not-original-hash")
    assert not module._is_fixed_eval_bank(str(tmp_path / "hh_rlhf_eval_300_seed42.json"), "not-original-hash")
