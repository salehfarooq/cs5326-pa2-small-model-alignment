"""Independent leakage exclusion and provenance checks at the real CLI boundary."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]


def load_cli():
    spec = importlib.util.spec_from_file_location("review_exclusions", REPO / "scripts/prepare_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inputs(tmp_path):
    path = tmp_path / "training.json"
    path.write_text(json.dumps([{"prompt": f"Q{i}", "response": "A"} for i in range(6)]))
    bank = tmp_path / "eval.json"
    payload = json.dumps([{"instruction": " Q0\r\n"}, {"instruction": "Q2"}, {"instruction": "Q2"}]).encode()
    bank.write_bytes(payload)
    args = ["--input", str(path), "--kind", "sft", "--source-id", "fixture", "--revision", "test",
            "--source-split", "train", "--exclude-eval", str(bank)]
    return args, bank, payload


def test_exclusion_happens_before_selection_and_records_exact_bank_bytes(tmp_path):
    args, bank, payload = inputs(tmp_path)
    dest = tmp_path / "out"
    load_cli().main(args + ["--output-dir", str(dest), "--train-size", "2", "--validation-size", "2"])
    manifest = json.loads((dest / "manifest.json").read_text())
    config = manifest["config"]
    assert config["filter_counts"]["kept"] == 6
    assert config["exclusions"] == {"removed": 2, "eligible": 4, "banks": [
        {"name": bank.name, "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload),
         "instructions": 3, "unique_instructions": 2}]}
    selected = []
    for name in ["train", "validation"]:
        selected += [json.loads(line) for line in (dest / (name + ".jsonl")).read_text().splitlines()]
    assert {row["prompt"] for row in selected} == {"Q1", "Q3", "Q4", "Q5"}
    assert {row["source_index"] for row in selected} == {1, 3, 4, 5}


def test_quota_error_reports_post_exclusion_available_without_output(tmp_path):
    args, _, _ = inputs(tmp_path)
    dest = tmp_path / "out"
    with pytest.raises(ValueError, match="have 4"):
        load_cli().main(args + ["--output-dir", str(dest), "--train-size", "3", "--validation-size", "2"])
    assert not dest.exists()


@pytest.mark.parametrize("bank_rows", [[], [None], [{"instruction": None}], [{"instruction": " "}], [{"prompt": "Q"}]])
def test_invalid_exclusion_bank_fails_without_output(tmp_path, bank_rows):
    args, bank, _ = inputs(tmp_path)
    bank.write_text(json.dumps(bank_rows))
    with pytest.raises(ValueError):
        load_cli().main(args + ["--output-dir", str(tmp_path / "out"), "--train-size", "1", "--validation-size", "1"])
    assert not (tmp_path / "out").exists()


def test_fixed_hh_exclusion_bank_retains_300_input_rows_and_185_unique_keys(tmp_path):
    bank = REPO / "eval_sets/hh_rlhf_eval_300_seed42.json"
    instruction = json.loads(bank.read_text())[0]["instruction"]
    path = tmp_path / "train.json"
    path.write_text(json.dumps([{"prompt": instruction, "response": "A"},
                                {"prompt": "Unrelated1", "response": "A"},
                                {"prompt": "Unrelated2", "response": "A"}]))
    dest = tmp_path / "out"
    load_cli().main(["--input", str(path), "--kind", "sft", "--source-id", "fixture", "--revision", "test",
        "--source-split", "train", "--exclude-eval", str(bank), "--output-dir", str(dest),
        "--train-size", "1", "--validation-size", "1"])
    excluded = json.loads((dest / "manifest.json").read_text())["config"]["exclusions"]
    assert excluded["removed"] == 1 and excluded["eligible"] == 2
    assert excluded["banks"][0]["instructions"] == 300
    assert excluded["banks"][0]["unique_instructions"] == 185


def test_repeatable_exclusions_union_banks_and_do_not_double_count_removed_rows(tmp_path):
    args, _, _ = inputs(tmp_path)
    second = tmp_path / "second.jsonl"
    second.write_text('{"instruction":"Q2"}\n{"instruction":"Q4"}\n')
    dest = tmp_path / "out"
    load_cli().main(args + ["--exclude-eval", str(second), "--output-dir", str(dest),
                           "--train-size", "2", "--validation-size", "1"])
    excluded = json.loads((dest / "manifest.json").read_text())["config"]["exclusions"]
    assert excluded["removed"] == 3 and excluded["eligible"] == 3 and len(excluded["banks"]) == 2


def test_generic_eval_selection_rejects_exclusion_flag(tmp_path):
    args, bank, _ = inputs(tmp_path)
    path = tmp_path / "eval-source.json"
    path.write_text('[{"instruction":"Q"}]')
    with pytest.raises(ValueError, match="eval"):
        load_cli().main(["--input", str(path), "--kind", "eval", "--source-id", "fixture", "--revision", "test",
            "--source-split", "eval", "--exclude-eval", str(bank), "--eval-size", "1",
            "--output-dir", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("case", ["bad_total", "bool_removed", "negative_eligible", "empty_banks", "bad_sha", "bool_instructions", "impossible_unique"])
def test_writer_validates_exclusion_provenance_and_accounting(tmp_path, case):
    from tests.test_review_datafiles import sample_data
    from src.datafiles import write_prepared
    splits, metadata = sample_data()
    metadata["config"]["filter_counts"].update(input=6, kept=6)
    excluded = {"removed": 2, "eligible": 4, "banks": [
        {"name": "bank.json", "sha256": hashlib.sha256(b"bank").hexdigest(), "bytes": 4,
         "instructions": 3, "unique_instructions": 2}]}
    metadata["config"]["exclusions"] = excluded
    if case == "bad_total": excluded["removed"] = 1
    elif case == "bool_removed": excluded["removed"] = True
    elif case == "negative_eligible": excluded["eligible"] = -1
    elif case == "empty_banks": excluded["banks"] = []
    elif case == "bad_sha": excluded["banks"][0]["sha256"] = "wrong"
    elif case == "bool_instructions": excluded["banks"][0]["instructions"] = True
    elif case == "impossible_unique": excluded["banks"][0]["unique_instructions"] = 4
    with pytest.raises(ValueError):
        write_prepared(tmp_path / "out", splits, metadata)
    assert list(tmp_path.iterdir()) == []
