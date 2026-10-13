"""Independent failure-report checks; no real model or CUDA download locally."""

import importlib.util
import json
from pathlib import Path

import pytest


def load_smoke_module():
    path = Path(__file__).resolve().parents[1] / "scripts/kaggle_smoke.py"
    spec = importlib.util.spec_from_file_location("review_smoke", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_failed_cuda_gate_replaces_previous_success_report(tmp_path, monkeypatch):
    module = load_smoke_module()
    output = tmp_path / "status.json"
    output.write_text('{"status": "success", "old_result": true}')
    monkeypatch.setattr(module, "_torch_versions", lambda: {"torch": "local-test"})
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        module.main(["--revision", "a" * 40, "--output", str(output)])
    report = json.loads(output.read_text())
    assert report["status"] == "failed" and "old_result" not in report
    assert report["optimizer_update_performed"] is False
    assert report["full_training"] is False


@pytest.mark.parametrize("revision", ["main", "", "a" * 39, "g" * 40, "a" * 41])
def test_nonimmutable_revision_rejected_before_cuda_or_model(revision, tmp_path, monkeypatch):
    module = load_smoke_module()
    def forbidden_cuda_check():
        pytest.fail("invalid revision must fail before CUDA or model loading")
    monkeypatch.setattr(module.torch.cuda, "is_available", forbidden_cuda_check)
    with pytest.raises(ValueError, match="revision"):
        module.main(["--revision", revision, "--output", str(tmp_path / "report.json")])


def test_atomic_report_contains_valid_json_and_no_leftover_temp(tmp_path):
    module = load_smoke_module()
    output = tmp_path / "report.json"
    module._atomic_write_json(str(output), {"status": "success", "loss": 0.7})
    assert json.loads(output.read_text()) == {"status": "success", "loss": 0.7}
    assert list(tmp_path.iterdir()) == [output]
