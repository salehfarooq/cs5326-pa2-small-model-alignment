"""Blinded pairwise comparison and uncertainty for SFT vs DPO evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
from pathlib import Path

from src.data import _canonical_prompt

_FIXED_BANK_COUNT = 300
_VALID_DECISIONS = frozenset({"A", "B", "tie", "unknown"})
_KNOWN_BANK_KINDS = frozenset({"alpaca_eval", "hh_rlhf_eval"})
_REQUIRED_RUN_FIELDS = frozenset({
    "status", "completed", "total", "bank_sha256", "decoding",
    "parameter_dtype", "tokenizer_fingerprint", "prompt_policy", "outputs_sha256",
    "bank_kind", "checkpoint",
})
_BANKS = {
    'alpaca_eval': ('alpaca_eval', '3989bb58dae97eb5fded3b21de17f8d76605488d0ba3ab7d17fc77e38fdfd6c1'),
    'hh': ('hh_rlhf_eval', '0731105407d06152e584c722717712be4f418f221a8d5c487bb924c4d91c38c6'),
}


def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _require_nonblank_string(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonblank string")
    return value


def _require_sha256(value, label):
    _require_nonblank_string(value, label)
    if len(value) != 64 or not all(c in "0123456789abcdef" for c in value):
        raise ValueError(f"{label} must be 64 lowercase hex characters")
    return value


def _read_json_object(path):
    with open(path, "rb") as handle:
        payload = handle.read()
    try:
        data = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}")
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level JSON must be an object")
    return data


def _read_json_array(path):
    with open(path, "rb") as handle:
        payload = handle.read()
    try:
        data = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}")
    if not isinstance(data, list):
        raise ValueError(f"{path}: top-level JSON must be an array")
    return data


def _verify_run(run, label):
    missing = _REQUIRED_RUN_FIELDS - set(run)
    if missing:
        raise ValueError(f"{label}: run.json missing fields: {sorted(missing)}")
    if run["status"] != "completed":
        raise ValueError(f"{label}: status is {run['status']!r}, expected 'completed'")
    if isinstance(run["completed"], bool) or not isinstance(run["completed"], int):
        raise ValueError(f"{label}: completed must be an integer")
    if isinstance(run["total"], bool) or not isinstance(run["total"], int):
        raise ValueError(f"{label}: total must be an integer")
    if run["completed"] != _FIXED_BANK_COUNT or run["total"] != _FIXED_BANK_COUNT:
        raise ValueError(
            f"{label}: expected {_FIXED_BANK_COUNT} rows, got {run['completed']}/{run['total']}"
        )
    _require_sha256(run["bank_sha256"], f"{label}: bank_sha256")
    _require_nonblank_string(run["parameter_dtype"], f"{label}: parameter_dtype")
    _require_nonblank_string(run["tokenizer_fingerprint"], f"{label}: tokenizer_fingerprint")
    _require_nonblank_string(run["prompt_policy"], f"{label}: prompt_policy")
    _require_sha256(run["outputs_sha256"], f"{label}: outputs_sha256")
    if not isinstance(run["decoding"], dict):
        raise ValueError(f"{label}: decoding must be a dict")
    decoder = run['decoding']
    if set(decoder) != {'do_sample', 'max_new_tokens', 'repetition_penalty', 'no_repeat_ngram_size'}:
        raise ValueError('incomplete decoding settings')
    if decoder['do_sample'] is not False:
        raise ValueError('comparison requires greedy decoding')
    for key, minimum in [('max_new_tokens', 1), ('no_repeat_ngram_size', 0)]:
        if isinstance(decoder[key], bool) or not isinstance(decoder[key], int) or decoder[key] < minimum:
            raise ValueError('invalid decoding setting: ' + key)
    penalty = decoder['repetition_penalty']
    if isinstance(penalty, bool) or not isinstance(penalty, (int, float)) or not math.isfinite(penalty) or penalty < 1:
        raise ValueError('invalid repetition penalty')
    if run['bank_kind'] not in _BANKS or run['bank_sha256'] != _BANKS[run['bank_kind']][1]:
        raise ValueError('fixed bank identity mismatch')


def _checkpoint_record(run, expected_kind):
    checkpoint = run['checkpoint']
    if not isinstance(checkpoint, dict):
        raise ValueError('missing checkpoint identity')
    state = checkpoint.get('state', {})
    parent = checkpoint.get('parent_run', {})
    if not isinstance(state, dict) or not isinstance(parent, dict):
        raise ValueError('invalid checkpoint identity')
    state_bytes = (json.dumps(state, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    if checkpoint.get('state_sha256') != _sha256_bytes(state_bytes):
        raise ValueError('checkpoint state identity mismatch')
    config, metadata = state.get('config', {}), state.get('metadata', {})
    if not isinstance(config, dict) or not isinstance(metadata, dict):
        raise ValueError('invalid checkpoint metadata')
    track = metadata.get('track')
    if config.get('kind') != expected_kind or track not in (('sft',) if expected_kind == 'sft' else ('hh', 'uf')):
        raise ValueError('wrong model track in comparison')
    if (metadata.get('run_scope') != 'full' or parent.get('status') != 'completed' or
            metadata != parent.get('metadata') or config != parent.get('config') or
            state.get('checkpoint') not in ('best', 'final')):
        raise ValueError('incomplete checkpoint provenance')
    updates, terminal = state.get('updates'), parent.get('updates')
    if (isinstance(updates, bool) or not isinstance(updates, int) or updates <= 0 or
            isinstance(terminal, bool) or not isinstance(terminal, int) or terminal < updates or
            (state['checkpoint'] == 'final' and terminal != updates)):
        raise ValueError('invalid checkpoint update provenance')
    pin = ('Qwen/Qwen2.5-0.5B', '060db6499f32faf8b98477b0a26969ef7d8b9987')
    if (metadata.get('model_id'), metadata.get('model_revision')) != pin:
        raise ValueError('checkpoint model pin mismatch')
    files = state.get('files')
    if not isinstance(files, dict) or not isinstance(files.get('model.safetensors'), dict):
        raise ValueError('invalid checkpoint weight manifest')
    weight = files['model.safetensors'].get('sha256')
    _require_sha256(weight, 'checkpoint weight hash')
    if checkpoint.get('model_sha256') != weight:
        raise ValueError('checkpoint weight identity mismatch')
    return state, weight


def _row_bank(rows):
    values = {row.get('dataset') for row in rows}
    if len(values) != 1:
        raise ValueError('mixed dataset rows')
    return next(iter(values))


def _verify_rows(rows, label):
    if len(rows) != _FIXED_BANK_COUNT:
        raise ValueError(f"{label}: expected {_FIXED_BANK_COUNT} rows, got {len(rows)}")
    seen_ids = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{label}: each row must be a dict")
        for field in ("id", "instruction", "output"):
            if field not in row:
                raise ValueError(f"{label}: row missing field {field!r}")
        _require_nonblank_string(row["id"], f"{label}: row id")
        if row["id"] in seen_ids:
            raise ValueError(f"{label}: duplicate id {row['id']!r}")
        seen_ids.add(row["id"])
        _require_nonblank_string(row["instruction"], f"{label}: row {row['id']!r} instruction")
        if not isinstance(row["output"], str):
            raise ValueError(f"{label}: row {row['id']!r} output must be a string")


def _verify_export(export_dir, label):
    run_path = export_dir / "run.json"
    outputs_path = export_dir / "outputs.json"
    if not run_path.is_file():
        raise ValueError(f"{label}: run.json not found")
    if not outputs_path.is_file():
        raise ValueError(f"{label}: outputs.json not found")
    run = _read_json_object(run_path)
    _verify_run(run, label)
    outputs_bytes = outputs_path.read_bytes()
    actual_sha = _sha256_bytes(outputs_bytes)
    if actual_sha != run["outputs_sha256"]:
        raise ValueError(
            f"{label}: outputs.json SHA256 mismatch: expected {run['outputs_sha256']}, got {actual_sha}"
        )
    rows = _read_json_array(outputs_path)
    _verify_rows(rows, label)
    return run, rows, outputs_bytes, run_path.read_bytes()


def _verify_pairing(sft_run, dpo_run, sft_rows, dpo_rows):
    for field in ("bank_sha256", "decoding", "parameter_dtype", "tokenizer_fingerprint", "prompt_policy"):
        if sft_run[field] != dpo_run[field]:
            raise ValueError(f"metadata mismatch on {field!r}")
    sft_origin = sft_run.get("origin_sha256")
    dpo_origin = dpo_run.get("frozen_sft_origin_sha256")
    _require_sha256(sft_origin, "SFT run.json origin_sha256")
    _require_sha256(dpo_origin, "DPO run.json frozen_sft_origin_sha256")
    if sft_origin != dpo_origin:
        raise ValueError("SFT origin checkpoint does not match DPO frozen SFT origin")
    sft_state, sft_weight = _checkpoint_record(sft_run, 'sft')
    dpo_state, _ = _checkpoint_record(dpo_run, 'dpo')
    if sft_state['checkpoint'] != 'best' or sft_weight != sft_origin:
        raise ValueError('comparison SFT must be the frozen best checkpoint')
    origin = dpo_state['metadata'].get('sft_origin', {})
    if not isinstance(origin, dict) or origin.get('state') != sft_state:
        raise ValueError('DPO was initialized from a different SFT checkpoint')
    if origin.get('parent_run') != sft_run['checkpoint']['parent_run']:
        raise ValueError('DPO SFT completion provenance mismatch')
    if sft_run['bank_kind'] != dpo_run['bank_kind']:
        raise ValueError('bank kinds differ')
    expected_dataset = _BANKS[sft_run['bank_kind']][0]
    if _row_bank(sft_rows) != expected_dataset or _row_bank(dpo_rows) != expected_dataset:
        raise ValueError('output row dataset disagrees with fixed bank')
    for sft_row, dpo_row in zip(sft_rows, dpo_rows):
        if sft_row["id"] != dpo_row["id"]:
            raise ValueError(f"id order mismatch: {sft_row['id']!r} vs {dpo_row['id']!r}")
        if sft_row["instruction"] != dpo_row["instruction"]:
            raise ValueError(f"instruction mismatch for id {sft_row['id']!r}")


def _blind(sft_rows, dpo_rows, seed):
    rng = random.Random(seed)
    pairs = []
    mapping_pairs = []
    for sft_row, dpo_row in zip(sft_rows, dpo_rows):
        if rng.randrange(2) == 0:
            a_resp, b_resp = sft_row["output"], dpo_row["output"]
            a_track, b_track = "sft", "dpo"
        else:
            a_resp, b_resp = dpo_row["output"], sft_row["output"]
            a_track, b_track = "dpo", "sft"
        pairs.append({
            "id": sft_row["id"],
            "instruction": sft_row["instruction"],
            "response_a": a_resp,
            "response_b": b_resp,
        })
        mapping_pairs.append({
            "id": sft_row["id"],
            "a": a_track,
            "b": b_track,
        })
    return pairs, mapping_pairs


def _serialize_pairs(pairs):
    lines = []
    for pair in pairs:
        lines.append(json.dumps(pair, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _serialize_json(obj):
    return (json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _atomic_write(path, data):
    temp = path.parent / f".{path.name}.tmp"
    with open(temp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _percentile(sorted_values, percentile):
    if not sorted_values:
        return None
    n = len(sorted_values)
    index = (percentile / 100) * (n - 1)
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return sorted_values[lower]
    frac = index - lower
    return sorted_values[lower] * (1 - frac) + sorted_values[upper] * frac


def _cluster_by_instruction(pairs):
    clusters = {}
    for pair in pairs:
        canonical = _canonical_prompt(pair["instruction"])
        if canonical not in clusters:
            clusters[canonical] = []
        clusters[canonical].append(pair["id"])
    return clusters


def build_pairs(sft_dir, dpo_dir, output_dir, seed=42):
    """Build blinded pairs from completed SFT and DPO export directories."""
    sft_path = Path(sft_dir)
    dpo_path = Path(dpo_dir)
    out_path = Path(output_dir)
    if not sft_path.is_dir():
        raise ValueError(f"SFT directory not found: {sft_path}")
    if not dpo_path.is_dir():
        raise ValueError(f"DPO directory not found: {dpo_path}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    sft_run, sft_rows, sft_outputs_bytes, sft_run_bytes = _verify_export(sft_path, "SFT")
    dpo_run, dpo_rows, dpo_outputs_bytes, dpo_run_bytes = _verify_export(dpo_path, "DPO")
    _verify_pairing(sft_run, dpo_run, sft_rows, dpo_rows)

    bank_kind = sft_rows[0].get("dataset", "")
    if bank_kind not in _KNOWN_BANK_KINDS:
        raise ValueError(f"unknown dataset: {bank_kind!r}")

    pairs, mapping_pairs = _blind(sft_rows, dpo_rows, seed)

    mapping = {
        "inputhashes": {
            "sft": {"run_sha256": _sha256_bytes(sft_run_bytes), "outputs_sha256": _sha256_bytes(sft_outputs_bytes)},
            "dpo": {"run_sha256": _sha256_bytes(dpo_run_bytes), "outputs_sha256": _sha256_bytes(dpo_outputs_bytes)},
        },
        "bankkind": bank_kind,
        "banksha": sft_run["bank_sha256"],
        "decoding": sft_run["decoding"],
        "seed": seed,
        "pairs": mapping_pairs,
    }

    pairs_bytes = _serialize_pairs(pairs)
    mapping_bytes = _serialize_json(mapping)

    manifest = {
        "files": {
            "pairs.jsonl": {"sha256": _sha256_bytes(pairs_bytes), "bytes": len(pairs_bytes)},
            "mapping.json": {"sha256": _sha256_bytes(mapping_bytes), "bytes": len(mapping_bytes)},
        },
        "metadata": mapping,
    }
    manifest_bytes = _serialize_json(manifest)

    if out_path.exists() or out_path.is_symlink():
        raise FileExistsError(f"output path already exists: {out_path}")

    out_path.mkdir(parents=False)
    try:
        _atomic_write(out_path / "pairs.jsonl", pairs_bytes)
        _atomic_write(out_path / "mapping.json", mapping_bytes)
        _atomic_write(out_path / "pairs_manifest.json", manifest_bytes)
    except Exception:
        shutil.rmtree(out_path, ignore_errors=True)
        raise

    return pairs, mapping


def make_pairs(rows_sft, rows_dpo, seed=42):
    """Pure blinding function for dev use without bank-300 restriction."""
    if not isinstance(rows_sft, (list, tuple)) or not isinstance(rows_dpo, (list, tuple)):
        raise ValueError("rows must be lists or tuples")
    if len(rows_sft) != len(rows_dpo):
        raise ValueError(f"row count mismatch: SFT has {len(rows_sft)}, DPO has {len(rows_dpo)}")
    if len(rows_sft) == 0:
        raise ValueError("rows must be non-empty")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    for label, rows in (("sft", rows_sft), ("dpo", rows_dpo)):
        seen_ids = set()
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"{label}: each row must be a dict")
            for field in ("id", "instruction", "output"):
                if field not in row:
                    raise ValueError(f"{label}: row missing field {field!r}")
            _require_nonblank_string(row["id"], f"{label}: row id")
            if row["id"] in seen_ids:
                raise ValueError(f"{label}: duplicate id {row['id']!r}")
            seen_ids.add(row["id"])
            _require_nonblank_string(row["instruction"], f"{label}: row {row['id']!r} instruction")
            if not isinstance(row["output"], str):
                raise ValueError(f"{label}: row {row['id']!r} output must be a string")

    for sft_row, dpo_row in zip(rows_sft, rows_dpo):
        if sft_row["id"] != dpo_row["id"]:
            raise ValueError(f"id order mismatch: {sft_row['id']!r} vs {dpo_row['id']!r}")
        if sft_row["instruction"] != dpo_row["instruction"]:
            raise ValueError(f"instruction mismatch for id {sft_row['id']!r}")

    pairs, mapping_pairs = _blind(rows_sft, rows_dpo, seed)

    bank_kind = _row_bank(rows_sft) or 'unknown'
    if (_row_bank(rows_dpo) or 'unknown') != bank_kind:
        raise ValueError('development bank kinds differ')
    mapping = {
        "inputhashes": {},
        "bankkind": bank_kind,
        "banksha": "",
        "decoding": {},
        "seed": seed,
        "pairs": mapping_pairs,
    }
    return pairs, mapping


def summarize(pairs, mapping, judgments, failures, bootstrap_samples=2000, seed=42):
    """Compute result counts and bootstrap intervals for blinded pairs."""
    if not isinstance(pairs, (list, tuple)):
        raise ValueError("pairs must be a list or tuple")
    if not isinstance(mapping, dict):
        raise ValueError("mapping must be a dict")
    if not isinstance(judgments, (list, tuple)):
        raise ValueError("judgments must be a list or tuple")
    if not isinstance(failures, (list, tuple)):
        raise ValueError("failures must be a list or tuple")
    if isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, int) or bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    pair_ids = set()
    pair_instruction_map = {}
    for pair in pairs:
        if not isinstance(pair, dict):
            raise ValueError("each pair must be a dict")
        for field in ("id", "instruction", "response_a", "response_b"):
            if field not in pair:
                raise ValueError(f"pair missing field {field!r}")
        _require_nonblank_string(pair["id"], "pair id")
        if pair["id"] in pair_ids:
            raise ValueError(f"duplicate pair id {pair['id']!r}")
        pair_ids.add(pair["id"])
        _require_nonblank_string(pair["instruction"], f"pair {pair['id']!r} instruction")
        if not isinstance(pair["response_a"], str):
            raise ValueError(f"pair {pair['id']!r} response_a must be a string")
        if not isinstance(pair["response_b"], str):
            raise ValueError(f"pair {pair['id']!r} response_b must be a string")
        pair_instruction_map[pair["id"]] = pair["instruction"]

    mapping_pairs = mapping.get("pairs")
    if not isinstance(mapping_pairs, list):
        raise ValueError("mapping.pairs must be a list")
    mapping_ids = set()
    id_to_track = {}
    for mp in mapping_pairs:
        if not isinstance(mp, dict):
            raise ValueError("each mapping pair must be a dict")
        for field in ("id", "a", "b"):
            if field not in mp:
                raise ValueError(f"mapping pair missing field {field!r}")
        _require_nonblank_string(mp["id"], "mapping pair id")
        if mp["id"] in mapping_ids:
            raise ValueError(f"duplicate mapping pair id {mp['id']!r}")
        mapping_ids.add(mp["id"])
        if mp["a"] not in ("sft", "dpo") or mp["b"] not in ("sft", "dpo"):
            raise ValueError(f"mapping pair {mp['id']!r} has invalid track values")
        if mp["a"] == mp["b"]:
            raise ValueError(f"mapping pair {mp['id']!r} must have one A and one B")
        id_to_track[mp["id"]] = {"a": mp["a"], "b": mp["b"]}

    if pair_ids != mapping_ids:
        missing_in_mapping = pair_ids - mapping_ids
        missing_in_pairs = mapping_ids - pair_ids
        raise ValueError(
            f"ID sets do not match: {len(missing_in_mapping)} missing in mapping, "
            f"{len(missing_in_pairs)} missing in pairs"
        )

    judgment_map = {}
    for j in judgments:
        if not isinstance(j, dict):
            raise ValueError("each judgment must be a dict")
        if "id" not in j or "decision" not in j:
            raise ValueError("judgment missing id or decision")
        jid = j["id"]
        _require_nonblank_string(jid, "judgment id")
        if jid not in pair_ids:
            raise ValueError(f"judgment for unknown id {jid!r}")
        if jid in judgment_map:
            raise ValueError(f"duplicate judgment for id {jid!r}")
        decision = j["decision"]
        if decision not in _VALID_DECISIONS:
            raise ValueError(f"invalid decision {decision!r} for id {jid!r}")
        judgment_map[jid] = decision

    failure_ids = set()
    for f in failures:
        if not isinstance(f, dict):
            raise ValueError("each failure must be a dict")
        if "id" not in f:
            raise ValueError("failure missing id")
        fid = f["id"]
        _require_nonblank_string(fid, "failure id")
        if fid not in pair_ids:
            raise ValueError(f"failure for unknown id {fid!r}")
        failure_ids.add(fid)

    counts = {"total": len(pairs), "wins": 0, "ties": 0, "losses": 0, "unknown": 0, "failures": 0, "pending": 0}
    valid_pairs = []

    for pair in pairs:
        pid = pair["id"]
        if pid in judgment_map:
            decision = judgment_map[pid]
            if decision == "A":
                winner_track = id_to_track[pid]["a"]
            elif decision == "B":
                winner_track = id_to_track[pid]["b"]
            else:
                winner_track = None

            if decision == "A" or decision == "B":
                valid_pairs.append({"id": pid, "winner": winner_track})
                if winner_track == "dpo":
                    counts["wins"] += 1
                else:
                    counts["losses"] += 1
            elif decision == "tie":
                counts["ties"] += 1
                valid_pairs.append({"id": pid, "winner": None})
            else:
                counts["unknown"] += 1
        elif pid in failure_ids:
            counts["failures"] += 1
        else:
            counts["pending"] += 1

    w = counts["wins"]
    t = counts["ties"]
    l = counts["losses"]
    denominator = w + t + l
    tie_adjusted = (w + 0.5 * t) / denominator if denominator > 0 else None
    decisive = w / (w + l) if (w + l) > 0 else None

    bankkind = mapping.get("bankkind", "unknown")
    bootstrap_result = _bootstrap(valid_pairs, pair_instruction_map, bankkind, bootstrap_samples, seed)

    evidence = {
        "pairs_count": len(pairs),
        "mapping_pairs_count": len(mapping_pairs),
        "judgments_count": len(judgments),
        "failures_count": len(failures),
        "valid_pairs_count": len(valid_pairs),
        "bankkind": bankkind,
    }
    if bankkind == "hh_rlhf_eval":
        clusters = _cluster_by_instruction(pairs)
        evidence["effective_clusters"] = len(clusters)
        evidence["valid_clusters"] = len({_canonical_prompt(pair_instruction_map[p['id']])
                                           for p in valid_pairs})

    return {
        "perspective": "dpo_vs_sft",
        "counts": counts,
        "rates": {
            "tie_adjusted": tie_adjusted,
            "decisive": decisive,
        },
        "bootstrap": bootstrap_result,
        "evidence": evidence,
        "limitations": [
            "single-seed bootstrap",
            "intervals condition on successfully judged non-unknown pairs; missing judgments may be selective",
            "bootstrap intervals do not measure judge bias or training-seed variability",
        ],
    }


def _bootstrap(valid_pairs, pair_instruction_map, bankkind, bootstrap_samples, seed):
    if not valid_pairs:
        return {
            "tie_adjusted": {"ci_low": None, "ci_high": None, "samples": bootstrap_samples, "seed": seed},
            "decisive": {"ci_low": None, "ci_high": None, "samples": bootstrap_samples, "seed": seed},
        }

    rng = random.Random(seed)
    outcome_map = {vp["id"]: vp["winner"] for vp in valid_pairs}

    if bankkind == "hh_rlhf_eval":
        clusters = {}
        for vp in valid_pairs:
            pid = vp["id"]
            canonical = _canonical_prompt(pair_instruction_map[pid])
            if canonical not in clusters:
                clusters[canonical] = []
            clusters[canonical].append(pid)
        cluster_list = list(clusters.values())
        n_clusters = len(cluster_list)

        estimates, decisive_estimates = [], []
        for _ in range(bootstrap_samples):
            w, t, l = 0, 0, 0
            for _ in range(n_clusters):
                cluster_idx = rng.randrange(n_clusters)
                for pid in cluster_list[cluster_idx]:
                    outcome = outcome_map.get(pid)
                    if outcome == "dpo":
                        w += 1
                    elif outcome == "sft":
                        l += 1
                    elif outcome is None:
                        t += 1
            denom = w + t + l
            if denom > 0:
                estimates.append((w + 0.5 * t) / denom)
            if w + l:
                decisive_estimates.append(w / (w + l))
    else:
        valid_ids = [vp["id"] for vp in valid_pairs]
        n = len(valid_ids)

        estimates, decisive_estimates = [], []
        for _ in range(bootstrap_samples):
            w, t, l = 0, 0, 0
            for _ in range(n):
                pid = valid_ids[rng.randrange(n)]
                outcome = outcome_map[pid]
                if outcome == "dpo":
                    w += 1
                elif outcome == "sft":
                    l += 1
                else:
                    t += 1
            denom = w + t + l
            if denom > 0:
                estimates.append((w + 0.5 * t) / denom)
            if w + l:
                decisive_estimates.append(w / (w + l))

    estimates.sort()
    ci_low = _percentile(estimates, 2.5)
    ci_high = _percentile(estimates, 97.5)

    return {
        "tie_adjusted": {"ci_low": ci_low, "ci_high": ci_high, "samples": bootstrap_samples, "seed": seed},
        "decisive": {"ci_low": _percentile(sorted(decisive_estimates), 2.5),
                     "ci_high": _percentile(sorted(decisive_estimates), 97.5),
                     "samples": bootstrap_samples, "seed": seed},
    }
