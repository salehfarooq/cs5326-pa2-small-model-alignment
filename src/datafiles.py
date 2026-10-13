"""Deterministic artifact IO: read input banks, write prepared splits + manifest."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat as stat_module
import tempfile

from src.data import _VALID_SCHEMAS, _canonical_prompt, format_prompt

_FILTER_COUNT_KEYS = (
    "input",
    "kept",
    "invalid_record",
    "empty_text",
    "multi_turn",
    "prompt_mismatch",
    "identical_responses",
    "duplicate_prompt",
)


def _reject_constant(value):
    raise ValueError(f"non-finite JSON constant {value!r} is not allowed")


def parse_rows(payload, path):
    """Parse UTF-8 bytes into a row list, rejecting empty banks and bad syntax."""
    if not isinstance(payload, (bytes, bytearray)):
        raise ValueError(f"{path}: input must be UTF-8 bytes")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path}: invalid UTF-8: {exc}")

    suffix = os.path.splitext(str(path))[1].lower()
    if suffix == ".json":
        try:
            data = json.loads(text, parse_constant=_reject_constant)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{exc.lineno}: invalid JSON: {exc.msg}")
        except ValueError as exc:
            raise ValueError(f"{path}: {exc}")
        if not isinstance(data, list):
            raise ValueError(f"{path}: top-level JSON must be an array")
        if len(data) == 0:
            raise ValueError(f"{path}: input bank is empty")
        return data
    if suffix == ".jsonl":
        rows = []
        for line_no, line in enumerate(text.split("\n"), 1):
            if line.endswith("\r"):
                line = line[:-1]
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line, parse_constant=_reject_constant))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc.msg}")
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: {exc}")
        if len(rows) == 0:
            raise ValueError(f"{path}: input bank is empty")
        return rows
    raise ValueError(f"{path}: unsupported suffix {suffix!r}, expected .json or .jsonl")


def read_rows(path):
    """Read a .json/.jsonl input bank once and return its row list."""
    with open(path, "rb") as handle:
        payload = handle.read()
    return parse_rows(payload, path)


def read_normalized(path, kind, schema="normalized"):
    """Return clean rows, counts and byte provenance; stream raw JSONL input.

    Clean rows and deduplication state stay in memory; raw JSONL does not.
    """
    from src.data import normalize_records, normalize_rows

    suffix = os.path.splitext(str(path))[1].lower()
    if suffix == ".json":
        from pathlib import Path
        raw = Path(path).read_bytes()
        rows = parse_rows(raw, path)
        clean, counts = normalize_rows(rows, kind, schema)
        input_metadata = {
            "name": os.path.basename(str(path)),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }
        return clean, counts, input_metadata
    if suffix == ".jsonl":
        holder = {"hasher": hashlib.sha256(), "bytes": 0}

        def record_generator():
            with open(path, "rb") as handle:
                for line_no, raw_line in enumerate(handle, 1):
                    holder["hasher"].update(raw_line)
                    holder["bytes"] += len(raw_line)
                    if raw_line.endswith(b"\r\n"):
                        raw_line = raw_line[:-2]
                    elif raw_line.endswith(b"\n"):
                        raw_line = raw_line[:-1]
                    if not raw_line.strip():
                        continue
                    try:
                        text = raw_line.decode("utf-8")
                    except UnicodeDecodeError as exc:
                        raise ValueError(f"{path}:{line_no}: invalid UTF-8: {exc}")
                    try:
                        yield json.loads(text, parse_constant=_reject_constant)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"{path}:{line_no}: invalid JSON: {exc.msg}")
                    except ValueError as exc:
                        raise ValueError(f"{path}:{line_no}: {exc}")

        records = record_generator()
        try:
            clean, counts = normalize_records(records, kind, schema)
        finally:
            records.close()
        if counts["input"] == 0:
            raise ValueError(f"{path}: input bank is empty")
        input_metadata = {
            "name": os.path.basename(str(path)),
            "sha256": holder["hasher"].hexdigest(),
            "bytes": holder["bytes"],
        }
        return clean, counts, input_metadata
    raise ValueError(f"{path}: unsupported suffix {suffix!r}, expected .json or .jsonl")


def _require_nonblank_string(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonblank string")
    return value


def _validate_metadata(metadata):
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a dict")
    allowed_top = {"source", "input", "config"}
    unknown = set(metadata) - allowed_top
    if unknown:
        raise ValueError(f"unknown top-level metadata keys: {sorted(unknown)}")
    for ns in ("source", "input", "config"):
        if ns not in metadata:
            raise ValueError(f"metadata missing required namespace {ns!r}")
        if not isinstance(metadata[ns], dict):
            raise ValueError(f"metadata {ns!r} must be a dict")

    source = metadata["source"]
    for key in ("id", "revision", "split"):
        _require_nonblank_string(source.get(key), f"source.{key}")

    inp = metadata["input"]
    _require_nonblank_string(inp.get("name"), "input.name")
    sha = inp.get("sha256")
    if not isinstance(sha, str) or re.fullmatch(r"[0-9a-f]{64}", sha) is None:
        raise ValueError("input.sha256 must be 64 lowercase hex characters")
    nbytes = inp.get("bytes")
    if isinstance(nbytes, bool) or not isinstance(nbytes, int) or nbytes <= 0:
        raise ValueError("input.bytes must be a positive integer")

    config = metadata["config"]
    kind = config.get("kind")
    schema = config.get("schema")
    if not isinstance(kind, str) or kind not in _VALID_SCHEMAS:
        raise ValueError(f"unknown kind: {kind!r}")
    if not isinstance(schema, str) or schema not in _VALID_SCHEMAS[kind]:
        raise ValueError(f"unsupported kind/schema pair: {kind}/{schema}")
    seed = config.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("config.seed must be an integer")

    sizes = config.get("sizes")
    if not isinstance(sizes, dict) or not sizes:
        raise ValueError("config.sizes must be a non-empty dict")
    required = {"eval"} if kind == "eval" else {"train", "validation"}
    if set(sizes) != required:
        raise ValueError(f"config.sizes must have exactly {sorted(required)} for kind {kind!r}")
    for name, size in sizes.items():
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError(f"config.sizes[{name!r}] must be a positive integer")

    filter_counts = config.get("filter_counts")
    if not isinstance(filter_counts, dict):
        raise ValueError("config.filter_counts must be a dict")
    if set(filter_counts) != set(_FILTER_COUNT_KEYS):
        raise ValueError(f"config.filter_counts keys must be {sorted(_FILTER_COUNT_KEYS)}")
    for key, val in filter_counts.items():
        if isinstance(val, bool) or not isinstance(val, int) or val < 0:
            raise ValueError(f"config.filter_counts[{key!r}] must be a nonnegative integer")
    reason_sum = sum(filter_counts[k] for k in _FILTER_COUNT_KEYS if k not in ("input", "kept"))
    if filter_counts["input"] != filter_counts["kept"] + reason_sum:
        raise ValueError("filter_counts.input must equal kept + sum of reason counts")
    if filter_counts["kept"] < sum(sizes.values()):
        raise ValueError("filter_counts.kept must be >= sum of selected sizes")

    if "exclusions" in config:
        exclusions = config["exclusions"]
        if kind == "eval" or not isinstance(exclusions, dict):
            raise ValueError("config.exclusions is only supported for training kinds")
        removed = exclusions.get("removed")
        eligible = exclusions.get("eligible")
        if isinstance(removed, bool) or not isinstance(removed, int) or removed < 0:
            raise ValueError("exclusions.removed must be a nonnegative integer")
        if isinstance(eligible, bool) or not isinstance(eligible, int) or eligible < sum(sizes.values()):
            raise ValueError("exclusions.eligible must cover all selected rows")
        if removed + eligible != filter_counts["kept"]:
            raise ValueError("exclusions removed + eligible must equal filter_counts.kept")
        banks = exclusions.get("banks")
        if not isinstance(banks, list) or not banks:
            raise ValueError("exclusions.banks must be a nonempty list")
        for bank in banks:
            if not isinstance(bank, dict):
                raise ValueError("each exclusion bank must be a dict")
            _require_nonblank_string(bank.get("name"), "exclusion bank name")
            sha = bank.get("sha256")
            if not isinstance(sha, str) or re.fullmatch(r"[0-9a-f]{64}", sha) is None:
                raise ValueError("exclusion bank sha256 must be 64 lowercase hex characters")
            for key in ("bytes", "instructions", "unique_instructions"):
                value = bank.get(key)
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ValueError(f"exclusion bank {key} must be a positive integer")
            if bank["unique_instructions"] > bank["instructions"]:
                raise ValueError("exclusion bank unique instructions cannot exceed input instructions")

    if "token_filter" in config:
        token_filter = config["token_filter"]
        if kind == "eval" or not isinstance(token_filter, dict):
            raise ValueError("config.token_filter is only supported for training kinds")
        _require_nonblank_string(token_filter.get("model_id"), "token_filter.model_id")
        revision = token_filter.get("revision")
        if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise ValueError("token_filter.revision must be a pinned lowercase commit")
        for key in ("max_length", "max_prompt_tokens"):
            value = token_filter.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"token_filter.{key} must be a positive integer")
        if token_filter["max_prompt_tokens"] >= token_filter["max_length"]:
            raise ValueError("token_filter prompt budget must leave response space")
        if token_filter.get("policy") != "preserve_full_prompt_filter":
            raise ValueError("unknown token eligibility policy")
        token_counts = token_filter.get("counts")
        keys = {"input", "kept", "prompt_too_long", "empty_tokenized_response", "identical_retained_pair"}
        if not isinstance(token_counts, dict) or set(token_counts) != keys:
            raise ValueError("invalid token eligibility counter keys")
        for key, value in token_counts.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"token eligibility {key} must be a nonnegative integer")
        if token_counts["input"] != token_counts["kept"] + sum(token_counts[key] for key in keys - {"input", "kept"}):
            raise ValueError("token eligibility counters do not sum to input")
        previous_kept = config["exclusions"]["eligible"] if "exclusions" in config else filter_counts["kept"]
        if token_counts["input"] != previous_kept or token_counts["kept"] < sum(sizes.values()):
            raise ValueError("token eligibility input/selected quotas do not match preceding stage")

    template = config.get("prompt_template")
    if template != format_prompt("{instruction}"):
        raise ValueError("config.prompt_template must equal format_prompt('{instruction}')")


def _validate_row(row, kind, seen_ids, seen_source_indices, seen_prompts):
    if not isinstance(row, dict):
        raise ValueError("each row must be a dict")
    row_id = row.get("id")
    _require_nonblank_string(row_id, "row id")
    if row_id in seen_ids:
        raise ValueError(f"duplicate id {row_id!r}")
    seen_ids.add(row_id)
    source_index = row.get("source_index")
    if isinstance(source_index, bool) or not isinstance(source_index, int) or source_index < 0:
        raise ValueError(f"row {row_id!r} source_index must be a nonnegative integer")
    if source_index in seen_source_indices:
        raise ValueError(f"duplicate source_index {source_index}")
    seen_source_indices.add(source_index)

    if kind == "sft":
        prompt = row.get("prompt")
        response = row.get("response")
        _require_nonblank_string(prompt, f"row {row_id!r} prompt")
        _require_nonblank_string(response, f"row {row_id!r} response")
        canonical = _canonical_prompt(prompt)
    elif kind == "dpo":
        prompt = row.get("prompt")
        chosen = row.get("chosen")
        rejected = row.get("rejected")
        _require_nonblank_string(prompt, f"row {row_id!r} prompt")
        _require_nonblank_string(chosen, f"row {row_id!r} chosen")
        _require_nonblank_string(rejected, f"row {row_id!r} rejected")
        if _canonical_prompt(chosen) == _canonical_prompt(rejected):
            raise ValueError(f"row {row_id!r} chosen/rejected cannot be canonically identical")
        canonical = _canonical_prompt(prompt)
    else:
        instruction = row.get("instruction")
        _require_nonblank_string(instruction, f"row {row_id!r} instruction")
        for opt in ("dataset", "category", "expected_behavior"):
            if opt in row:
                val = row[opt]
                if not isinstance(val, str):
                    raise ValueError(f"row {row_id!r} field {opt!r} must be a string when present")
        canonical = _canonical_prompt(instruction)

    if canonical in seen_prompts:
        raise ValueError(f"duplicate canonical prompt for id {row_id!r}")
    seen_prompts.add(canonical)


def _validate_splits(splits, kind, sizes):
    if not isinstance(splits, dict) or not splits:
        raise ValueError("splits must be a non-empty dict")
    required = {"eval"} if kind == "eval" else {"train", "validation"}
    if set(splits) != required:
        raise ValueError(f"splits must have exactly {sorted(required)} for kind {kind!r}")

    seen_ids = set()
    seen_source_indices = set()
    seen_prompts = set()
    for name in sorted(splits):
        rows = splits[name]
        if not isinstance(rows, (list, tuple)):
            raise ValueError(f"split {name!r} must be a list or tuple")
        if len(rows) == 0:
            raise ValueError(f"split {name!r} must be non-empty")
        if len(rows) != sizes[name]:
            raise ValueError(f"split {name!r} has {len(rows)} rows but size is {sizes[name]}")
        for row in rows:
            _validate_row(row, kind, seen_ids, seen_source_indices, seen_prompts)


def _serialize_split(rows):
    lines = []
    for row in rows:
        lines.append(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _serialize_json(obj):
    return (json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _build_manifest(metadata, serialized, splits):
    split_entries = {}
    for name in sorted(splits):
        rows = splits[name]
        split_entries[name] = {
            "file": name + ".jsonl",
            "rows": len(rows),
            "sha256": hashlib.sha256(serialized[name]).hexdigest(),
            "ids": [row["id"] for row in rows],
            "source_indices": [row["source_index"] for row in rows],
        }
    return {
        "version": 1,
        "source": metadata["source"],
        "input": metadata["input"],
        "config": metadata["config"],
        "splits": split_entries,
    }


def _remove_staging(staging):
    shutil.rmtree(staging, ignore_errors=True)


def _rollback(output_dir, created_links, staging, owned_dir_stat):
    if owned_dir_stat is not None:
        try:
            current = os.lstat(output_dir)
        except OSError:
            current = None
        if (current is not None and current.st_ino == owned_dir_stat.st_ino
                and current.st_dev == owned_dir_stat.st_dev
                and stat_module.S_ISDIR(current.st_mode)):
            for name, ino, dev in created_links:
                path = os.path.join(output_dir, name)
                try:
                    stat = os.lstat(path)
                except OSError:
                    continue
                if (stat.st_ino == ino and stat.st_dev == dev
                        and stat_module.S_ISREG(stat.st_mode)):
                    os.remove(path)
            try:
                os.rmdir(output_dir)
            except OSError:
                pass
    _remove_staging(staging)


def write_prepared(output_dir, splits, metadata):
    """Validate and serialize everything, then publish split files and manifest.

    Manifest is the completion marker; each link is an individual atomic
    exclusive publication, not a whole-directory atomic operation.
    """
    _validate_metadata(metadata)
    config = metadata["config"]
    kind = config["kind"]
    sizes = config["sizes"]
    _validate_splits(splits, kind, sizes)

    output_dir = os.path.abspath(output_dir)
    parent = os.path.dirname(output_dir)
    if not os.path.isdir(parent):
        raise ValueError(f"output parent does not exist: {parent}")
    if os.path.lexists(output_dir):
        raise FileExistsError(f"output path already exists: {output_dir}")

    serialized = {}
    for name in sorted(splits):
        serialized[name] = _serialize_split(splits[name])
    manifest = _build_manifest(metadata, serialized, splits)
    manifest_bytes = _serialize_json(manifest)

    staging = tempfile.mkdtemp(prefix=".prepare-", dir=parent)
    created_links = []
    owned_dir_stat = None
    try:
        staged_paths = {}
        for name in sorted(splits):
            staged = os.path.join(staging, name + ".jsonl")
            with open(staged, "wb") as handle:
                handle.write(serialized[name])
            staged_paths[name] = staged

        staged_manifest = os.path.join(staging, "manifest.json")
        with open(staged_manifest, "wb") as handle:
            handle.write(manifest_bytes)

        os.mkdir(output_dir)
        owned_dir_stat = os.lstat(output_dir)

        for name in sorted(splits):
            src = staged_paths[name]
            src_stat = os.lstat(src)
            dst = os.path.join(output_dir, name + ".jsonl")
            os.link(src, dst)
            created_links.append((name + ".jsonl", src_stat.st_ino, src_stat.st_dev))

        manifest_src_stat = os.lstat(staged_manifest)
        os.link(staged_manifest, os.path.join(output_dir, "manifest.json"))
        created_links.append(("manifest.json", manifest_src_stat.st_ino, manifest_src_stat.st_dev))
    except Exception:
        _rollback(output_dir, created_links, staging, owned_dir_stat)
        raise
    else:
        _remove_staging(staging)
