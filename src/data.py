"""Deterministic data normalization for SFT/DPO/eval preparation.

Supported kind/schema pairs:
- sft/normalized, sft/ultrachat, sft/ultrachat_first_turn
- dpo/normalized, dpo/hh, dpo/ultrafeedback
- eval/normalized

Row containers must be a list or tuple of raw row mappings. Each rejected row
is assigned exactly one primary reason; reason counts sum to the input count.
Clean rows keep first occurrence per canonical prompt (NFC, CRLF/CR -> LF,
outer strip; internal whitespace preserved). IDs come from a nonblank raw
id/prompt_id when present, else a SHA256 of the canonical normalized content
(including kind). A duplicate ID with different normalized content is fatal.
Generic eval normalization/sampling is for newly prepared evaluation banks.
Use validate_fixed_eval for the supplied HH bank: repeated instructions with
distinct IDs must remain in original order.

UltraFeedback prompt_id is a source prompt-group identifier. Different pairs
for that same canonical prompt are deduplicated by first occurrence; reuse
across different prompts and conflicts in explicit row id remain fatal.
"""

from __future__ import annotations

import hashlib
import json
import random
import unicodedata

_VALID_SCHEMAS = {
    "sft": {"normalized", "ultrachat", "ultrachat_first_turn"},
    "dpo": {"normalized", "hh", "ultrafeedback"},
    "eval": {"normalized"},
}

_HUMAN_MARKER = "\n\nHuman: "
_ASSISTANT_MARKER = "\n\nAssistant: "

_DEDUP_KEY = {"sft": "prompt", "dpo": "prompt", "eval": "instruction"}


def format_prompt(instruction):
    """Return the saved Qwen instruction template for a single instruction."""
    if not isinstance(instruction, str):
        raise ValueError(f"instruction must be a string, got {type(instruction).__name__}")
    if not instruction.strip():
        raise ValueError("instruction must be non-blank")
    return f"Instruction:\n{instruction}\n\nResponse:\n"


def _canonical_prompt(text):
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def _content_signature(kind, fields):
    payload = {"kind": kind, "fields": {key: _canonical_prompt(fields[key]) for key in sorted(fields)}}
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _assign_id(row, kind, fields):
    if isinstance(row, dict):
        raw_id = row.get("id")
        if isinstance(raw_id, str) and raw_id.strip():
            return raw_id
        prompt_id = row.get("prompt_id")
        if isinstance(prompt_id, str) and prompt_id.strip():
            return prompt_id
    sig = _content_signature(kind, fields)
    return hashlib.sha256(sig.encode("utf-8")).hexdigest()


def _parse_hh_transcript(text):
    if not isinstance(text, str):
        return None, "invalid_record"
    human_count = text.count(_HUMAN_MARKER)
    assistant_count = text.count(_ASSISTANT_MARKER)
    if human_count > 1 or assistant_count > 1:
        return None, "multi_turn"
    if not text.startswith(_HUMAN_MARKER) or assistant_count != 1:
        return None, "invalid_record"
    rest = text[len(_HUMAN_MARKER):]
    prompt, answer = rest.split(_ASSISTANT_MARKER, 1)
    return (prompt, answer), None


def _extract_normalized_sft(row):
    if not isinstance(row, dict):
        return None, "invalid_record"
    prompt = row.get("prompt")
    response = row.get("response")
    if not isinstance(prompt, str) or not isinstance(response, str):
        return None, "invalid_record"
    if not prompt.strip() or not response.strip():
        return None, "empty_text"
    return {"prompt": prompt, "response": response}, None


def _extract_normalized_dpo(row):
    if not isinstance(row, dict):
        return None, "invalid_record"
    prompt = row.get("prompt")
    chosen = row.get("chosen")
    rejected = row.get("rejected")
    if not isinstance(prompt, str) or not isinstance(chosen, str) or not isinstance(rejected, str):
        return None, "invalid_record"
    if not prompt.strip() or not chosen.strip() or not rejected.strip():
        return None, "empty_text"
    if _canonical_prompt(chosen) == _canonical_prompt(rejected):
        return None, "identical_responses"
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}, None


def _extract_normalized_eval(row):
    if not isinstance(row, dict):
        return None, "invalid_record"
    instruction = row.get("instruction")
    if not isinstance(instruction, str):
        return None, "invalid_record"
    if not instruction.strip():
        return None, "empty_text"
    fields = {"instruction": instruction}
    for opt in ("dataset", "category", "expected_behavior"):
        if opt in row:
            val = row[opt]
            if not isinstance(val, str):
                return None, "invalid_record"
            fields[opt] = val
    return fields, None


def _extract_ultrachat(row):
    if not isinstance(row, dict):
        return None, "invalid_record"
    messages = row.get("messages")
    if not isinstance(messages, list):
        return None, "invalid_record"
    if len(messages) > 2:
        return None, "multi_turn"
    if len(messages) != 2:
        return None, "invalid_record"
    first, second = messages
    if not isinstance(first, dict) or not isinstance(second, dict):
        return None, "invalid_record"
    if first.get("role") != "user" or second.get("role") != "assistant":
        return None, "invalid_record"
    user_content = first.get("content")
    assistant_content = second.get("content")
    if not isinstance(user_content, str) or not isinstance(assistant_content, str):
        return None, "invalid_record"
    if not user_content.strip() or not assistant_content.strip():
        return None, "empty_text"
    if "prompt" in row:
        prompt_col = row["prompt"]
        if not isinstance(prompt_col, str) or prompt_col != user_content:
            return None, "prompt_mismatch"
    return {"prompt": user_content, "response": assistant_content}, None


def _extract_ultrafeedback(row):
    if not isinstance(row, dict):
        return None, "invalid_record"
    chosen = row.get("chosen")
    rejected = row.get("rejected")
    if not isinstance(chosen, list) or not isinstance(rejected, list):
        return None, "invalid_record"
    if len(chosen) > 2 or len(rejected) > 2:
        return None, "multi_turn"
    if len(chosen) != 2 or len(rejected) != 2:
        return None, "invalid_record"
    for side in (chosen, rejected):
        if not isinstance(side[0], dict) or not isinstance(side[1], dict):
            return None, "invalid_record"
        if side[0].get("role") != "user" or side[1].get("role") != "assistant":
            return None, "invalid_record"
    chosen_user = chosen[0].get("content")
    chosen_asst = chosen[1].get("content")
    rejected_user = rejected[0].get("content")
    rejected_asst = rejected[1].get("content")
    if not all(isinstance(x, str) for x in (chosen_user, chosen_asst, rejected_user, rejected_asst)):
        return None, "invalid_record"
    if not chosen_user.strip() or not chosen_asst.strip() or not rejected_user.strip() or not rejected_asst.strip():
        return None, "empty_text"
    if chosen_user != rejected_user:
        return None, "prompt_mismatch"
    if "prompt" in row:
        top_prompt = row["prompt"]
        if not isinstance(top_prompt, str) or top_prompt != chosen_user:
            return None, "prompt_mismatch"
    if _canonical_prompt(chosen_asst) == _canonical_prompt(rejected_asst):
        return None, "identical_responses"
    return {"prompt": chosen_user, "chosen": chosen_asst, "rejected": rejected_asst}, None


def _extract_hh(row):
    if not isinstance(row, dict):
        return None, "invalid_record"
    chosen = row.get("chosen")
    rejected = row.get("rejected")
    if not isinstance(chosen, str) or not isinstance(rejected, str):
        return None, "invalid_record"
    chosen_parsed, chosen_reason = _parse_hh_transcript(chosen)
    if chosen_parsed is None:
        return None, chosen_reason
    rejected_parsed, rejected_reason = _parse_hh_transcript(rejected)
    if rejected_parsed is None:
        return None, rejected_reason
    chosen_prompt, chosen_answer = chosen_parsed
    rejected_prompt, rejected_answer = rejected_parsed
    if not chosen_prompt.strip() or not chosen_answer.strip() or not rejected_prompt.strip() or not rejected_answer.strip():
        return None, "empty_text"
    if chosen_prompt != rejected_prompt:
        return None, "prompt_mismatch"
    if _canonical_prompt(chosen_answer) == _canonical_prompt(rejected_answer):
        return None, "identical_responses"
    return {"prompt": chosen_prompt, "chosen": chosen_answer, "rejected": rejected_answer}, None


def _extract_ultrachat_first_turn(row):
    if not isinstance(row, dict):
        return None, "invalid_record"
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2 or len(messages) % 2 != 0:
        return None, "invalid_record"
    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            return None, "invalid_record"
        expected_role = "user" if idx % 2 == 0 else "assistant"
        if msg.get("role") != expected_role:
            return None, "invalid_record"
        content = msg.get("content")
        if not isinstance(content, str):
            return None, "invalid_record"
        if not content.strip():
            return None, "empty_text"
    user_content = messages[0]["content"]
    assistant_content = messages[1]["content"]
    if "prompt" in row:
        prompt_col = row["prompt"]
        if not isinstance(prompt_col, str) or prompt_col != user_content:
            return None, "prompt_mismatch"
        prompt = prompt_col
    else:
        prompt = user_content
    return {"prompt": prompt, "response": assistant_content}, None


_EXTRACTORS = {
    ("sft", "normalized"): _extract_normalized_sft,
    ("sft", "ultrachat"): _extract_ultrachat,
    ("sft", "ultrachat_first_turn"): _extract_ultrachat_first_turn,
    ("dpo", "normalized"): _extract_normalized_dpo,
    ("dpo", "hh"): _extract_hh,
    ("dpo", "ultrafeedback"): _extract_ultrafeedback,
    ("eval", "normalized"): _extract_normalized_eval,
}


def normalize_records(records, kind, schema="normalized"):
    """Stream-normalize an iterable of raw records once into (clean_rows, counts)."""
    if not isinstance(kind, str) or kind not in _VALID_SCHEMAS:
        raise ValueError(f"unknown kind: {kind!r}")
    if not isinstance(schema, str) or schema not in _VALID_SCHEMAS[kind]:
        raise ValueError(f"unsupported kind/schema pair: {kind}/{schema}")
    if isinstance(records, (str, bytes, dict)):
        raise ValueError("records must be an iterable of dicts, not str/bytes/dict")
    try:
        iterator = iter(records)
    except TypeError:
        raise ValueError("records must be an iterable")

    counts = {
        "input": 0,
        "kept": 0,
        "invalid_record": 0,
        "empty_text": 0,
        "multi_turn": 0,
        "prompt_mismatch": 0,
        "identical_responses": 0,
        "duplicate_prompt": 0,
    }

    extractor = _EXTRACTORS[(kind, schema)]
    id_to_sig = {}
    seen_prompts = set()
    clean_rows = []
    for row in iterator:
        fields, reason = extractor(row)
        if reason is not None:
            counts[reason] += 1
            counts["input"] += 1
            continue
        row_id = _assign_id(row, kind, fields)
        sig = hashlib.sha256(_content_signature(kind, fields).encode("utf-8")).hexdigest()
        canonical = _canonical_prompt(fields[_DEDUP_KEY[kind]])
        prompt_sig = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if row_id in id_to_sig:
            previous_sig, previous_prompt = id_to_sig[row_id]
            raw_id = row.get("id")
            # UltraFeedback's prompt_id groups pairs for one prompt, rather
            # than promising a unique preference row. Explicit row IDs stay strict.
            prompt_group = schema == "ultrafeedback" and not (isinstance(raw_id, str) and raw_id.strip())
            if previous_sig != sig and not (prompt_group and previous_prompt == prompt_sig):
                raise ValueError(f"duplicate id {row_id!r} with different normalized content")
            counts["duplicate_prompt"] += 1
            counts["input"] += 1
            continue
        id_to_sig[row_id] = (sig, prompt_sig)
        if canonical in seen_prompts:
            counts["duplicate_prompt"] += 1
            counts["input"] += 1
            continue
        seen_prompts.add(canonical)
        clean_row = {
            "id": row_id,
            "source_index": counts["input"],
            **fields,
        }
        clean_rows.append(clean_row)
        counts["input"] += 1

    counts["kept"] = len(clean_rows)
    return clean_rows, counts


def normalize_rows(rows, kind, schema="normalized"):
    """Return (clean_rows, counts) for the given kind/schema pair."""
    if not isinstance(rows, (list, tuple)):
        raise ValueError("rows must be a list or tuple")
    return normalize_records(rows, kind, schema)


def _row_id_and_key(row, key_field):
    if not isinstance(row, dict):
        return None, None
    row_id = row.get("id")
    if not isinstance(row_id, str) or not row_id.strip():
        return None, None
    key_text = row.get(key_field)
    if not isinstance(key_text, str) or not key_text.strip():
        return None, None
    return row_id, _canonical_prompt(key_text)


def _validate_unique_rows(rows, key_field):
    seen_ids = set()
    seen_keys = set()
    for row in rows:
        row_id, key = _row_id_and_key(row, key_field)
        if row_id is None:
            raise ValueError(f"each row must have a nonblank string id and a nonblank {key_field}")
        if row_id in seen_ids:
            raise ValueError(f"duplicate id {row_id!r}")
        seen_ids.add(row_id)
        if key in seen_keys:
            raise ValueError(f"duplicate canonical {key_field} for id {row_id!r}")
        seen_keys.add(key)


def _check_seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed must be an integer, got {seed!r}")


def _shuffled_indices(rows, seed):
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    return indices


def split_rows(rows, train_size, validation_size, seed=42):
    """Return (train, validation) with exact disjoint quotas and no mutation."""
    if not isinstance(rows, (list, tuple)):
        raise ValueError("rows must be a list or tuple")
    if isinstance(train_size, bool) or not isinstance(train_size, int) or train_size <= 0:
        raise ValueError(f"train_size must be a positive integer, got {train_size!r}")
    if isinstance(validation_size, bool) or not isinstance(validation_size, int) or validation_size <= 0:
        raise ValueError(f"validation_size must be a positive integer, got {validation_size!r}")
    _check_seed(seed)
    _validate_unique_rows(rows, "prompt")

    if len(rows) < train_size + validation_size:
        raise ValueError(f"not enough rows: need {train_size + validation_size}, have {len(rows)}")

    indices = _shuffled_indices(rows, seed)
    train = [rows[i] for i in indices[:train_size]]
    validation = [rows[i] for i in indices[train_size:train_size + validation_size]]
    return train, validation


def select_eval(rows, count=300, seed=42):
    """Select unique instructions for a new eval bank using a local RNG."""
    if not isinstance(rows, (list, tuple)):
        raise ValueError("rows must be a list or tuple")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError(f"count must be a positive integer, got {count!r}")
    _check_seed(seed)
    _validate_unique_rows(rows, "instruction")

    if len(rows) < count:
        raise ValueError(f"not enough rows: need {count}, have {len(rows)}")

    indices = _shuffled_indices(rows, seed)
    return [rows[i] for i in indices[:count]]


def validate_fixed_eval(rows):
    """Return fresh shallow copies of a fixed eval bank, preserving all fields/order."""
    if not isinstance(rows, (list, tuple)):
        raise ValueError("rows must be a list or tuple")
    if len(rows) == 0:
        raise ValueError("rows must be non-empty")

    seen_ids = set()
    clean_rows = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"each row must be a dict, got {type(row).__name__}")
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id.strip():
            raise ValueError("each row must have a nonblank string id")
        if row_id in seen_ids:
            raise ValueError(f"duplicate id {row_id!r}")
        seen_ids.add(row_id)
        instruction = row.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"row {row_id!r} must have a nonblank string instruction")
        for opt in ("dataset", "category", "expected_behavior"):
            if opt in row:
                val = row[opt]
                if not isinstance(val, str):
                    raise ValueError(f"row {row_id!r} field {opt!r} must be a string when present")
        clean_rows.append(dict(row))
    return clean_rows
