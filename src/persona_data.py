"""Prepare conversational preference records for alignment training."""

import random


def prepare_data(rows, seed=42):
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError("rows must be a list with at least 2 entries")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an int")

    seen_prompts = set()
    unique_idxs = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("each row must be a dict")
        prompt = row.get("prompt")
        chosen = row.get("chosen")
        rejected = row.get("rejected")
        if not isinstance(prompt, str) or len(prompt.strip()) == 0:
            raise ValueError("prompt must be a nonempty string")
        if not isinstance(chosen, str) or len(chosen.strip()) == 0:
            raise ValueError("chosen must be a nonempty string")
        if not isinstance(rejected, str) or len(rejected.strip()) == 0:
            raise ValueError("rejected must be a nonempty string")
        if chosen == rejected:
            raise ValueError("chosen and rejected must be distinct")
        canonical = " ".join(prompt.split()).casefold()
        if canonical in seen_prompts:
            raise ValueError("duplicate canonical prompt")
        seen_prompts.add(canonical)
        unique_idxs.append(idx)

    idxs = list(unique_idxs)
    rng = random.Random(seed)
    rng.shuffle(idxs)

    eval_count = max(1, round(0.1 * len(rows)))
    validation_ids = idxs[:eval_count]
    train_ids = idxs[eval_count:]

    if len(train_ids) == 0:
        raise ValueError("train split is empty")

    def make_record(row):
        return {
            "prompt": [{"role": "user", "content": row["prompt"]}],
            "chosen": [{"role": "assistant", "content": row["chosen"]}],
            "rejected": [{"role": "assistant", "content": row["rejected"]}],
        }

    train_records = [make_record(rows[idx]) for idx in train_ids]
    evaluation_records = [make_record(rows[idx]) for idx in validation_ids]

    metadata = {
        "train_ids": train_ids,
        "validation_ids": validation_ids,
        "seed": seed,
    }
    return train_records, evaluation_records, metadata
