"""Training eligibility for a fixed tokenizer/context; no model or downloads."""

from src.data import format_prompt


def filter_for_context(rows, kind, tokenizer, max_length=512, max_prompt_tokens=256, batch_size=128):
    """Keep full prompts that fit the budget and distinguishable retained pairs.

    Responses follow the adapter's EOS-before-right-truncation contract. Rows
    are returned in source order without changing text, IDs or source indices.
    """
    if not isinstance(rows, (list, tuple)) or kind not in ("sft", "dpo"):
        raise ValueError("rows must be a list/tuple and kind must be sft/dpo")
    for name, value in [("max_length", max_length), ("max_prompt_tokens", max_prompt_tokens), ("batch_size", batch_size)]:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if max_prompt_tokens >= max_length:
        raise ValueError("max_prompt_tokens must leave response space within max_length")
    eos = tokenizer.eos_token_id
    if isinstance(eos, bool) or not isinstance(eos, int) or eos < 0:
        raise ValueError("tokenizer requires an integer EOS token ID")
    keys = ["response"] if kind == "sft" else ["chosen", "rejected"]
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each row must be a dict")
        for key in ["prompt"] + keys:
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"each row requires nonblank {key}")

    counts = {"input": len(rows), "kept": 0, "prompt_too_long": 0,
              "empty_tokenized_response": 0, "identical_retained_pair": 0}
    clean = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        prompts = tokenizer([format_prompt(row["prompt"]) for row in batch],
                            add_special_tokens=False)["input_ids"]
        if len(prompts) != len(batch):
            raise ValueError("tokenizer returned the wrong prompt batch size")
        eligible, prompt_sizes = [], []
        for row, tokens in zip(batch, prompts):
            if not tokens:
                raise ValueError("formatted prompt tokenizes to an empty sequence")
            if len(tokens) > max_prompt_tokens:
                counts["prompt_too_long"] += 1
            else:
                eligible.append(row)
                prompt_sizes.append(len(tokens))
        if not eligible:
            continue
        texts = [row[key] for row in eligible for key in keys]
        responses = tokenizer(texts, add_special_tokens=False, truncation=True,
                              max_length=max_length)["input_ids"]
        if len(responses) != len(texts):
            raise ValueError("tokenizer returned the wrong response batch size")
        for index, (row, prompt_size) in enumerate(zip(eligible, prompt_sizes)):
            sides = responses[index * len(keys):(index + 1) * len(keys)]
            if any(not side for side in sides):
                counts["empty_tokenized_response"] += 1
                continue
            # Truncating the tokenizer output to max_length cannot affect the
            # retained prefix: a nonempty prompt always leaves less than that.
            retained = [(side + [eos])[:max_length - prompt_size] for side in sides]
            if kind == "dpo" and retained[0] == retained[1]:
                counts["identical_retained_pair"] += 1
                continue
            clean.append(row)
    counts["kept"] = len(clean)
    return clean, counts
