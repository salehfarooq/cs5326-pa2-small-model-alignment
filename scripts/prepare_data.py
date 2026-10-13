"""CLI for deterministic data preparation: read, normalize, split, write artifacts."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from pathlib import Path

from src.data import _canonical_prompt, split_rows, select_eval, format_prompt
from src.datafiles import parse_rows, read_normalized, write_prepared

FIXED_EVAL_BANK_SHA256 = "0731105407d06152e584c722717712be4f418f221a8d5c487bb924c4d91c38c6"
FIXED_EVAL_BANK_PATH = Path(__file__).resolve().parents[1] / "eval_sets/hh_rlhf_eval_300_seed42.json"

_DEFAULT_SIZES = {
    "sft": {"train": 30000, "validation": 1000},
    "dpo": {"train": 15000, "validation": 1000},
    "eval": {"eval": 300},
}


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Prepare deterministic SFT/DPO/eval artifacts")
    parser.add_argument("--input", required=True, help="input .json or .jsonl bank")
    parser.add_argument("--kind", required=True, choices=["sft", "dpo", "eval"])
    parser.add_argument("--schema", default="normalized")
    parser.add_argument("--output-dir", required=True, help="new output directory (must not exist)")
    parser.add_argument("--source-id", required=True, help="provenance source id")
    parser.add_argument("--revision", required=True, help="provenance source revision")
    parser.add_argument("--source-split", required=True, help="provenance source split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--validation-size", type=int, default=None)
    parser.add_argument("--eval-size", type=int, default=None)
    parser.add_argument("--exclude-eval", action="append", default=[], metavar="PATH",
                        help="exclude these evaluation instructions before training split selection (repeatable)")
    parser.add_argument("--tokenizer-id", help="optional cached tokenizer for training eligibility")
    parser.add_argument("--tokenizer-revision", help="pinned 40-character tokenizer commit")
    parser.add_argument("--tokenizer-cache", help="local Hugging Face cache; this command never downloads")
    parser.add_argument("--max-length", type=int, help="token eligibility context, default 512")
    parser.add_argument("--max-prompt-tokens", type=int, help="retain complete prompts up to this size, default 256")
    return parser.parse_args(argv)


def _nonblank(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonblank string")
    return value


def _resolve_sizes(args):
    defaults = _DEFAULT_SIZES[args.kind]
    if args.kind in ("sft", "dpo"):
        if args.eval_size is not None:
            raise ValueError("--eval-size is not supported for sft/dpo")
        train_size = args.train_size if args.train_size is not None else defaults["train"]
        validation_size = args.validation_size if args.validation_size is not None else defaults["validation"]
        return {"train": train_size, "validation": validation_size}
    if args.train_size is not None or args.validation_size is not None:
        raise ValueError("--train-size/--validation-size are not supported for eval")
    eval_size = args.eval_size if args.eval_size is not None else defaults["eval"]
    return {"eval": eval_size}


def _is_fixed_eval_bank(input_path, sha256):
    if sha256 == FIXED_EVAL_BANK_SHA256:
        return True
    return Path(input_path).resolve() == FIXED_EVAL_BANK_PATH.resolve()


def _read_exclusions(paths):
    instructions, banks, seen_paths = set(), [], set()
    for path in paths:
        resolved = Path(path).resolve()
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        raw = Path(path).read_bytes()
        rows = parse_rows(raw, path)
        keys = set()
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"{path}: exclusion row {index} must be a dict")
            instruction = row.get("instruction")
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError(f"{path}: exclusion row {index} needs a nonblank instruction")
            keys.add(_canonical_prompt(instruction))
        instructions.update(keys)
        banks.append({"name": Path(path).name, "sha256": hashlib.sha256(raw).hexdigest(),
                      "bytes": len(raw), "instructions": len(rows), "unique_instructions": len(keys)})
    return instructions, banks


def main(argv=None):
    args = _parse_args(argv)

    _nonblank(args.source_id, "--source-id")
    _nonblank(args.revision, "--revision")
    _nonblank(args.source_split, "--source-split")

    sizes = _resolve_sizes(args)

    if args.kind == "eval" and args.exclude_eval:
        raise ValueError("--exclude-eval is only supported for sft/dpo")
    if args.kind == "eval" and _is_fixed_eval_bank(args.input, None):
        raise ValueError("input is the fixed HH eval bank; use validate_fixed_eval to preserve all 300 rows")
    tokenizer = None
    if args.tokenizer_id is not None:
        if args.kind == "eval":
            raise ValueError("token eligibility filtering is only supported for sft/dpo")
        _nonblank(args.tokenizer_id, "--tokenizer-id")
        if args.tokenizer_revision is None or re.fullmatch(r"[0-9a-f]{40}", args.tokenizer_revision) is None:
            raise ValueError("--tokenizer-revision requires a pinned 40-character lowercase commit")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id, revision=args.tokenizer_revision,
            cache_dir=args.tokenizer_cache, local_files_only=True, token=False)
    elif any(value is not None for value in [args.tokenizer_revision, args.tokenizer_cache, args.max_length, args.max_prompt_tokens]):
        raise ValueError("tokenization flags require --tokenizer-id and --tokenizer-revision")
    excluded_keys, exclusion_banks = _read_exclusions(args.exclude_eval)

    clean_rows, counts, input_metadata = read_normalized(args.input, args.kind, args.schema)

    if args.kind == "eval" and _is_fixed_eval_bank(args.input, input_metadata["sha256"]):
        raise ValueError(
            "input is the fixed HH eval bank; use validate_fixed_eval to preserve its "
            "300 authoritative IDs/order instead of generic eval preparation"
        )

    exclusions = None
    if exclusion_banks:
        clean_rows = [row for row in clean_rows if _canonical_prompt(row["prompt"]) not in excluded_keys]
        exclusions = {"banks": exclusion_banks, "removed": counts["kept"] - len(clean_rows),
                      "eligible": len(clean_rows)}

    token_filter = None
    if tokenizer is not None:
        from src.data_tokens import filter_for_context
        context = args.max_length if args.max_length is not None else 512
        prompt_limit = args.max_prompt_tokens if args.max_prompt_tokens is not None else 256
        clean_rows, token_counts = filter_for_context(clean_rows, args.kind, tokenizer,
                                                     max_length=context, max_prompt_tokens=prompt_limit)
        token_filter = {"model_id": args.tokenizer_id, "revision": args.tokenizer_revision,
                        "max_length": context, "max_prompt_tokens": prompt_limit,
                        "policy": "preserve_full_prompt_filter", "counts": token_counts}

    if args.kind == "eval":
        selected = select_eval(clean_rows, count=sizes["eval"], seed=args.seed)
        splits = {"eval": selected}
    else:
        train_rows, validation_rows = split_rows(
            clean_rows,
            train_size=sizes["train"],
            validation_size=sizes["validation"],
            seed=args.seed,
        )
        splits = {"train": train_rows, "validation": validation_rows}

    metadata = {
        "source": {
            "id": args.source_id,
            "revision": args.revision,
            "split": args.source_split,
        },
        "input": input_metadata,
        "config": {
            "kind": args.kind,
            "schema": args.schema,
            "seed": args.seed,
            "sizes": sizes,
            "filter_counts": counts,
            "prompt_template": format_prompt("{instruction}"),
        },
    }
    if exclusions is not None:
        metadata["config"]["exclusions"] = exclusions
    if token_filter is not None:
        metadata["config"]["token_filter"] = token_filter

    write_prepared(args.output_dir, splits, metadata)

    filtered = counts["input"] - counts["kept"]
    selected_total = sum(sizes.values())
    print(f"input: {counts['input']}")
    print(f"kept: {counts['kept']}")
    print(f"filtered: {filtered}")
    if exclusions is not None:
        print(f"evaluation overlaps removed: {exclusions['removed']}")
        print(f"eligible after exclusions: {exclusions['eligible']}")
    if token_filter is not None:
        print(f"token eligibility: {token_filter['counts']}")
    print(f"selected: {selected_total}")
    print(f"output: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    sys.exit(main())
