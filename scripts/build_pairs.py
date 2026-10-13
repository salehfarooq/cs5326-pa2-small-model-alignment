"""Construct saved, blinded SFT versus DPO comparisons."""
import argparse
from pathlib import Path

from src.pairwise import build_pairs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sft-dir', type=Path, required=True)
    parser.add_argument('--dpo-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    pairs, _ = build_pairs(args.sft_dir, args.dpo_dir, args.output_dir, args.seed)
    print(f'Saved {len(pairs)} blinded pairs to {args.output_dir}')


if __name__ == '__main__':
    main()
