"""Judge saved pairs in a supported Kaggle benchmark environment."""
import argparse
from pathlib import Path

from src.judging import run_judging


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs_path', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--judge_model', required=True)
    parser.add_argument('--bank_kind', choices=['hh', 'alpaca_eval'], required=True)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--sleep_seconds', type=float, default=5)
    parser.add_argument('--max_attempts', type=int, default=3)
    parser.add_argument('--reuse_judgments', action='store_true')
    args = parser.parse_args(argv)
    import kaggle_benchmarks as kbench
    judge = kbench.llms[args.judge_model]
    @kbench.task(name='pa2_saved_pair_judging')
    def judge_saved(llm):
        def request(prompt):
            with kbench.chats.new('pa2_anonymous_pairwise_attempt'):
                return llm.prompt(prompt)
        run_judging(args.pairs_path, args.output_dir, request, args.judge_model,
                    batch_size=args.batch_size, sleep_seconds=args.sleep_seconds,
                    max_attempts=args.max_attempts, reuse=args.reuse_judgments,
                    bank_kind=args.bank_kind)
    judge_saved.run(llm=judge)


if __name__ == '__main__':
    main()
