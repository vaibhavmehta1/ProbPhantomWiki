from __future__ import annotations

import argparse
import json

from .dataset import DatasetConfig, generate_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prob-phantom-wiki",
        description="Generate a probabilistic fictional wiki and CFG-derived questions.",
    )
    parser.add_argument("output_dir")
    parser.add_argument("--size", type=int, default=10, help="number of fictional people")
    parser.add_argument("--questions", type=int, default=20, help="number of QA items")
    parser.add_argument("--max-depth", type=int, default=2, help="maximum nested relation depth")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--average-friends", type=float, default=2.0)
    parser.add_argument(
        "--question-types",
        default="prior,posterior,conditional,joint,disjunction,comparison",
        help="comma-separated subset of prior,posterior,conditional,joint,disjunction,comparison",
    )
    parser.add_argument("--report-sensitivity", type=float, default=0.90)
    parser.add_argument("--report-false-positive-rate", type=float, default=0.05)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = generate_dataset(
        args.output_dir,
        DatasetConfig(
            size=args.size,
            questions=args.questions,
            max_depth=args.max_depth,
            seed=args.seed,
            average_friends=args.average_friends,
            question_types=tuple(item.strip() for item in args.question_types.split(",") if item.strip()),
            report_sensitivity=args.report_sensitivity,
            report_false_positive_rate=args.report_false_positive_rate,
        ),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

