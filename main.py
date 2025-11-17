"""Lightweight entrypoint for running the synthetic table flow."""

from __future__ import annotations

from generate_synthetic_table.runner import build_arg_parser, run_with_args


def main() -> None:
    args = build_arg_parser().parse_args()
    print(f"Running with args: {args}")
    run_with_args(args)


if __name__ == "__main__":
    main()
