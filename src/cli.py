"""Command line interface for running the synthetic table LangGraph flow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .flow import run_synthetic_table_flow


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic HTML table from a Korean table image using LangGraph."
    )
    parser.add_argument("image", type=Path, help="Path to the input table image")
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="OpenAI model name to use (default: gpt-4.1-mini)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for the model (default: 0.2)",
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        help="Optional path to save the resulting state as JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_synthetic_table_flow(
        str(args.image), model=args.model, temperature=args.temperature
    )

    if args.save_json:
        args.save_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✅ Saved flow result to {args.save_json}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
