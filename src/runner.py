"""Reusable runner utilities for the synthetic table flow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .flow import TableState, run_synthetic_table_flow


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the common argument parser used by CLI entrypoints."""

    parser = argparse.ArgumentParser(
        description="Generate a synthetic HTML table from a Korean table image using LangGraph.",
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
    return parser


def run_flow_for_image(
    image: Path, *, model: str = "gpt-4.1-mini", temperature: float = 0.2
) -> TableState:
    """Execute the synthetic table flow for a given image path."""

    return run_synthetic_table_flow(str(image), model=model, temperature=temperature)


def run_with_args(args: argparse.Namespace) -> TableState:
    """Run the flow using parsed CLI arguments and handle optional persistence."""

    result = run_flow_for_image(
        args.image, model=args.model, temperature=args.temperature
    )

    if args.save_json:
        args.save_json.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"✅ Saved flow result to {args.save_json}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))

    return result


__all__ = ["build_arg_parser", "run_flow_for_image", "run_with_args"]
