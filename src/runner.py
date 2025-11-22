"""Reusable runner utilities for the synthetic table flow."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable

from dotenv import load_dotenv

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
        help="Optional path to save the parsed result as JSON (HTML saved separately)",
    )
    return parser


def run_flow_for_image(
    image: Path, *, model: str = "gpt-4.1-mini", temperature: float = 0.2
) -> TableState:
    """Execute the synthetic table flow for a given image path."""

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        msg = "OPENAI_API_KEY is not set. Add it to a .env file or your environment."
        raise RuntimeError(msg)

    return run_synthetic_table_flow(str(image), model=model, temperature=temperature)


def _write_html(path: Path, content: str | None) -> Path | None:
    """Persist HTML content to disk if available."""

    if content is None:
        return None

    path.write_text(content, encoding="utf-8")
    return path


def _filter_json_safe_state(state: TableState, *, html_paths: Iterable[tuple[str, Path | None]]) -> Dict:
    """Remove large HTML payloads from JSON output and include file references."""

    payload: Dict[str, object] = {
        "image_path": state.get("image_path"),
        "table_summary": state.get("table_summary"),
        "reflection": state.get("reflection"),
        "errors": state.get("errors"),
    }

    for label, path in html_paths:
        if path:
            payload[label] = str(path)

    return payload


def run_with_args(args: argparse.Namespace) -> TableState:
    """Run the flow using parsed CLI arguments and handle optional persistence."""

    result = run_flow_for_image(
        args.image, model=args.model, temperature=args.temperature
    )

    html_refs: list[tuple[str, Path | None]] = []
    if args.save_json:
        base = args.save_json.with_suffix("")
        parsed_html_path = base.with_name(base.name + "_parsed.html")
        synthetic_html_path = base.with_name(base.name + "_synthetic.html")

        html_refs.append(("html_table_path", _write_html(parsed_html_path, result.get("html_table"))))
        html_refs.append(("synthetic_table_path", _write_html(synthetic_html_path, result.get("synthetic_table"))))

        payload = _filter_json_safe_state(result, html_paths=html_refs)
        args.save_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✅ Saved parsed JSON to {args.save_json}")

        for label, path in html_refs:
            if path:
                print(f"📄 Saved {label} to {path}")
    else:
        payload = _filter_json_safe_state(result, html_paths=[])
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    return result


__all__ = ["build_arg_parser", "run_flow_for_image", "run_with_args"]
