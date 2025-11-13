"""Agentic flow for generating synthetic tables from images."""

from .flow import build_synthetic_table_graph, run_synthetic_table_flow

__all__ = [
    "build_synthetic_table_graph",
    "run_synthetic_table_flow",
]
