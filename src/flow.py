"""LangGraph flow for generating synthetic tables from Korean table images."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Dict, List, TypedDict

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from dotenv import load_dotenv


class TableState(TypedDict, total=False):
    """Shared state passed between graph nodes."""

    image_path: str
    html_table: str
    table_summary: str
    synthetic_table: str
    reflection: str
    errors: List[str]


def _encode_image(image_path: Path) -> str:
    """Return the image encoded as a data URL."""

    mime = "image/png"
    suffix = image_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    elif suffix == ".webp":
        mime = "image/webp"
    elif suffix == ".gif":
        mime = "image/gif"

    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def _call_llm(llm: ChatOpenAI, prompt: str, *, image_url: str | None = None) -> str:
    """Call the multi-modal LLM with an optional image."""

    content: List[Dict[str, str]] = [{"type": "text", "text": prompt}]
    if image_url:
        content.append({"type": "image_url", "image_url": image_url})

    response = llm.invoke([HumanMessage(content=content)])
    return response.content if isinstance(response.content, str) else json.dumps(response.content)


def _load_prompt(name: str) -> str:
    """Load a prompt text from the prompts directory."""

    prompt_path = Path(__file__).parent / "prompts" / f"{name}.txt"
    return prompt_path.read_text(encoding="utf-8")


def image_to_html_node(llm: ChatOpenAI):
    """Create a node that extracts the table structure as HTML."""

    prompt = _load_prompt("image_to_html")

    def _node(state: TableState) -> TableState:
        image_path = Path(state["image_path"])
        if not image_path.exists():
            errors = state.get("errors", []) + [f"Image not found: {image_path}"]
            return {**state, "errors": errors}

        html = _call_llm(llm, prompt, image_url=_encode_image(image_path))
        return {**state, "html_table": html}

    return _node


def parse_contents_node(llm: ChatOpenAI):
    """Create a node that summarizes the table contents."""

    prompt_template = _load_prompt("parse_contents")

    def _node(state: TableState) -> TableState:
        html = state.get("html_table")
        if not html:
            errors = state.get("errors", []) + ["Missing HTML table representation."]
            return {**state, "errors": errors}

        prompt = prompt_template.format(html=html)
        summary = _call_llm(llm, prompt)
        return {**state, "table_summary": summary}

    return _node


def generate_synthetic_table_node(llm: ChatOpenAI):
    """Create a node that generates a synthetic dataset with the same structure."""

    prompt_template = _load_prompt("generate_synthetic_table")

    def _node(state: TableState) -> TableState:
        html = state.get("html_table")
        summary = state.get("table_summary")
        if not html or not summary:
            errors = state.get("errors", []) + ["Insufficient information to generate synthetic table."]
            return {**state, "errors": errors}

        prompt = prompt_template.format(html=html, summary=summary)
        synthetic_html = _call_llm(llm, prompt)
        return {**state, "synthetic_table": synthetic_html}

    return _node


def self_reflection_node(llm: ChatOpenAI):
    """Create a node that performs a final self-check of the synthetic table."""

    prompt_template = _load_prompt("self_reflection")

    def _node(state: TableState) -> TableState:
        synthetic_html = state.get("synthetic_table")
        if not synthetic_html:
            errors = state.get("errors", []) + ["Synthetic table generation failed."]
            return {**state, "errors": errors}

        prompt = prompt_template.format(synthetic_html=synthetic_html)
        reflection = _call_llm(llm, prompt)
        return {**state, "reflection": reflection}

    return _node


def build_synthetic_table_graph(llm: ChatOpenAI) -> StateGraph:
    """Assemble the LangGraph pipeline."""

    graph = StateGraph(TableState)
    graph.add_node("image_to_html", image_to_html_node(llm))
    graph.add_node("parse_contents", parse_contents_node(llm))
    graph.add_node("generate_synthetic_table", generate_synthetic_table_node(llm))
    graph.add_node("self_reflection", self_reflection_node(llm))

    graph.add_edge(START, "image_to_html")
    graph.add_edge("image_to_html", "parse_contents")
    graph.add_edge("parse_contents", "generate_synthetic_table")
    graph.add_edge("generate_synthetic_table", "self_reflection")
    graph.add_edge("self_reflection", END)

    return graph


def run_synthetic_table_flow(image_path: str, *, model: str = "gpt-4.1-mini", temperature: float = 0.2) -> TableState:
    """Execute the synthetic table flow for a single image path."""

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        msg = "OPENAI_API_KEY is not set. Add it to a .env file or your environment."
        raise RuntimeError(msg)

    llm = ChatOpenAI(model=model, temperature=temperature)
    app = build_synthetic_table_graph(llm).compile()
    final_state: TableState = app.invoke({"image_path": image_path})
    return final_state


__all__ = [
    "TableState",
    "build_synthetic_table_graph",
    "run_synthetic_table_flow",
]
