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
        content.append({"type": "image_url", "image_url": {"url": image_url}})

    response = llm.invoke([HumanMessage(content=content)])
    return response.content if isinstance(response.content, str) else json.dumps(response.content)


def image_to_html_node(llm: ChatOpenAI):
    """Create a node that extracts the table structure as HTML."""

    def _node(state: TableState) -> TableState:
        image_path = Path(state["image_path"])
        if not image_path.exists():
            errors = state.get("errors", []) + [f"Image not found: {image_path}"]
            return {**state, "errors": errors}

        prompt = (
            "당신은 인공지능 비서입니다. 한국어 표 이미지를 입력으로 받아 표의 구조를 정확하게 HTML table 태그로 복원하세요. "
            "텍스트는 가능한 한 그대로 복원하되, 표 구조(행, 열, 병합 정보)를 정확히 표현하세요. "
            "결과는 <table>...</table> 형태의 순수 HTML 문자열만 반환하세요."
        )
        html = _call_llm(llm, prompt, image_url=_encode_image(image_path))
        print(f"Done extracting HTML table from image: {image_path}")
        return {**state, "html_table": html}

    return _node


def parse_contents_node(llm: ChatOpenAI):
    """Create a node that summarizes the table contents."""

    def _node(state: TableState) -> TableState:
        html = state.get("html_table")
        if not html:
            errors = state.get("errors", []) + ["Missing HTML table representation."]
            return {**state, "errors": errors}

        prompt = (
            "다음 HTML 표를 보고 핵심적인 열 이름과 각 행의 요점을 요약하세요. "
            "가능하면 숫자 범위, 단위, 패턴도 정리하세요.\n\n"
            f"HTML:\n{html}"
        )
        summary = _call_llm(llm, prompt)
        print("Done summarizing table contents.")
        return {**state, "table_summary": summary}

    return _node


def generate_synthetic_table_node(llm: ChatOpenAI):
    """Create a node that generates a synthetic dataset with the same structure."""

    def _node(state: TableState) -> TableState:
        html = state.get("html_table")
        summary = state.get("table_summary")
        if not html or not summary:
            errors = state.get("errors", []) + ["Insufficient information to generate synthetic table."]
            return {**state, "errors": errors}

        prompt = (
            "당신은 데이터 생성 전문가입니다. 주어진 HTML 표의 구조를 유지하되, 모든 셀 내용을 라이선스 문제가 없는 합성 데이터로 대체하세요. "
            "실제 인물이나 조직명을 사용하지 마세요. 한국어 자연스러운 텍스트와 현실적인 수치를 사용하세요. "
            "가능하면 서로 일관성 있는 데이터로 채우고, 표 구조는 유지하세요.\n\n"
            "[표 구조]\n"
            f"{html}\n\n"
            "[요약 정보]\n"
            f"{summary}\n\n"
            "최종 결과는 <table>...</table> HTML 문자열 하나만 출력하세요."
        )
        synthetic_html = _call_llm(llm, prompt)
        print("Done generating synthetic table.")
        return {**state, "synthetic_table": synthetic_html}

    return _node


def self_reflection_node(llm: ChatOpenAI):
    """Create a node that performs a final self-check of the synthetic table."""

    def _node(state: TableState) -> TableState:
        synthetic_html = state.get("synthetic_table")
        if not synthetic_html:
            errors = state.get("errors", []) + ["Synthetic table generation failed."]
            return {**state, "errors": errors}

        prompt = (
            "생성된 합성 표를 검토하여 다음 항목을 점검하세요:\n"
            "1. 표 구조가 합리적인가?\n"
            "2. 개인정보나 저작권 문제가 없는가?\n"
            "3. 수치나 날짜가 상식적인 범위인가?\n\n"
            "문제가 있다면 수정 제안을 서술하고, 없다면 '승인'이라고만 답하세요.\n\n"
            f"합성 표:\n{synthetic_html}"
        )
        reflection = _call_llm(llm, prompt)
        print("Done with self-reflection on synthetic table.")
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
