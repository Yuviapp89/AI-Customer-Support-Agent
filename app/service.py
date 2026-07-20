"""
High-level façade used by the Streamlit UI (or any other client) to talk
to the Customer Support Agent without depending on LangGraph internals.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, TypedDict

from langchain_core.documents import Document

from app.config import configure_langsmith, configure_logging
from app.graph import build_customer_support_graph
from app.state import new_initial_state

configure_logging()
configure_langsmith()
logger = logging.getLogger(__name__)


class AgentResponse(TypedDict):
    final_response: str
    answer: str
    validated_answer: str
    recommendations: str
    intent: str
    rewritten_question: str
    sources: List[Document]
    conversation_history: List[Dict[str, str]]
    status: str
    error: Optional[str]


def ask(
    question: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> AgentResponse:
    """
    Run a single conversation turn through the Customer Support Agent graph.

    Args:
        question: The customer's current message.
        conversation_history: Prior turns as a list of {"role", "content"}
            dicts. Pass back `result["conversation_history"]` from the
            previous call to keep multi-turn context (pronoun resolution,
            "what's the price of it?", etc.).

    Returns:
        An AgentResponse with the final answer, retrieved sources and the
        updated conversation history to persist for the next turn.
    """
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string")

    logger.info("New question received: %s", question)

    graph = build_customer_support_graph()
    state = new_initial_state(question, conversation_history)

    result = graph.invoke(state)

    return {
        "final_response": (
            result.get("final_response")
            or result.get("validated_answer")
            or result.get("answer", "")
        ),
        "answer": result.get("answer", ""),
        "validated_answer": result.get("validated_answer", ""),
        "recommendations": result.get("recommendations", ""),
        "intent": result.get("intent", ""),
        "rewritten_question": result.get("rewritten_question", ""),
        "sources": result.get("documents", []),
        "conversation_history": result.get("conversation_history", []),
        "status": result.get("status", ""),
        "error": result.get("error"),
    }
