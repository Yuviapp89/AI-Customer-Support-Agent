"""
Shared LangGraph state definition for the Customer Support Agent.

`CustomerSupportState` mirrors the TypedDict evolved throughout the
notebook (initial -> +intent -> +rewritten_question/validation ->
+recommendations), consolidated into its final, fullest shape.
"""

from __future__ import annotations

from typing import Dict, List, Optional, TypedDict

from langchain_core.documents import Document


class CustomerSupportState(TypedDict, total=False):
    """State object passed between every LangGraph node."""

    question: str
    rewritten_question: str
    intent: str
    documents: List[Document]
    context: str
    answer: str
    source_documents: List[Document]
    status: str
    validation_status: str
    validated_answer: str
    recommendations: str
    final_response: str
    conversation_history: List[Dict[str, str]]
    error: Optional[str]


def new_initial_state(
    question: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> CustomerSupportState:
    """Build a fresh state dict for a single conversation turn."""
    return {
        "question": question,
        "rewritten_question": "",
        "intent": "",
        "documents": [],
        "context": "",
        "answer": "",
        "source_documents": [],
        "status": "",
        "validation_status": "",
        "validated_answer": "",
        "recommendations": "",
        "final_response": "",
        "conversation_history": conversation_history or [],
        "error": None,
    }
