"""
Builds and compiles the LangGraph state machine for the Customer Support Agent.

Graph shape (matches the final version reached in the notebook):

    START -> initialize_agent -> classify_intent -> rewrite_query
        --(route_intent)--> {product|warranty|shipping|returns|care|
                              recommendation|general|unknown}
        -> retrieve_documents -> generate_response -> validate_response
        -> recommendation_agent -> END
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langgraph.graph import END, START, StateGraph

from app import nodes
from app.state import CustomerSupportState


@lru_cache(maxsize=1)
def build_customer_support_graph() -> Any:
    """Compile the full customer support workflow graph (cached singleton)."""

    builder = StateGraph(CustomerSupportState)

    builder.add_node("initialize_agent", nodes.initialize_agent)
    builder.add_node("classify_intent", nodes.classify_intent)
    builder.add_node("rewrite_query", nodes.rewrite_query)
    builder.add_node("product", nodes.product_workflow)
    builder.add_node("warranty", nodes.warranty_workflow)
    builder.add_node("shipping", nodes.shipping_workflow)
    builder.add_node("returns", nodes.returns_workflow)
    builder.add_node("care", nodes.care_workflow)
    builder.add_node("recommendation", nodes.recommendation_workflow)
    builder.add_node("general", nodes.general_workflow)
    builder.add_node("unknown", nodes.unknown_workflow)
    builder.add_node("retrieve_documents", nodes.retrieve_documents)
    builder.add_node("generate_response", nodes.generate_response)
    builder.add_node("validate_response", nodes.validate_response)
    builder.add_node("recommendation_agent", nodes.recommendation_agent)

    builder.add_edge(START, "initialize_agent")
    builder.add_edge("initialize_agent", "classify_intent")
    builder.add_edge("classify_intent", "rewrite_query")

    builder.add_conditional_edges(
        "rewrite_query",
        nodes.route_intent,
        {
            nodes.ROUTE_PRODUCT: "product",
            nodes.ROUTE_WARRANTY: "warranty",
            nodes.ROUTE_RETURNS: "returns",
            nodes.ROUTE_SHIPPING: "shipping",
            nodes.ROUTE_CARE: "care",
            nodes.ROUTE_RECOMMENDATION: "recommendation",
            nodes.ROUTE_GENERAL: "general",
            nodes.ROUTE_UNKNOWN: "unknown",
        },
    )

    for branch in (
        "product",
        "warranty",
        "shipping",
        "returns",
        "care",
        "recommendation",
        "general",
        "unknown",
    ):
        builder.add_edge(branch, "retrieve_documents")

    builder.add_edge("retrieve_documents", "generate_response")
    builder.add_edge("generate_response", "validate_response")
    builder.add_edge("validate_response", "recommendation_agent")
    builder.add_edge("recommendation_agent", END)

    return builder.compile()
