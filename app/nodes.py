"""
LangGraph node functions for the Customer Support Agent workflow.

This mirrors the final graph built up incrementally in the notebook's
"Customer Support Agent" section:

    initialize_agent -> classify_intent -> rewrite_query
        -> (routed per-intent placeholder node) -> retrieve_documents
        -> generate_response -> validate_response -> recommendation_agent
"""

from __future__ import annotations

import logging

from app import config
from app.chains import (
    format_documents,
    get_intent_chain,
    get_recommendation_chain,
    get_response_chain,
    get_rewrite_chain,
    get_validation_chain,
)
from app.state import CustomerSupportState
from app.vector_store import get_vector_store

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Routing constants
# --------------------------------------------------------------------------

ROUTE_PRODUCT = "product"
ROUTE_WARRANTY = "warranty"
ROUTE_RETURNS = "returns"
ROUTE_SHIPPING = "shipping"
ROUTE_CARE = "care"
ROUTE_RECOMMENDATION = "recommendation"
ROUTE_GENERAL = "general"
ROUTE_UNKNOWN = "unknown"

VALID_INTENTS = {
    "PRODUCT",
    "WARRANTY",
    "RETURNS",
    "SHIPPING",
    "CARE",
    "RECOMMENDATION",
    "GENERAL",
    "UNKNOWN",
}

ROUTING_TABLE = {
    "PRODUCT": ROUTE_PRODUCT,
    "WARRANTY": ROUTE_WARRANTY,
    "RETURNS": ROUTE_RETURNS,
    "SHIPPING": ROUTE_SHIPPING,
    "CARE": ROUTE_CARE,
    "RECOMMENDATION": ROUTE_RECOMMENDATION,
    "GENERAL": ROUTE_GENERAL,
}

# Intents for which suggesting alternative products makes sense. Policy
# questions (warranty/shipping/returns/care/general) never get product
# recommendations bolted on, even if unanswered.
PRODUCT_RELATED_INTENTS = {"PRODUCT", "RECOMMENDATION"}


def is_catalogue_query(query: str) -> bool:
    """Detect questions that ask for the whole product catalogue.

    Checks a list of exact phrases first, then falls back to a
    trigger-word + subject-word heuristic so rephrasings like "Display all
    the product names present in the product catalog" are also caught
    (previously only literal phrases such as "list all products" matched,
    which silently fell back to a narrow top-2 search for anything else).
    """
    query_lower = query.lower()

    if any(keyword in query_lower for keyword in config.CATALOGUE_KEYWORDS):
        return True

    has_trigger = any(word in query_lower for word in config.CATALOGUE_TRIGGER_WORDS)
    has_subject = any(word in query_lower for word in config.CATALOGUE_SUBJECT_WORDS)

    return has_trigger and has_subject


# --------------------------------------------------------------------------
# Core nodes
# --------------------------------------------------------------------------


def initialize_agent(state: CustomerSupportState) -> CustomerSupportState:
    """Reset per-turn fields while preserving conversation history."""
    logger.info(
        "Initialize Agent | history=%d", len(state.get("conversation_history", []))
    )

    question = state["question"].strip()

    return {
        **state,
        "question": question,
        "documents": [],
        "context": "",
        "answer": "",
        "source_documents": [],
        "conversation_history": state.get("conversation_history", []),
        "status": "initialized",
        "error": None,
    }


def classify_intent(state: CustomerSupportState) -> CustomerSupportState:
    """Classify the customer's question into one of the valid intent buckets."""
    logger.info(
        "Intent Agent | history=%d", len(state.get("conversation_history", []))
    )

    try:
        intent = (
            get_intent_chain()
            .invoke({"question": state["question"]})
            .strip()
            .upper()
        )

        if intent not in VALID_INTENTS:
            intent = "UNKNOWN"

        logger.info("Intent classified as %s", intent)

        return {**state, "intent": intent, "status": "intent_classified"}

    except Exception as exc:  # noqa: BLE001 - surfaced to the caller via state
        logger.exception("Intent classification failed")
        return {
            **state,
            "intent": "UNKNOWN",
            "status": "intent_failed",
            "error": str(exc),
        }


def rewrite_query(state: CustomerSupportState) -> CustomerSupportState:
    """Rewrite ambiguous, pronoun-laden follow-up questions using history."""
    logger.info(
        "Query Rewriter | history=%d", len(state.get("conversation_history", []))
    )

    history = state.get("conversation_history", [])
    history_text = "\n".join(
        f"{msg['role'].capitalize()}: {msg['content']}" for msg in history
    )

    try:
        rewritten = (
            get_rewrite_chain()
            .invoke({"question": state["question"], "history": history_text})
            .strip()
        )

        logger.info("Original: %s | Rewritten: %s", state["question"], rewritten)

        return {**state, "rewritten_question": rewritten, "status": "query_rewritten"}

    except Exception as exc:  # noqa: BLE001
        logger.exception("Query rewrite failed")
        return {
            **state,
            "rewritten_question": state["question"],
            "status": "rewrite_failed",
            "error": str(exc),
        }


def route_intent(state: CustomerSupportState) -> str:
    """Map the classified intent to a graph branch name."""
    intent = state.get("intent", "UNKNOWN")
    route = ROUTING_TABLE.get(intent, ROUTE_UNKNOWN)
    logger.info("Routing -> %s", route)
    return route


# --------------------------------------------------------------------------
# Per-intent placeholder nodes.
#
# These are pass-through extension points (kept from the original design)
# so future per-intent pre-processing can be added without reshaping the
# graph or touching the conditional routing logic.
# --------------------------------------------------------------------------


def product_workflow(state: CustomerSupportState) -> CustomerSupportState:
    return state


def warranty_workflow(state: CustomerSupportState) -> CustomerSupportState:
    return state


def shipping_workflow(state: CustomerSupportState) -> CustomerSupportState:
    return state


def returns_workflow(state: CustomerSupportState) -> CustomerSupportState:
    return state


def care_workflow(state: CustomerSupportState) -> CustomerSupportState:
    return state


def recommendation_workflow(state: CustomerSupportState) -> CustomerSupportState:
    return state


def general_workflow(state: CustomerSupportState) -> CustomerSupportState:
    return state


def unknown_workflow(state: CustomerSupportState) -> CustomerSupportState:
    return state


def retrieve_documents(state: CustomerSupportState) -> CustomerSupportState:
    """Run an intent-aware similarity search against the vector store."""
    logger.info(
        "Retrieval Agent | history=%d", len(state.get("conversation_history", []))
    )

    vector_store = get_vector_store()
    query = state.get("rewritten_question") or state["question"]
    intent = state["intent"]

    if intent == "PRODUCT":
        k = config.PRODUCT_TOP_K
    elif intent in ("RECOMMENDATION", "GENERAL", "UNKNOWN"):
        k = config.RECOMMENDATION_TOP_K
    else:
        k = config.DEFAULT_TOP_K

    if intent == "PRODUCT":
        if is_catalogue_query(query):
            logger.info("Catalogue query detected.")
            documents = vector_store.similarity_search(
                query="Product Catalogue",
                k=config.CATALOGUE_TOP_K,
                filter={"document_name": config.DOCUMENT_NAME_FILTERS["PRODUCT"]},
            )
        else:
            documents = vector_store.similarity_search(
                query=query,
                k=k,
                filter={"document_name": config.DOCUMENT_NAME_FILTERS["PRODUCT"]},
            )
    elif intent == "WARRANTY":
        documents = vector_store.similarity_search(
            query=query,
            k=k,
            filter={"document_name": config.DOCUMENT_NAME_FILTERS["WARRANTY"]},
        )
    elif intent == "SHIPPING":
        documents = vector_store.similarity_search(
            query=query,
            k=k,
            filter={"document_name": config.DOCUMENT_NAME_FILTERS["SHIPPING"]},
        )
    else:
        documents = vector_store.similarity_search(query=query, k=k)

    # Rank retrieved documents by metadata importance (High > Medium > Low).
    documents = sorted(
        documents, key=lambda doc: doc.metadata.get("importance", 0), reverse=True
    )

    top_k_context = (
        config.CATALOGUE_CONTEXT_LIMIT
        if intent == "PRODUCT" and is_catalogue_query(query)
        else config.DEFAULT_CONTEXT_LIMIT
    )
    documents = documents[:top_k_context]

    logger.info("Retrieved %d document(s).", len(documents))

    context = format_documents(documents)

    return {**state, "documents": documents, "context": context, "status": "retrieved"}


def generate_response(state: CustomerSupportState) -> CustomerSupportState:
    """Generate the answer from retrieved context and update chat history."""
    logger.info(
        "Response Agent | history=%d", len(state.get("conversation_history", []))
    )

    conversation_history = state.get("conversation_history", []).copy()
    intent = state.get("intent", "UNKNOWN")

    try:
        if intent == "RECOMMENDATION" and is_recommendation_request(state["question"]):
            # A pure "recommend me something" style request isn't a factual
            # question the RESPONSE_PROMPT can answer, so skip straight to
            # the Recommendation Agent instead of forcing a Q&A answer that
            # validation would otherwise flag as unsupported.
            answer = ""
            status = "recommendation_only"
        elif not state["context"]:
            answer = (
                "I'm sorry, but I couldn't find any relevant information "
                "in the knowledge base."
            )
            status = "no_context"
        else:
            answer = get_response_chain().invoke(
                {"question": state["question"], "context": state["context"]}
            )
            status = "answered"

        conversation_history.append({"role": "user", "content": state["question"]})
        conversation_history.append({"role": "assistant", "content": answer})

        return {
            **state,
            "answer": answer,
            "conversation_history": conversation_history,
            "status": status,
        }

    except Exception as exc:  # noqa: BLE001
        logger.exception("Response generation failed")
        return {**state, "status": "failed", "error": str(exc)}


def validate_response(state: CustomerSupportState) -> CustomerSupportState:
    """Ground the generated answer back in the retrieved context."""
    logger.info(
        "Validation Agent | history=%d", len(state.get("conversation_history", []))
    )

    if not state["answer"]:
        # Nothing to validate for pure recommendation requests (see
        # generate_response) - pass the empty answer through untouched so
        # the Recommendation Agent isn't handed a fabricated fallback answer.
        return {**state, "validated_answer": "", "validation_status": "skipped"}

    validated_answer = get_validation_chain().invoke(
        {
            "question": state["question"],
            "context": state["context"],
            "answer": state["answer"],
        }
    )

    return {**state, "validated_answer": validated_answer, "validation_status": "validated"}


def answer_signals_unavailability(answer: str) -> bool:
    """Check whether the validated answer indicates the product is unavailable."""
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in config.UNAVAILABLE_ANSWER_PHRASES)


def is_recommendation_request(question: str) -> bool:
    """Check whether the customer explicitly asked for suggestions/alternatives."""
    question_lower = question.lower()
    return any(
        phrase in question_lower for phrase in config.RECOMMENDATION_REQUEST_PHRASES
    )


def recommendation_agent(state: CustomerSupportState) -> CustomerSupportState:
    """Suggest catalogue alternatives, but only when actually relevant.

    Recommendations are shown ONLY in these three cases:
    1. The requested product does not exist in the catalogue at all.
    2. The requested product exists but is out of stock/unavailable.
    3. The customer explicitly asked for recommendations/suggestions.

    Catalogue-wide listing queries (e.g. "List all the products") are
    excluded outright, even if one of the listed products happens to be out
    of stock - a request for the whole catalogue is never a request for
    alternatives to a single unavailable item.
    """
    logger.info(
        "Recommendation Agent | history=%d", len(state.get("conversation_history", []))
    )

    validated_answer = state["validated_answer"]
    intent = state.get("intent", "UNKNOWN")
    has_context = bool(state.get("context"))
    query = state.get("rewritten_question") or state["question"]

    if is_catalogue_query(query):
        should_recommend = False
    else:
        should_recommend = (
            intent in PRODUCT_RELATED_INTENTS
            and has_context
            and (
                answer_signals_unavailability(validated_answer)
                or is_recommendation_request(state["question"])
            )
        )

    if not should_recommend:
        logger.info(
            "Skipping recommendations (intent=%s, has_context=%s).", intent, has_context
        )
        return {
            **state,
            "recommendations": "No Recommendation",
            "final_response": validated_answer,
            "status": "recommendation_skipped",
        }

    # Widen the retrieval context for the recommendation chain. The
    # PRODUCT-intent retrieval that fed generate_response only pulls
    # k=PRODUCT_TOP_K (2) nearest documents, which is often too narrow to
    # offer viable alternatives for a product that turned out to be
    # unavailable. Re-query the catalogue more broadly here so the
    # Recommendation Agent has enough choice to work with.
    recommendation_context = state["context"]
    if intent == "PRODUCT":
        broader_documents = get_vector_store().similarity_search(
            query=query,
            k=config.RECOMMENDATION_TOP_K,
            filter={"document_name": config.DOCUMENT_NAME_FILTERS["PRODUCT"]},
        )
        recommendation_context = format_documents(broader_documents) or recommendation_context

    recommendations = (
        get_recommendation_chain()
        .invoke(
            {
                "question": state["question"],
                "context": recommendation_context,
                "answer": validated_answer,
            }
        )
        .strip()
    )

    normalized = recommendations.lower().rstrip(".! ")

    if normalized == "no recommendation":
        final_response = validated_answer or (
            "I'm sorry, I don't have enough information in our catalogue "
            "to recommend a product right now."
        )
    elif validated_answer:
        final_response = (
            f"{validated_answer}\n\nRecommended Products:\n{recommendations}"
        )
    else:
        # Pure recommendation request - no Q&A answer to lead with, so the
        # reply is only the recommended products, nothing else.
        final_response = f"Recommended Products:\n{recommendations}"

    return {
        **state,
        "recommendations": recommendations,
        "final_response": final_response,
        "status": "recommendation_generated",
    }
