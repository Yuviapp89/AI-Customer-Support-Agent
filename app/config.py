"""
Centralized configuration for the AI Customer Support Agent.

Loads environment variables (via python-dotenv) and exposes the constants
used across the retrieval, generation and Streamlit layers. Nothing in this
module talks to OpenAI/Pinecone directly - it only prepares configuration.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.environ.get("LANGSMITH_PROJECT", "customer-support-agent")
LANGSMITH_ENDPOINT = os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
_LANGSMITH_TRACING_RAW = os.environ.get("LANGSMITH_TRACING")
# Default to "on" whenever a key is present, but always respect an explicit override.
LANGSMITH_TRACING = (
    _LANGSMITH_TRACING_RAW.lower() == "true"
    if _LANGSMITH_TRACING_RAW is not None
    else bool(LANGSMITH_API_KEY)
)


def require_api_keys() -> None:
    """Raise a clear error if mandatory API keys are missing.

    Called lazily (not at import time) so the package can be imported for
    tooling/tests without valid credentials being present.
    """
    missing = [
        name
        for name, value in (
            ("OPENAI_API_KEY", OPENAI_API_KEY),
            ("PINECONE_API_KEY", PINECONE_API_KEY),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Missing required environment variable(s): "
            f"{', '.join(missing)}. Add them to a .env file "
            "(see .env.example) or your environment before starting the app."
        )


# --------------------------------------------------------------------------
# Pinecone / Embeddings
# --------------------------------------------------------------------------

INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "customer-support-agent")
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
METRIC = "cosine"
CLOUD_PROVIDER = "aws"
REGION = "us-east-1"

# --------------------------------------------------------------------------
# LLM
# --------------------------------------------------------------------------

LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0
LLM_MAX_TOKENS = 700

# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------

DEFAULT_TOP_K = 5
PRODUCT_TOP_K = 2
RECOMMENDATION_TOP_K = 6
CATALOGUE_TOP_K = 100
CATALOGUE_CONTEXT_LIMIT = 20
DEFAULT_CONTEXT_LIMIT = 5

DOCUMENT_NAME_FILTERS = {
    "PRODUCT": "Product_Catalog",
    "WARRANTY": "Warranty_Policy",
    "SHIPPING": "Shipping_Policy",
}

CATALOGUE_KEYWORDS = [
    "list all products",
    "show all products",
    "show catalogue",
    "show catalog",
    "what products",
    "available products",
    "display products",
    "list products",
]

# Fallback heuristic for catalogue-wide questions that don't match one of the
# exact phrases above (e.g. "Display all the product names present in the
# product catalog"). A query is treated as catalogue-wide when it combines
# a "give me everything" trigger word with a "products/catalog" subject word.
CATALOGUE_TRIGGER_WORDS = [
    "all",
    "every",
    "list",
    "display",
    "show",
    "entire",
    "complete",
    "full",
    "how many",
]

CATALOGUE_SUBJECT_WORDS = [
    "product",
    "products",
    "catalog",
    "catalogue",
]

# --------------------------------------------------------------------------
# Recommendation gating
#
# Recommendations should only ever be shown when a product turned out to be
# unavailable/out of stock, or when the customer explicitly asked for
# suggestions/alternatives. These phrase lists back a deterministic code-side
# check in `recommendation_agent` so the feature doesn't depend solely on the
# LLM correctly following the recommendation prompt's instructions.
# --------------------------------------------------------------------------

UNAVAILABLE_ANSWER_PHRASES = [
    "we don't sell that product",
    "we dont sell that product",
    "currently not available",
    "not available in our catalog",
    "not available in our catalogue",
    "out of stock",
    "not in stock",
    "does not exist",
    "doesn't exist",
    "no longer available",
    "sold out",
]

RECOMMENDATION_REQUEST_PHRASES = [
    "recommend",
    "suggest",
    "alternative",
    "similar product",
    "similar option",
    "something else",
    "other option",
]

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

LOGS_DIR = BASE_DIR / "logs"


def configure_logging() -> None:
    """Configure root logging to write to logs/app.log and the console."""
    LOGS_DIR.mkdir(exist_ok=True)

    if logging.getLogger().handlers:
        # Already configured (e.g. Streamlit re-running the script).
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(LOGS_DIR / "app.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def configure_langsmith() -> None:
    """Enable LangSmith tracing for every LangChain/LangGraph call, if configured.

    Tracing is opt-in: it only turns on when LANGSMITH_API_KEY is set (or
    LANGSMITH_TRACING=true is set explicitly). Both the current (LANGSMITH_*)
    and legacy (LANGCHAIN_*) environment variable names are set because
    different langchain-core/langsmith versions read one or the other.
    """
    logger = logging.getLogger(__name__)

    if not (LANGSMITH_TRACING and LANGSMITH_API_KEY):
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        logger.info("LangSmith tracing disabled.")
        return

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGSMITH_ENDPOINT"] = LANGSMITH_ENDPOINT
    os.environ["LANGCHAIN_ENDPOINT"] = LANGSMITH_ENDPOINT
    os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT
    os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT

    logger.info("LangSmith tracing enabled for project '%s'.", LANGSMITH_PROJECT)
