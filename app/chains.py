"""
LLM and LCEL chain construction for the Customer Support Agent.

Each chain is a thin `prompt | llm | parser` pipeline, cached as a
singleton so Streamlit re-runs don't rebuild them on every interaction.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from app import config
from app.prompts import (
    INTENT_PROMPT,
    RECOMMENDATION_PROMPT,
    RESPONSE_PROMPT,
    REWRITE_PROMPT,
    VALIDATION_PROMPT,
)


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    config.require_api_keys()
    return ChatOpenAI(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
    )


def format_documents(documents: List[Document]) -> str:
    """Concatenate retrieved document chunks into a single context string."""
    return "\n\n".join(doc.page_content for doc in documents)


@lru_cache(maxsize=1)
def get_intent_chain():
    return INTENT_PROMPT | get_llm() | StrOutputParser()


@lru_cache(maxsize=1)
def get_rewrite_chain():
    return REWRITE_PROMPT | get_llm() | StrOutputParser()


@lru_cache(maxsize=1)
def get_response_chain():
    return RESPONSE_PROMPT | get_llm() | StrOutputParser()


@lru_cache(maxsize=1)
def get_validation_chain():
    return VALIDATION_PROMPT | get_llm() | StrOutputParser()


@lru_cache(maxsize=1)
def get_recommendation_chain():
    return RECOMMENDATION_PROMPT | get_llm() | StrOutputParser()
