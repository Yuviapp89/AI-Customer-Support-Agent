"""
Pinecone connection, embeddings and retriever helpers.

Assumes the Pinecone index has already been created and populated by an
offline ingestion pipeline (PDF loading -> semantic chunking -> metadata
enrichment -> embedding -> upsert), matching the notebook's cells before
"Build and Test the Semantic Retriever". That ingestion logic is
intentionally NOT reproduced in this app; this module only *reads* from an
existing index.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from app import config


@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    config.require_api_keys()
    return OpenAIEmbeddings(model=config.EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_pinecone_client() -> Pinecone:
    config.require_api_keys()
    return Pinecone(api_key=config.PINECONE_API_KEY)


@lru_cache(maxsize=1)
def get_index():
    return get_pinecone_client().Index(config.INDEX_NAME)


@lru_cache(maxsize=1)
def get_vector_store() -> PineconeVectorStore:
    return PineconeVectorStore(index=get_index(), embedding=get_embeddings())


def get_retriever(k: int = 3):
    """Return a plain similarity-search retriever over the knowledge base."""
    return get_vector_store().as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def index_stats() -> dict:
    """Return Pinecone index stats - handy for a Streamlit health-check panel."""
    return get_index().describe_index_stats()
