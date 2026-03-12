"""RAG pipeline package - public API re-exports."""

from .config import RagSettings, build_llm, build_embeddings, SUPPORTED_SUMMARY
from .ingestion import IngestionReport, iter_documents, get_doc_priority_and_type
from .chunking import chunk_documents
from .vectorstore import (
    build_client,
    build_vector_store,
    collection_exists,
    index_documents,
)
from .retrieval import ask_question, invalidate_answer_cache
from .validation import validate_architecture_conflicts, run_retrieval_smoke_test
from .autotune import run_autotune
from .cli import main

__all__ = [
    "RagSettings",
    "build_llm",
    "build_embeddings",
    "SUPPORTED_SUMMARY",
    "IngestionReport",
    "iter_documents",
    "get_doc_priority_and_type",
    "chunk_documents",
    "build_client",
    "build_vector_store",
    "collection_exists",
    "index_documents",
    "ask_question",
    "invalidate_answer_cache",
    "validate_architecture_conflicts",
    "run_retrieval_smoke_test",
    "run_autotune",
    "main",
]
