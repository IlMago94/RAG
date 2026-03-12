"""Local RAG pipeline for Ollama and OpenAI-compatible backends.

Documents are loaded from the ``data/`` directory, split into chunks, and
indexed in a local Qdrant vector database. Queries are answered by retrieving
the most relevant chunks and sending them to an LLM.

Main features:
- Automatic ingestion with fallback across multiple parsers and file quarantine
- Hybrid indexing: single main collection + collections per document type
- Architectural conflict detection (e.g., incompatible CSP versions)
- Retrieval consistency smoke test
- Autotune mode for automatic indexing and validation cycles

Utilizzo rapido::

    python test_rag.py index [--rebuild]
    python test_rag.py ask "<question>" [--show-context]
    python test_rag.py validate [--report <path>]
    python test_rag.py autotune [--rebuild] [--watch]
    python test_rag.py config
    python test_rag.py formats

This file is a thin entry point. All logic lives in the ``rag`` package.
"""
# Re-export public API for backward compatibility
from rag import (  # noqa: F401
    RagSettings,
    IngestionReport,
    build_llm,
    build_embeddings,
    iter_documents,
    get_doc_priority_and_type,
    chunk_documents,
    build_client,
    build_vector_store,
    collection_exists,
    index_documents,
    ask_question,
    validate_architecture_conflicts,
    run_retrieval_smoke_test,
    run_autotune,
    main,
    SUPPORTED_SUMMARY,
)

if __name__ == "__main__":
    main()
