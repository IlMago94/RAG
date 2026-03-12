"""Qdrant vector store helpers and document indexing orchestration."""
from __future__ import annotations

import shutil
from datetime import datetime, timezone

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from .config import RagSettings, SUPPORTED_SUMMARY, build_embeddings, write_report
from .ingestion import IngestionReport, iter_documents
from .chunking import chunk_documents


# ---------------------------------------------------------------------------
# Client / store helpers
# ---------------------------------------------------------------------------

def build_client(settings: RagSettings) -> QdrantClient:
    """Create a Qdrant client for local vector storage."""
    settings.persist_dir.parent.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(settings.persist_dir))


def collection_exists(settings: RagSettings) -> bool:
    """Check if the vector collection exists in Qdrant storage."""
    client = build_client(settings)
    return client.collection_exists(settings.collection_name)


def build_vector_store(settings: RagSettings) -> QdrantVectorStore:
    """Load an existing Qdrant vector store for retrieval."""
    return QdrantVectorStore.from_existing_collection(
        collection_name=settings.collection_name,
        embedding=build_embeddings(settings),
        path=str(settings.persist_dir),
    )


def build_vector_store_for_collection(settings: RagSettings, collection_name: str) -> QdrantVectorStore:
    return QdrantVectorStore.from_existing_collection(
        collection_name=collection_name,
        embedding=build_embeddings(settings),
        path=str(settings.persist_dir),
    )


def _delete_collection_if_exists(client: QdrantClient, collection_name: str) -> None:
    """Delete a Qdrant collection only if it exists."""
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)


def cleanup_collections(settings: RagSettings, collection_names: list[str]) -> None:
    """Use a short-lived client so local Qdrant file lock is released before indexing."""
    client = build_client(settings)
    try:
        for collection_name in collection_names:
            _delete_collection_if_exists(client, collection_name)
    finally:
        close_method = getattr(client, "close", None)
        if callable(close_method):
            close_method()


# ---------------------------------------------------------------------------
# Indexing orchestration
# ---------------------------------------------------------------------------

def index_documents(settings: RagSettings, rebuild: bool) -> IngestionReport:
    """Index documents with autofix parsing and hybrid storage strategy."""
    report = IngestionReport()

    if rebuild and settings.persist_dir.exists():
        shutil.rmtree(settings.persist_dir)

    cleanup_collections(settings, [settings.collection_name])

    documents = list(
        iter_documents(
            settings.data_dir,
            report=report,
            quarantine_dirname=settings.quarantine_dirname,
        )
    )
    if not documents:
        raise RuntimeError(
            f"No parseable documents found in {settings.data_dir}. "
            "Add files in priority 1/2 formats or supported fallback formats."
        )

    report.total_docs = len(documents)
    chunks = chunk_documents(documents, settings)
    report.total_chunks = len(chunks)

    if not chunks:
        report.anomalies.append("All parsed documents produced zero chunks")
        raise RuntimeError("Parsed documents are empty after chunking.")

    embeddings = build_embeddings(settings)

    QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        path=str(settings.persist_dir),
        collection_name=settings.collection_name,
    )

    # Hybrid mode: per-type collections
    type_groups: dict[str, list[Document]] = {}
    for chunk in chunks:
        type_groups.setdefault(chunk.metadata.get("doc_type", "generic"), []).append(chunk)

    for doc_type, type_chunks in sorted(type_groups.items()):
        type_collection = f"{settings.collection_name}-{doc_type}"
        cleanup_collections(settings, [type_collection])
        QdrantVectorStore.from_documents(
            documents=type_chunks,
            embedding=embeddings,
            path=str(settings.persist_dir),
            collection_name=type_collection,
        )

    if report.failed_files > 0:
        report.anomalies.append(
            f"Parser failures detected: {report.failed_files} file(s) moved to quarantine"
        )

    failure_rate = report.failed_files / max(report.failed_files + report.indexed_files, 1)
    if failure_rate > 0.35:
        report.anomalies.append(f"High parse failure rate: {failure_rate:.2%}")

    report_payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "collection": settings.collection_name,
        "supported_formats": SUPPORTED_SUMMARY,
        "ingestion": report.to_dict(),
    }
    write_report(settings.reports_dir / "latest_index_report.json", report_payload)

    print(
        f"Indexed {len(documents)} documents into {len(chunks)} chunks at {settings.persist_dir}. "
        f"Failed files: {report.failed_files}, quarantined: {report.quarantined_files}."
    )
    print(f"Report: {settings.reports_dir / 'latest_index_report.json'}")
    return report
