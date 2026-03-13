"""Qdrant vector store helpers and document indexing orchestration."""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct

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
    try:
        return client.collection_exists(settings.collection_name)
    finally:
        close_fn = getattr(client, "close", None)
        if callable(close_fn):
            close_fn()


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
# Incremental indexing helpers
# ---------------------------------------------------------------------------

_MANIFEST_FILENAME = "index_manifest.json"


def _file_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file in streaming mode."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _load_manifest(persist_dir: Path) -> dict[str, str]:
    """Load the file-hash manifest from disk (empty dict if missing or corrupt)."""
    manifest_path = persist_dir / _MANIFEST_FILENAME
    if manifest_path.exists():
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_manifest(persist_dir: Path, manifest: dict[str, str]) -> None:
    """Persist the file-hash manifest to disk."""
    persist_dir.mkdir(parents=True, exist_ok=True)
    (persist_dir / _MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _delete_chunks_by_source(client: QdrantClient, collection_name: str, relative_source: str) -> None:
    """Remove all Qdrant points whose metadata.source matches the given relative path."""
    try:
        client.delete(
            collection_name=collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="metadata.source", match=MatchValue(value=relative_source))]
            ),
        )
    except Exception:
        pass


def _copy_points_by_doctype(
    client: QdrantClient,
    src_collection: str,
    dst_collection: str,
    doc_type: str,
) -> int:
    """Copy points matching doc_type from src to dst without re-embedding.

    Reads the already-computed vectors via scroll() and writes them directly
    into the destination collection with upsert(). Zero embedding calls.
    Returns the number of points copied.
    """
    collection_info = client.get_collection(src_collection)
    vectors_config = collection_info.config.params.vectors

    if client.collection_exists(dst_collection):
        client.delete_collection(dst_collection)
    client.create_collection(collection_name=dst_collection, vectors_config=vectors_config)

    offset = None
    total = 0
    while True:
        records, next_offset = client.scroll(
            collection_name=src_collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="metadata.doc_type", match=MatchValue(value=doc_type))]
            ),
            with_vectors=True,
            with_payload=True,
            limit=256,
            offset=offset,
        )
        if not records:
            break
        client.upsert(
            collection_name=dst_collection,
            points=[PointStruct(id=r.id, vector=r.vector, payload=r.payload) for r in records],
        )
        total += len(records)
        if next_offset is None:
            break
        offset = next_offset
    return total


# ---------------------------------------------------------------------------
# Indexing orchestration
# ---------------------------------------------------------------------------

def index_documents(
    settings: RagSettings,
    rebuild: bool,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> IngestionReport:
    """Index documents with autofix parsing and hybrid storage strategy.

    In incremental mode (rebuild=False) files that have not changed since the
    last run are skipped entirely — only new/modified files are re-embedded.
    Deleted files are removed from the vector store automatically.
    """
    report = IngestionReport()

    if rebuild and settings.persist_dir.exists():
        shutil.rmtree(settings.persist_dir)

    # ----- Phase 1: classify files by change status -----
    manifest: dict[str, str] = {} if rebuild else _load_manifest(settings.persist_dir)

    all_files = [
        p for p in sorted(settings.data_dir.rglob("*"))
        if p.is_file() and settings.quarantine_dirname not in p.parts
    ]

    new_manifest: dict[str, str] = {}
    files_to_process: list[Path] = []
    unchanged_count = 0

    for path in all_files:
        rel = str(path.relative_to(settings.data_dir))
        sha = _file_sha256(path)
        new_manifest[rel] = sha
        if not rebuild and manifest.get(rel) == sha:
            unchanged_count += 1
        else:
            files_to_process.append(path)

    deleted_sources = set(manifest.keys()) - set(new_manifest.keys())

    # Fast-path: nothing changed
    if not rebuild and not files_to_process and not deleted_sources:
        print(
            f"[INFO] All {unchanged_count} files unchanged — skipping re-indexing."
        )
        return report

    # ----- Phase 2: decide collection strategy -----
    collection_exists_now = (
        settings.persist_dir.exists() and collection_exists(settings)
    )

    if rebuild or not collection_exists_now:
        # Full rebuild — wipe and re-create from all files
        cleanup_collections(settings, [settings.collection_name])
        files_to_process = all_files
        deleted_sources = set()
        incremental = False
    else:
        incremental = True

    # ----- Phase 3: delete stale/changed chunks from existing collection -----
    if incremental and (files_to_process or deleted_sources):
        client = build_client(settings)
        try:
            stale = deleted_sources | {str(p.relative_to(settings.data_dir)) for p in files_to_process}
            for rel_source in stale:
                _delete_chunks_by_source(client, settings.collection_name, rel_source)
        finally:
            close_fn = getattr(client, "close", None)
            if callable(close_fn):
                close_fn()

    # ----- Phase 4: parse & embed new/changed files -----
    documents = list(
        iter_documents(
            settings.data_dir,
            report=report,
            quarantine_dirname=settings.quarantine_dirname,
            files_filter=set(files_to_process) if incremental else None,
            progress_callback=progress_callback,
        )
    )

    if not documents:
        if incremental:
            print("[INFO] No new documents to embed after incremental classification.")
            _save_manifest(settings.persist_dir, new_manifest)
            return report
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

    if incremental:
        # Add new chunks to the existing collection
        vs = QdrantVectorStore.from_existing_collection(
            collection_name=settings.collection_name,
            embedding=embeddings,
            path=str(settings.persist_dir),
        )
        vs.add_documents(chunks)
    else:
        QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            path=str(settings.persist_dir),
            collection_name=settings.collection_name,
        )

    # ----- Phase 5: per-type hybrid collections (full rebuild only) -----
    # Vectors are already computed in Phase 4 — copy them via scroll+upsert,
    # no calls to the embedding model.
    if not incremental:
        type_groups: dict[str, list[Document]] = {}
        for chunk in chunks:
            type_groups.setdefault(chunk.metadata.get("doc_type", "generic"), []).append(chunk)

        client = build_client(settings)
        try:
            for doc_type in sorted(type_groups.keys()):
                type_collection = f"{settings.collection_name}-{doc_type}"
                count = _copy_points_by_doctype(
                    client, settings.collection_name, type_collection, doc_type
                )
                print(f"[INFO] {type_collection}: copied {count} points (no re-embedding)")
        finally:
            close_fn = getattr(client, "close", None)
            if callable(close_fn):
                close_fn()

    # ----- Phase 6: report & manifest -----
    if report.failed_files > 0:
        report.anomalies.append(
            f"Parser failures detected: {report.failed_files} file(s) moved to quarantine"
        )

    failure_rate = report.failed_files / max(report.failed_files + report.indexed_files, 1)
    if failure_rate > 0.35:
        report.anomalies.append(f"High parse failure rate: {failure_rate:.2%}")

    _save_manifest(settings.persist_dir, new_manifest)

    report_payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "collection": settings.collection_name,
        "supported_formats": SUPPORTED_SUMMARY,
        "ingestion": report.to_dict(),
    }
    write_report(settings.reports_dir / "latest_index_report.json", report_payload)

    mode_label = "incremental" if incremental else "full"
    print(
        f"[{mode_label}] Indexed {len(documents)} docs → {len(chunks)} chunks. "
        f"Unchanged: {unchanged_count}, failed: {report.failed_files}, quarantined: {report.quarantined_files}."
    )
    print(f"Report: {settings.reports_dir / 'latest_index_report.json'}")
    return report
