"""Document chunking with type-aware splitting strategies."""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import RagSettings


def chunk_documents(documents: list[Document], settings: RagSettings) -> list[Document]:
    """Chunk docs with strategy tuned by document type to reduce malformed context."""
    grouped: dict[tuple[int, int], list[Document]] = {}
    for doc in documents:
        doc_type = doc.metadata.get("doc_type", "generic")
        if doc_type in {"pdf", "docx", "html", "image", "web_url"}:
            chunk_size, chunk_overlap = max(settings.chunk_size, 1200), max(settings.chunk_overlap, 180)
        elif doc_type in {"json", "xml", "log"}:
            chunk_size, chunk_overlap = min(settings.chunk_size, 900), min(settings.chunk_overlap, 120)
        else:
            chunk_size, chunk_overlap = settings.chunk_size, settings.chunk_overlap
        grouped.setdefault((chunk_size, chunk_overlap), []).append(doc)

    chunks: list[Document] = []
    for (chunk_size, chunk_overlap), docs_group in grouped.items():
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks.extend(splitter.split_documents(docs_group))

    for index, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk_id"] = index
    return chunks
