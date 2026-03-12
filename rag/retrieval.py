"""Question answering with conflict-aware retrieval."""
from __future__ import annotations

import hashlib
import json

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

from .config import RagSettings, build_llm
from .vectorstore import build_client, build_vector_store, collection_exists
from .validation import validate_architecture_conflicts


# ---------------------------------------------------------------------------
# Optional cross-encoder for re-ranking (degrades gracefully if unavailable)
# ---------------------------------------------------------------------------

try:
    from sentence_transformers import CrossEncoder as _CE
    _cross_encoder = _CE("cross-encoder/ms-marco-MiniLM-L-6-v2")
except Exception:
    _cross_encoder = None


# ---------------------------------------------------------------------------
# In-memory answer cache
# ---------------------------------------------------------------------------

_answer_cache: dict[str, str] = {}
_CACHE_MAX = 128


def _cache_key(question: str, settings: RagSettings) -> str:
    raw = f"{question.strip().lower()}|{settings.collection_name}|{settings.top_k}"
    return hashlib.md5(raw.encode()).hexdigest()


def invalidate_answer_cache() -> None:
    """Clear the answer cache (call after re-indexing)."""
    _answer_cache.clear()


_CONFLICT_KEYWORDS = {
    "incongruen", "conflict", "compatib", "incompatib",
    "inconsisten", "problem", "error",
    "mismatch", "discrepanc",
}


def _is_conflict_query(question: str) -> bool:
    """Return True when the question is about conflicts or incompatibilities."""
    q_lower = question.lower()
    return any(kw in q_lower for kw in _CONFLICT_KEYWORDS)


def _rerank(docs: list[Document], question: str, top_k: int) -> list[Document]:
    """Re-rank docs with a cross-encoder; falls back to score order if unavailable."""
    if _cross_encoder is None or len(docs) <= 1:
        return docs[:top_k]
    pairs = [(question, doc.page_content) for doc in docs]
    scores = list(_cross_encoder.predict(pairs))
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]


def _expand_context_window(
    docs: list[Document],
    settings: RagSettings,
    window: int = 1,
) -> list[Document]:
    """For each retrieved chunk, try to fetch its immediate neighbours (±window)
    from Qdrant so the LLM sees a broader passage.
    Neighbour chunks are appended after the original list and deduplicated.
    """
    if not docs:
        return docs

    client = build_client(settings)
    seen_ids: set[int] = {d.metadata.get("chunk_id") for d in docs if "chunk_id" in d.metadata}
    extra: list[Document] = []

    try:
        for doc in docs:
            chunk_id = doc.metadata.get("chunk_id")
            source = doc.metadata.get("source")
            if chunk_id is None or source is None:
                continue
            neighbour_ids = [i for i in range(chunk_id - window, chunk_id + window + 1) if i != chunk_id and i > 0]
            if not neighbour_ids:
                continue
            results, _ = client.scroll(
                collection_name=settings.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="metadata.source", match=MatchValue(value=source)),
                        FieldCondition(key="metadata.chunk_id", match=MatchAny(any=neighbour_ids)),
                    ]
                ),
                limit=window * 2 + 2,
                with_payload=True,
            )
            for point in results:
                nid = (point.payload.get("metadata") or {}).get("chunk_id")
                if nid not in seen_ids:
                    seen_ids.add(nid)
                    extra.append(
                        Document(
                            page_content=point.payload.get("page_content", ""),
                            metadata=point.payload.get("metadata", {}),
                        )
                    )
    except Exception:
        pass
    finally:
        close_fn = getattr(client, "close", None)
        if callable(close_fn):
            close_fn()

    return docs + extra


def _multi_query_retrieval(vector_store: QdrantVectorStore, question: str, k: int) -> list[Document]:
    """Retrieve docs using the original query PLUS targeted sub-queries to maximize coverage."""
    sub_queries = [
        question,
        "avionics architecture boards CSP version",
        "CSP version 1 version 2 compatibility network",
        "libcsp protocol version CAN bus",
    ]
    seen_contents: set[str] = set()
    merged: list[Document] = []
    per_query_k = max(k // len(sub_queries), 4)

    for q in sub_queries:
        results = vector_store.similarity_search(q, k=per_query_k)
        for doc in results:
            content_key = doc.page_content[:200]
            if content_key not in seen_contents:
                seen_contents.add(content_key)
                merged.append(doc)

    return merged[:k]


def ask_question(settings: RagSettings, question: str, show_context: bool) -> str:
    """
    Retrieve relevant context from the vector store and answer the user's question.

    Returns the answer string (also prints it for CLI compatibility).
    Caches results in-memory; cache is keyed by question + collection + top_k.
    """
    # --- Cache lookup ---
    cache_key = _cache_key(question, settings)
    if cache_key in _answer_cache:
        cached = _answer_cache[cache_key]
        print(cached)
        return cached

    if not settings.persist_dir.exists() or not collection_exists(settings):
        raise FileNotFoundError(
            f"Vector store not found in {settings.persist_dir}. Run the index command first."
        )

    vector_store = build_vector_store(settings)
    llm = build_llm(settings)
    conflict_mode = _is_conflict_query(question)

    if conflict_mode:
        effective_k = max(settings.top_k * 3, 12)
        documents = _multi_query_retrieval(vector_store, question, k=effective_k)
    else:
        retriever = vector_store.as_retriever(search_kwargs={"k": settings.top_k * 2})
        documents = retriever.invoke(question)
        # Re-rank and apply context window only for non-conflict mode
        documents = _rerank(documents, question, settings.top_k)
        documents = _expand_context_window(documents, settings)

    if not documents:
        print("No relevant context found for the question.")
        return ""

    context = "\n\n".join(
        f"Source: {document.metadata.get('source', 'unknown')}\n{document.page_content}"
        for document in documents
    )

    validation_addendum = ""
    if conflict_mode:
        try:
            vresult = validate_architecture_conflicts(settings)
            if vresult.get("conflicts"):
                parts = []
                for c in vresult["conflicts"]:
                    parts.append(
                        f"DETECTED CONFLICT [{c.get('severity','?')}]: {c.get('description','')}\n"
                        f"  Involved documents: {json.dumps(c.get('found_in_documents', {}), ensure_ascii=False)}\n"
                        f"  Recommendation: {c.get('recommendation','')}"
                    )
                validation_addendum = (
                    "\n\n=== AUTOMATIC VALIDATION RESULTS ===\n"
                    + "\n".join(parts)
                    + "\n=== END VALIDATION ==="
                )
        except Exception:
            pass

    if conflict_mode:
        system_msg = (
            "You are a RAG assistant specialized in architectural analysis. "
            "Your task is to analyze the provided context and identify ALL inconsistencies, "
            "version conflicts, incompatibilities between components or protocols, "
            "and any discrepancy across different documents. "
            "Actively compare information between different documents. "
            "If you find conflicts, explain clearly: what conflicts, where it is defined, and why it is a problem. "
            "If you do not find conflicts, state it explicitly. "
            "Always end with a 'Sources:' section listing the files used.\n\n"
            "Context:\n{context}" + validation_addendum
        )
    else:
        system_msg = (
            "You are a RAG assistant. Answer only using the provided context. "
            "If the context is not sufficient, state it explicitly. "
            "Always end with a 'Sources:' section listing the files used.\n\n"
            "Context:\n{context}"
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_msg),
            ("human", "Question: {question}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    # --- Cache store (FIFO eviction) ---
    if len(_answer_cache) >= _CACHE_MAX:
        oldest = next(iter(_answer_cache))
        del _answer_cache[oldest]
    _answer_cache[cache_key] = answer

    print(answer)

    if show_context:
        print("\n--- CONTEXT ---")
        print(context)

    return answer
