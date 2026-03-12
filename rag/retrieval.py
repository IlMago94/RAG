"""Question answering with conflict-aware retrieval."""
from __future__ import annotations

import json

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_qdrant import QdrantVectorStore

from .config import RagSettings, build_llm
from .vectorstore import build_vector_store, collection_exists
from .validation import validate_architecture_conflicts


_CONFLICT_KEYWORDS = {
    "incongruen", "conflict", "compatib", "incompatib",
    "inconsisten", "problem", "error",
    "mismatch", "discrepanc",
}


def _is_conflict_query(question: str) -> bool:
    """Return True when the question is about conflicts or incompatibilities."""
    q_lower = question.lower()
    return any(kw in q_lower for kw in _CONFLICT_KEYWORDS)


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


def ask_question(settings: RagSettings, question: str, show_context: bool) -> None:
    """
    Retrieve relevant context from the vector store and answer the user's question.
    """
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
        retriever = vector_store.as_retriever(search_kwargs={"k": settings.top_k})
        documents = retriever.invoke(question)
    if not documents:
        print("No relevant context found for the question.")
        return

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

    print(answer)

    if show_context:
        print("\n--- CONTEXT ---")
        print(context)
