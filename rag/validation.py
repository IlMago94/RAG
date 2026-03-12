"""Architecture validation and retrieval smoke tests."""
from __future__ import annotations

from .config import RagSettings
from .vectorstore import build_client, build_vector_store, collection_exists


def run_retrieval_smoke_test(settings: RagSettings, max_checks: int = 8) -> dict:
    """Simple stability check: retrieve chunks with self-generated probes and measure source consistency."""
    if not collection_exists(settings):
        return {"checks": 0, "hits": 0, "hit_rate": 0.0, "anomalies": ["main collection missing"]}

    vector_store = build_vector_store(settings)
    probe_docs = vector_store.similarity_search("overview", k=max_checks)
    if not probe_docs:
        return {"checks": 0, "hits": 0, "hit_rate": 0.0, "anomalies": ["no chunks for smoke test"]}

    checks = 0
    hits = 0
    anomalies: list[str] = []
    retriever = vector_store.as_retriever(search_kwargs={"k": 1})

    for doc in probe_docs:
        snippet = " ".join(doc.page_content.split()[:18]).strip()
        if len(snippet) < 20:
            continue
        checks += 1
        top_result = retriever.invoke(snippet)
        if not top_result:
            anomalies.append(f"empty retrieval for source {doc.metadata.get('source', 'unknown')}")
            continue
        expected = doc.metadata.get("source", "")
        got = top_result[0].metadata.get("source", "")
        if expected and expected == got:
            hits += 1

    hit_rate = hits / max(checks, 1)
    if checks > 0 and hit_rate < 0.5:
        anomalies.append(f"low retrieval consistency: {hit_rate:.2%}")

    return {
        "checks": checks,
        "hits": hits,
        "hit_rate": round(hit_rate, 4),
        "anomalies": anomalies,
    }


def validate_architecture_conflicts(settings: RagSettings) -> dict:
    """Check for known architectural conflicts in indexed documents."""
    if not collection_exists(settings):
        return {"able_to_validate": False, "reason": "no indexed collection"}

    client = build_client(settings)
    conflicts: list[dict] = []

    try:
        collection_name = settings.collection_name

        # Paginated scroll — no hard limit, fetches all points
        all_points = []
        offset = None
        while True:
            batch, offset = client.scroll(
                collection_name=collection_name,
                limit=1000,
                offset=offset,
            )
            all_points.extend(batch)
            if offset is None:
                break

        csp14_sources: set = set()
        csp20_sources: set = set()
        incomp_warning_sources: set = set()

        for point in all_points:
            payload = point.payload
            text_content = payload.get("page_content", "")
            source = payload.get("source", "unknown")

            if "1.4" in text_content and "libcsp" in text_content.lower():
                csp14_sources.add(source)
            if "CSP version 2" in text_content or "version 2" in text_content.lower():
                csp20_sources.add(source)
            if "not supposed to be on the same network infrastructure" in text_content:
                incomp_warning_sources.add(source)

        if csp14_sources and csp20_sources and incomp_warning_sources:
            conflicts.append({
                "issue": "CSP_VERSION_CONFLICT",
                "severity": "CRITICAL",
                "description": "Architecture uses both CSP 1.4 and CSP 2.0, which are explicitly NOT compatible on the same network infrastructure.",
                "found_in_documents": {
                    "csp_1_4": sorted(list(csp14_sources)),
                    "csp_2_0": sorted(list(csp20_sources)),
                    "incompatibility_warning": sorted(list(incomp_warning_sources)),
                },
                "recommendation": "Your architecture_avionics file specifies libcsp 1.4. Verify all nodes use CSP 1.4 OR migrate entirely to CSP 2.0 (not mixed).",
            })
    except Exception as e:
        conflicts.append({
            "issue": "VALIDATION_ERROR",
            "error": str(e),
        })

    return {
        "able_to_validate": True,
        "conflicts_found": len(conflicts),
        "conflicts": conflicts,
    }
