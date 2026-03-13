"""Document ingestion: loading, parsing with autofix fallback, and quarantine."""
from __future__ import annotations

import concurrent.futures
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

_DEFAULT_INGEST_WORKERS = 8


def _render_progress_bar(done: int, total: int, current: str, width: int = 28) -> str:
    """Build a single-line ASCII progress bar string."""
    pct = done / total if total else 0
    filled = int(width * pct)
    bar = "=" * filled + (">" if filled < width else "") + " " * max(0, width - filled - 1)
    name = Path(current).name[:38] if current else ""
    return f"\r[{bar}] {done}/{total} ({pct:.0%})  {name:<38}"

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    PDFPlumberLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredURLLoader,
    UnstructuredPowerPointLoader,
)
from langchain_unstructured import UnstructuredLoader

from .config import PRIORITY_2_EXTENSIONS, TEXT_FALLBACK_EXTENSIONS


# ---------------------------------------------------------------------------
# Ingestion report
# ---------------------------------------------------------------------------

@dataclass
class IngestionReport:
    """Accumulate statistics and diagnostics during a single ingestion run."""

    indexed_files: int = 0
    failed_files: int = 0
    total_docs: int = 0
    total_chunks: int = 0
    quarantined_files: int = 0
    parser_usage: dict[str, int] = field(default_factory=dict)
    failures: list[dict[str, str]] = field(default_factory=list)
    anomalies: list[str] = field(default_factory=list)

    def register_parser(self, parser_name: str) -> None:
        self.parser_usage[parser_name] = self.parser_usage.get(parser_name, 0) + 1

    def to_dict(self) -> dict:
        return {
            "indexed_files": self.indexed_files,
            "failed_files": self.failed_files,
            "total_docs": self.total_docs,
            "total_chunks": self.total_chunks,
            "quarantined_files": self.quarantined_files,
            "parser_usage": self.parser_usage,
            "failures": self.failures,
            "anomalies": self.anomalies,
        }


# ---------------------------------------------------------------------------
# File classification
# ---------------------------------------------------------------------------

def get_doc_priority_and_type(path: Path) -> tuple[int, str]:
    """Classify file with explicit document priorities and type labels."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        return 1, "pdf"
    if ext in {".docx", ".doc"}:
        return 1, "docx"
    if ext in {".html", ".htm"}:
        return 1, "html"
    if ext in {".jpeg", ".jpg"}:
        return 1, "image"
    if ext in {".pptx", ".ppt"}:
        return 1, "presentation"
    if ext == ".md":
        return 2, "markdown"
    if ext == ".txt":
        return 2, "text"
    if ext == ".json":
        return 2, "json"
    if ext == ".xml":
        return 2, "xml"
    if ext == ".log":
        return 2, "log"
    if ext in TEXT_FALLBACK_EXTENSIONS:
        return 3, "text_generic"
    return 3, "generic"


# ---------------------------------------------------------------------------
# Loader candidates (autofix fallback chain)
# ---------------------------------------------------------------------------

def _loader_candidates(path: Path) -> list[tuple[str, Callable[[], object]]]:
    """Build ordered parser candidates for autofix fallback."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        return [
            ("PDFPlumberLoader", lambda: PDFPlumberLoader(str(path))),
            ("UnstructuredLoader", lambda: UnstructuredLoader(file_path=str(path))),
        ]
    if ext in {".docx", ".doc", ".jpeg", ".jpg"}:
        return [
            ("UnstructuredLoader", lambda: UnstructuredLoader(file_path=str(path))),
        ]
    if ext in {".html", ".htm"}:
        return [
            ("UnstructuredHTMLLoader", lambda: UnstructuredHTMLLoader(str(path))),
            ("TextLoader(utf-8)", lambda: TextLoader(str(path), encoding="utf-8")),
            ("UnstructuredLoader", lambda: UnstructuredLoader(file_path=str(path))),
        ]
    if ext == ".md":
        return [
            ("UnstructuredMarkdownLoader", lambda: UnstructuredMarkdownLoader(str(path))),
            ("TextLoader(utf-8)", lambda: TextLoader(str(path), encoding="utf-8")),
            ("TextLoader(latin-1)", lambda: TextLoader(str(path), encoding="latin-1")),
        ]
    if ext in {".pptx", ".ppt"}:
        return [
            ("UnstructuredPPTLoader", lambda: UnstructuredPowerPointLoader(file_path=str(path))),
            ("UnstructuredLoader", lambda: UnstructuredLoader(file_path=str(path))),
        ]
    if ext in PRIORITY_2_EXTENSIONS or ext in TEXT_FALLBACK_EXTENSIONS:
        return [
            ("TextLoader(utf-8)", lambda: TextLoader(str(path), encoding="utf-8")),
            ("TextLoader(latin-1)", lambda: TextLoader(str(path), encoding="latin-1")),
            ("UnstructuredLoader", lambda: UnstructuredLoader(file_path=str(path))),
        ]
    return [
        ("UnstructuredLoader", lambda: UnstructuredLoader(file_path=str(path))),
        ("TextLoader(latin-1)", lambda: TextLoader(str(path), encoding="latin-1")),
    ]


def _sanitize_loaded_docs(docs: list[Document]) -> list[Document]:
    """Keep only documents with non-empty text."""
    sanitized: list[Document] = []
    for doc in docs:
        content = (doc.page_content or "").strip()
        if content:
            doc.page_content = content
            sanitized.append(doc)
    return sanitized


# ---------------------------------------------------------------------------
# Quarantine
# ---------------------------------------------------------------------------

def _send_to_quarantine(path: Path, data_dir: Path, quarantine_dirname: str) -> Path:
    """Move an unparseable file into the quarantine subdirectory."""
    quarantine_dir = data_dir / quarantine_dirname
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    relative = path.relative_to(data_dir)
    target = quarantine_dir / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(target))
    return target


# ---------------------------------------------------------------------------
# Single-file parser (used by parallel ingestion)
# ---------------------------------------------------------------------------

def _parse_single_file(
    path: Path,
    data_dir: Path,
) -> tuple[list[Document], str, list[str]]:
    """Parse one file through the autofix fallback chain.

    Returns (docs_with_metadata, parser_used, errors).
    Caller is responsible for quarantine and report updates.
    """
    priority, doc_type = get_doc_priority_and_type(path)
    candidates = _loader_candidates(path)
    docs: list[Document] = []
    parser_used = "unknown"
    errors: list[str] = []

    for parser_name, factory in candidates:
        try:
            loader = factory()
            loaded = loader.load()
            cleaned = _sanitize_loaded_docs(loaded)
            if not cleaned:
                errors.append(f"{parser_name}: empty content")
                continue
            parser_used = parser_name
            docs = cleaned
            break
        except Exception as exc:
            errors.append(f"{parser_name}: {exc}")

    for index, doc in enumerate(docs, start=1):
        doc.metadata["source"] = str(path.relative_to(data_dir))
        doc.metadata["doc_type"] = doc_type
        doc.metadata["priority"] = priority
        doc.metadata["extension"] = path.suffix.lower()
        doc.metadata["parser_used"] = parser_used
        doc.metadata["source_path"] = str(path)
        doc.metadata["doc_part"] = index
        doc.metadata["ingested_at_utc"] = datetime.now(timezone.utc).isoformat()

    return docs, parser_used, errors


# ---------------------------------------------------------------------------
# Main iteration
# ---------------------------------------------------------------------------

def iter_documents(
    data_dir: Path,
    report: IngestionReport | None = None,
    quarantine_dirname: str = "quarantine",
    max_workers: int = _DEFAULT_INGEST_WORKERS,
    files_filter: set[Path] | None = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Iterable[Document]:
    """Iterate files with parallel parsing, autofix fallback, and quarantine.

    Args:
        data_dir: Root directory containing documents.
        report: Optional report accumulator.
        quarantine_dirname: Subdirectory name used for quarantined files.
        max_workers: Number of parallel parser threads (default 4).
        files_filter: If provided, only process files in this set.
        progress_callback: Optional callable(processed, total, current_file).
            If None, a progress bar is printed to stderr in CLI mode.
    """
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}. Create it and add your documents first."
        )

    file_paths = [
        p for p in sorted(data_dir.rglob("*"))
        if p.is_file()
        and quarantine_dirname not in p.parts
        and (files_filter is None or p in files_filter)
    ]

    # Submit all files to the thread pool; preserve sorted order via ordered futures list.
    total = len(file_paths)
    done = 0
    use_cli_bar = progress_callback is None and sys.stderr.isatty()

    if use_cli_bar and total:
        sys.stderr.write(_render_progress_bar(0, total, ""))
        sys.stderr.flush()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            (path, executor.submit(_parse_single_file, path, data_dir))
            for path in file_paths
        ]

        for path, future in futures:
            try:
                docs, parser_used, errors = future.result()
            except Exception as exc:
                docs, parser_used, errors = [], "unknown", [str(exc)]

            done += 1
            if progress_callback is not None:
                progress_callback(done, total, str(path))
            elif use_cli_bar:
                sys.stderr.write(_render_progress_bar(done, total, str(path)))
                sys.stderr.flush()

            if not docs:
                print(f"[WARN] Failed parsing {path}: {' | '.join(errors)}")
                if report:
                    report.failed_files += 1
                    report.failures.append(
                        {
                            "file": str(path.relative_to(data_dir)),
                            "reason": "all parsers failed",
                            "errors": " | ".join(errors),
                        }
                    )
                try:
                    quarantined_to = _send_to_quarantine(path, data_dir, quarantine_dirname)
                    if report:
                        report.quarantined_files += 1
                    print(f"[INFO] Quarantined {path} -> {quarantined_to}")
                except Exception as quarantine_error:
                    print(f"[WARN] Quarantine failed for {path}: {quarantine_error}")
                continue

            if report:
                report.indexed_files += 1
                report.register_parser(parser_used)

            yield from docs

    if use_cli_bar and total:
        sys.stderr.write("\n")
        sys.stderr.flush()

    # Support for web URLs
    url_file = data_dir / "urls.txt"
    if url_file.exists():
        with url_file.open("r", encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip()]
        if urls:
            try:
                url_loader = UnstructuredURLLoader(urls)
                docs = url_loader.load()
                for doc in docs:
                    doc.metadata["source"] = doc.metadata.get("source", "web")
                    doc.metadata["doc_type"] = "web_url"
                    doc.metadata["priority"] = 1
                    doc.metadata["extension"] = "url"
                    doc.metadata["parser_used"] = "UnstructuredURLLoader"
                    doc.metadata["ingested_at_utc"] = datetime.now(timezone.utc).isoformat()
                    if report:
                        report.register_parser("UnstructuredURLLoader")
                    yield doc
            except Exception as e:
                print(f"[WARN] Skipping URLs: {e}")
