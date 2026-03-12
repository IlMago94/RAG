"""Command-line interface for the RAG pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import RagSettings, SUPPORTED_SUMMARY
from .vectorstore import index_documents
from .retrieval import ask_question
from .validation import validate_architecture_conflicts
from .autotune import run_autotune


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(description="Local RAG starter for Ollama and LocalAI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Chunk and index local documents.")
    index_parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete the existing local vector store before indexing.",
    )

    ask_parser = subparsers.add_parser("ask", help="Ask a question over the indexed documents.")
    ask_parser.add_argument("question", help="Question to send to the RAG pipeline.")
    ask_parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print the retrieved chunks used as context.",
    )

    subparsers.add_parser("config", help="Print the active configuration.")
    subparsers.add_parser("formats", help="Print supported formats and priority rules.")

    validate_parser = subparsers.add_parser(
        "validate",
        help="Check indexed documents for architectural conflicts and inconsistencies.",
    )
    validate_parser.add_argument(
        "--report",
        type=str,
        default="",
        help="Save validation report to file.",
    )

    autotune_parser = subparsers.add_parser(
        "autotune",
        help="Run autonomous indexing + validation. Supports one-shot and watch mode.",
    )
    autotune_parser.add_argument("--rebuild", action="store_true", help="Rebuild vector storage at first cycle.")
    autotune_parser.add_argument("--watch", action="store_true", help="Run continuously.")
    autotune_parser.add_argument("--interval", type=int, default=60, help="Seconds between watch cycles (default: 60).")
    autotune_parser.add_argument("--max-cycles", type=int, default=0, help="Stop after N cycles. 0 = unlimited.")

    return parser


def print_config(settings: RagSettings) -> None:
    """Print the current RAG pipeline configuration."""
    print("RAG configuration")
    print(f"- LLM provider: {settings.llm_provider}")
    print(f"- Embedding provider: {settings.embed_provider}")
    print(f"- Chat model: {settings.chat_model}")
    print(f"- Embedding model: {settings.embedding_model}")
    print(f"- Ollama URL: {settings.ollama_base_url}")
    print(f"- OpenAI-compatible URL: {settings.openai_base_url}")
    print(f"- Data directory: {settings.data_dir}")
    print(f"- Persist directory: {settings.persist_dir}")
    print("- Vector store: qdrant-local")
    print(f"- Collection: {settings.collection_name}")
    print(f"- Chunk size: {settings.chunk_size}")
    print(f"- Chunk overlap: {settings.chunk_overlap}")
    print(f"- Top K: {settings.top_k}")
    print(f"- Quarantine dir name: {settings.quarantine_dirname}")
    print(f"- Reports dir: {settings.reports_dir}")


def print_supported_formats() -> None:
    """Print the supported format priority table."""
    print("Supported format priorities")
    print(f"- Priority 1: {', '.join(SUPPORTED_SUMMARY['priority_1'])}")
    print(f"- Priority 2: {', '.join(SUPPORTED_SUMMARY['priority_2'])}")
    print("- Priority 3: all remaining file extensions")
    print("Hybrid indexing enabled: main collection + per-doc-type collections")


def main() -> None:
    """Entry point: parse arguments and execute the selected command."""
    parser = build_parser()
    args = parser.parse_args()
    settings = RagSettings.from_env()

    if args.command == "index":
        index_documents(settings, rebuild=args.rebuild)
        return

    if args.command == "ask":
        ask_question(settings, question=args.question, show_context=args.show_context)
        return

    if args.command == "config":
        print_config(settings)
        return

    if args.command == "formats":
        print_supported_formats()
        return

    if args.command == "validate":
        result = validate_architecture_conflicts(settings)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if args.report:
            Path(args.report).parent.mkdir(parents=True, exist_ok=True)
            Path(args.report).write_text(
                json.dumps(result, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"Report saved to: {args.report}")
        return

    if args.command == "autotune":
        run_autotune(
            settings,
            rebuild=args.rebuild,
            watch=args.watch,
            interval_seconds=args.interval,
            max_cycles=args.max_cycles,
        )
        return

    parser.error(f"Unsupported command: {args.command}")
