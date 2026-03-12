"""Configuration, constants, and provider builders for the RAG pipeline."""
from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
    category=UserWarning,
)

from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


# ---------------------------------------------------------------------------
# Supported extensions and their priorities
# ---------------------------------------------------------------------------

# Extensions with high-fidelity dedicated parsers (PDF, Office, HTML, images).
PRIORITY_1_EXTENSIONS = {".pdf", ".docx", ".doc", ".html", ".htm", ".jpeg", ".jpg"}

# Plain-text and structured formats reliably loaded with simple loaders.
PRIORITY_2_EXTENSIONS = {".md", ".txt", ".json", ".xml", ".log"}

# Text-like extensions without dedicated loaders: TextLoader is used as fallback.
TEXT_FALLBACK_EXTENSIONS = {
    ".csv",
    ".py",
    ".rst",
    ".text",
    ".yaml",
    ".yml",
}

# Readable summary of supported formats.
SUPPORTED_SUMMARY = {
    "priority_1": sorted(PRIORITY_1_EXTENSIONS),
    "priority_2": sorted(PRIORITY_2_EXTENSIONS),
    "priority_3": "all other file extensions",
}


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RagSettings:
    """
    Configuration settings for the RAG pipeline, loaded from environment variables.
    """
    llm_provider: str
    embed_provider: str
    chat_model: str
    embedding_model: str
    ollama_base_url: str
    openai_base_url: str
    openai_api_key: str
    collection_name: str
    data_dir: Path
    persist_dir: Path
    chunk_size: int
    chunk_overlap: int
    top_k: int
    quarantine_dirname: str
    reports_dir: Path

    @classmethod
    def from_env(cls) -> "RagSettings":
        """
        Load settings from environment variables, with sensible defaults.
        """
        load_dotenv()
        base_dir = Path(__file__).resolve().parent.parent
        return cls(
            llm_provider=os.getenv("RAG_LLM_PROVIDER", "ollama").strip().lower(),
            embed_provider=os.getenv("RAG_EMBED_PROVIDER", "ollama").strip().lower(),
            chat_model=os.getenv("RAG_CHAT_MODEL", "qwen3:8b").strip(),
            embedding_model=os.getenv("RAG_EMBED_MODEL", "qwen3-embedding:4b").strip(),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip(),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:8080/v1").strip(),
            openai_api_key=os.getenv("OPENAI_API_KEY", "localai").strip(),
            collection_name=os.getenv("RAG_COLLECTION_NAME", "local-rag").strip(),
            data_dir=base_dir / os.getenv("RAG_DATA_DIR", "data"),
            persist_dir=base_dir / os.getenv("RAG_PERSIST_DIR", "storage/chroma"),
            chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "150")),
            top_k=int(os.getenv("RAG_TOP_K", "4")),
            quarantine_dirname=os.getenv("RAG_QUARANTINE_DIR", "quarantine").strip(),
            reports_dir=base_dir / os.getenv("RAG_REPORTS_DIR", "reports"),
        )


# ---------------------------------------------------------------------------
# Provider builders
# ---------------------------------------------------------------------------

def build_llm(settings: RagSettings):
    """
    Instantiate the chat LLM based on provider and model settings.
    """
    if settings.llm_provider == "ollama":
        return ChatOllama(
            model=settings.chat_model,
            base_url=settings.ollama_base_url,
            temperature=0,
        )
    if settings.llm_provider in {"localai", "openai_compatible", "openai-compatible"}:
        return ChatOpenAI(
            model=settings.chat_model,
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            temperature=0,
        )
    raise ValueError(
        "RAG_LLM_PROVIDER must be 'ollama' or 'localai'/'openai_compatible'."
    )


def build_embeddings(settings: RagSettings):
    """
    Instantiate the embedding model based on provider and model settings.
    """
    if settings.embed_provider == "ollama":
        return OllamaEmbeddings(
            model=settings.embedding_model,
            base_url=settings.ollama_base_url,
        )
    if settings.embed_provider in {"localai", "openai_compatible", "openai-compatible"}:
        return OpenAIEmbeddings(
            model=settings.embedding_model,
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
        )
    raise ValueError(
        "RAG_EMBED_PROVIDER must be 'ollama' or 'localai'/'openai_compatible'."
    )


# ---------------------------------------------------------------------------
# Report utility
# ---------------------------------------------------------------------------

def write_report(report_path: Path, payload: dict) -> None:
    """Serialize the payload as indented JSON and write it to report_path."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
