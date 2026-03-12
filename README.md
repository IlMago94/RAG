# Local RAG Starter

Local RAG proof of concept with:

- Ollama or LocalAI as LLM
- Qdrant local mode as persistent vector store
- LangChain as orchestration layer
- CLI (`test_rag.py`) + Web UI (`web_server.py`)

## Requirements

- Linux/macOS/WSL
- Python 3.10+
- Ollama or LocalAI running

## Quick setup

1. Create/activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

1. Install base dependencies:

```bash
pip install -r requirements.txt
```

1. Create local configuration:

```bash
cp .env.example .env
```

1. Put your documents in `data/`.

## `.env` configuration

Ollama example:

```env
RAG_LLM_PROVIDER=ollama
RAG_EMBED_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
RAG_CHAT_MODEL=qwen3:8b
RAG_EMBED_MODEL=qwen3-embedding:4b
```

LocalAI example (OpenAI-compatible):

```env
RAG_LLM_PROVIDER=localai
RAG_EMBED_PROVIDER=localai
OPENAI_BASE_URL=http://localhost:8080/v1
OPENAI_API_KEY=localai
RAG_CHAT_MODEL=<chat-model-name>
RAG_EMBED_MODEL=<embedding-model-name>
```

Supported hybrid mode:

```env
RAG_LLM_PROVIDER=localai
RAG_EMBED_PROVIDER=ollama
```

## CLI usage

Check config:

```bash
./.venv/bin/python test_rag.py config
```

Index from scratch:

```bash
./.venv/bin/python test_rag.py index --rebuild
```

Show format/priority rules:

```bash
./.venv/bin/python test_rag.py formats
```

One-shot autonomous routine (index + stability checks + report):

```bash
./.venv/bin/python test_rag.py autotune --rebuild
```

Continuous autonomous routine (watch mode):

```bash
./.venv/bin/python test_rag.py autotune --watch --interval 60
```

Ask a question:

```bash
./.venv/bin/python test_rag.py ask "Can CSP handle 60 nodes?" --show-context
```

## Web UI usage

Start server:

```bash
./.venv/bin/python -m uvicorn web_server:app --reload
```

Open: `http://127.0.0.1:8000`

From the UI you can:

- upload files
- delete files
- index documents
- run queries

## Supported formats

- Priority 1: `.pdf`, `.docx`, `.doc`, `.html`, `.htm`, `.jpeg`, `.jpg`
- Priority 2: `.md`, `.txt`, `.json`, `.xml`, `.log`
- Priority 3: all other formats (automatic fallback)
- Web URLs from `data/urls.txt` (one URL per line)
- Each file gets metadata (`doc_type`, `priority`, `extension`, `parser_used`, `source`)
- Hybrid indexing: main collection + separate collections by document type

## Autofix and quarantine

- Each file is parsed with a fallback parser chain (autofix).
- If all parsers fail or content is empty, the file is moved to `data/quarantine/`.
- The routine continues without interruption and produces JSON reports in `reports/`.

Main reports:

- `reports/latest_index_report.json`
- `reports/autotune_latest.json`

## Important notes

- The deprecation warning for `UnstructuredFileLoader` has been removed: `UnstructuredLoader` is now used.
- The `autotune` routine runs automatic retrieval stability checks (smoke test) and reports anomalies.
- If you see `python-dotenv could not parse statement...`, check your `.env` format (`KEY=VALUE` lines, no extra un-commented text).
- If the `python` command does not exist on your system, always use `python3` to create the venv and `./.venv/bin/python` to run project commands.
- If `uvicorn` is not in `PATH`, always start the Web UI with `./.venv/bin/python -m uvicorn web_server:app --reload`.
- For a first end-to-end check: `config` -> `index --rebuild` -> `ask`.

## Project structure

- `test_rag.py`: CLI RAG pipeline (config, indexing, ask)
- `web_server.py`: FastAPI API + UI integration
- `templates/index.html`: web page
- `static/style.css`: style
- `data/`: source documents
- `storage/`: local Qdrant persistence
