"""FastAPI web interface for the RAG pipeline."""
from __future__ import annotations

import os
import shutil
import threading
from pathlib import Path

from fastapi import FastAPI, File, Form, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from rag import (
    RagSettings,
    collection_exists,
    index_documents,
    ask_question,
    invalidate_answer_cache,
    validate_architecture_conflicts,
)

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

UPLOAD_DIR = BASE_DIR / "data"
UPLOAD_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Global indexing progress state
# ---------------------------------------------------------------------------
_index_lock = threading.Lock()
_index_progress: dict = {
    "total": 0,
    "processed": 0,
    "current": "",
    "done": True,
    "error": "",
}


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    files = []
    for root, dirs, filelist in os.walk(UPLOAD_DIR):
        for fname in filelist:
            rel = os.path.relpath(os.path.join(root, fname), UPLOAD_DIR)
            files.append(rel)

    settings = RagSettings.from_env()

    indexed = False
    try:
        indexed = settings.persist_dir.exists() and collection_exists(settings)
    except Exception:
        indexed = False

    validation = None
    try:
        validation = validate_architecture_conflicts(settings)
    except Exception:
        validation = {"able_to_validate": False}

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "files": files, "indexed": indexed, "validation": validation},
    )


@app.post("/delete")
def delete_file(filename: str = Query(...)):
    fpath = UPLOAD_DIR / filename
    if fpath.exists():
        fpath.unlink()
        return {"status": f"Deleted {filename}"}
    return {"status": f"File not found: {filename}"}


@app.post("/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    uploaded = []
    for file in files:
        dest = UPLOAD_DIR / file.filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        uploaded.append(file.filename)
    return {"filenames": uploaded, "status": f"{len(uploaded)} file(s) uploaded"}


@app.post("/index")
def index():
    def _run():
        def _cb(processed: int, total: int, current_file: str) -> None:
            with _index_lock:
                _index_progress["processed"] = processed
                _index_progress["total"] = total
                _index_progress["current"] = current_file

        with _index_lock:
            _index_progress.update({"total": 0, "processed": 0, "current": "", "done": False, "error": ""})
        try:
            settings = RagSettings.from_env()
            index_documents(settings, rebuild=False, progress_callback=_cb)
            invalidate_answer_cache()
        except Exception as exc:
            with _index_lock:
                _index_progress["error"] = str(exc)
        finally:
            with _index_lock:
                _index_progress["done"] = True

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "started"}


@app.get("/progress")
def progress():
    with _index_lock:
        return JSONResponse(dict(_index_progress))


@app.post("/validate")
def validate():
    settings = RagSettings.from_env()
    try:
        result = validate_architecture_conflicts(settings)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/query")
def query(question: str = Form(...)):
    settings = RagSettings.from_env()
    try:
        answer = ask_question(settings, question, show_context=False)
    except Exception as e:
        answer = f"Error: {e}"
    return JSONResponse({"answer": answer})
