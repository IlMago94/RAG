import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so tests can import top-level modules
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rag import RagSettings, validate_architecture_conflicts
from rag.ingestion import iter_documents, IngestionReport

def test_validate_conflict_exists():
    settings = RagSettings.from_env()
    result = validate_architecture_conflicts(settings)
    # We expect at least one conflict in the current dataset (CSP 1.4 vs 2.x)
    assert result.get("able_to_validate") is True
    assert isinstance(result.get("conflicts"), list)
    assert len(result.get("conflicts")) >= 1

def test_pptx_ingestion():
    """Test ingestion of PowerPoint files."""
    data_dir = Path("data")
    report = IngestionReport()

    pptx_file = data_dir / "example.pptx"
    ppt_file = data_dir / "example.ppt"

    # Ensure the files exist for the test
    assert pptx_file.exists(), "example.pptx is missing in data directory."
    assert ppt_file.exists(), "example.ppt is missing in data directory."

    # Test .pptx ingestion
    docs = list(iter_documents(data_dir, report))
    pptx_docs = [doc for doc in docs if doc.metadata["extension"] == ".pptx"]
    assert len(pptx_docs) > 0, "No documents ingested for .pptx file."

    # Test .ppt ingestion
    ppt_docs = [doc for doc in docs if doc.metadata["extension"] == ".ppt"]
    assert len(ppt_docs) > 0, "No documents ingested for .ppt file."

    # Validate report
    assert report.indexed_files >= 2, "Not all PowerPoint files were indexed."
    assert "UnstructuredPPTLoader" in report.parser_usage, "PowerPoint parser not used."
