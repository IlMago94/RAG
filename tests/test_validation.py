import os
import sys

# Ensure project root is on sys.path so tests can import top-level modules
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rag import RagSettings, validate_architecture_conflicts


def test_validate_conflict_exists():
    settings = RagSettings.from_env()
    result = validate_architecture_conflicts(settings)
    # We expect at least one conflict in the current dataset (CSP 1.4 vs 2.x)
    assert result.get("able_to_validate") is True
    assert isinstance(result.get("conflicts"), list)
    assert len(result.get("conflicts")) >= 1
