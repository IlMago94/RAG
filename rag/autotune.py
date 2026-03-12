"""Autonomous indexing + validation loop (autotune)."""
from __future__ import annotations

import time
from datetime import datetime, timezone

from .config import RagSettings, write_report
from .vectorstore import index_documents
from .validation import run_retrieval_smoke_test


def run_autotune(settings: RagSettings, rebuild: bool, watch: bool, interval_seconds: int, max_cycles: int) -> None:
    """Autonomous routine: index, validate, auto-retry in watch mode."""
    cycle = 0
    while True:
        cycle += 1
        cycle_rebuild = rebuild and cycle == 1
        print(f"[AUTOTUNE] cycle={cycle} rebuild={cycle_rebuild}")
        cycle_started = datetime.now(timezone.utc).isoformat()

        try:
            report = index_documents(settings, rebuild=cycle_rebuild)
            smoke = run_retrieval_smoke_test(settings)
            outcome = {
                "cycle": cycle,
                "started_at_utc": cycle_started,
                "ingestion": report.to_dict(),
                "smoke_test": smoke,
                "status": "ok" if not (report.anomalies or smoke.get("anomalies")) else "warning",
            }
        except Exception as exc:
            outcome = {
                "cycle": cycle,
                "started_at_utc": cycle_started,
                "status": "error",
                "error": str(exc),
            }

        write_report(settings.reports_dir / "autotune_latest.json", outcome)
        print(f"[AUTOTUNE] status={outcome['status']} report={settings.reports_dir / 'autotune_latest.json'}")

        if not watch:
            return
        if max_cycles > 0 and cycle >= max_cycles:
            print("[AUTOTUNE] reached max cycles, stopping.")
            return
        time.sleep(max(interval_seconds, 5))
