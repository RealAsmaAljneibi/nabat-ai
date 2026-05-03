"""
src/al_nassikh/ingest/queue_for_review.py
==========================================
Why this file exists: §M2 item 10 — after pdf_to_pages extracts page images and
triage.py runs, the pages that are NORMAL (not DEGRADED) need to be tracked so
the Operator Console knows what's waiting for annotation. This module writes and
reads `data/ground_truth/operator_queue.json` — a simple JSON list of pending
page entries. The Streamlit Archive Manager tab renders this queue as a task list.
Where it's called: app Tab B (live), tests. Purpose: HITL review queue — tracks page status (pending → in_review → complete)    

Why a JSON file (not a database): the whole system runs on any laptop with no
infrastructure beyond Python. A JSON queue is inspectable, diffable in git, and
trivially backed up. If the queue grows large (>10k items), a SQLite migration
is straightforward.

Architecture ref: §M2 item 10, §3.1 Tab A — Archive Manager Console.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ── Path resolution ───────────────────────────────────────────────────────────

def _repo_root() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in [here, here.parent, here.parent.parent, here.parent.parent.parent]:
        if (candidate / "requirements.txt").exists():
            return candidate
    return Path(".")


QUEUE_PATH = _repo_root() / "data" / "ground_truth" / "operator_queue.json"

# ── Queue entry status values ─────────────────────────────────────────────────

STATUS_PENDING    = "pending"       # awaiting human annotation in eScriptorium
STATUS_IN_REVIEW  = "in_review"     # annotator has opened the document
STATUS_COMPLETE   = "complete"      # annotator marked as done; PAGE-XML exported
STATUS_DEGRADED   = "degraded"      # triage flagged as DEGRADED; specialist track


def _load_queue(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def _save_queue(queue: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(queue, f, ensure_ascii=False, indent=2)


def enqueue_pages(
    pages: list[dict],
    queue_path: Optional[Path] = None,
) -> dict:
    """
    Add page entries to the operator review queue.

    Args:
        pages: List of page dicts. Each must have at minimum:
            {
              "page_path":    str   — absolute path to the PNG,
              "manuscript_id": str  — short stem (e.g., "manuscript07"),
              "page_number":  int,
              "triage_status": str  — "NORMAL" | "DEGRADED",
              "triage_score":  float,
            }
            Optional: "eScriptorium_document_id" (set after upload).

        queue_path: Override for testing. Defaults to data/ground_truth/operator_queue.json.

    Returns:
        {
          "enqueued":   int   — number of new entries added
          "skipped":    int   — pages already in queue (by page_path)
          "queue_size": int   — total queue size after enqueue
        }
    """
    path = Path(queue_path) if queue_path else QUEUE_PATH
    queue = _load_queue(path)

    existing_paths = {e["page_path"] for e in queue}
    enqueued = 0
    skipped  = 0

    for page in pages:
        page_path = str(page.get("page_path", ""))
        if page_path in existing_paths:
            skipped += 1
            continue

        triage_status = page.get("triage_status", "NORMAL")
        entry = {
            "page_path":              page_path,
            "manuscript_id":          page.get("manuscript_id", ""),
            "page_number":            page.get("page_number", 0),
            "triage_status":          triage_status,
            "triage_score":           page.get("triage_score", 0.0),
            "status": (
                STATUS_DEGRADED if triage_status == "DEGRADED"
                else STATUS_PENDING
            ),
            "escriptorium_document_id": page.get("escriptorium_document_id"),
            "enqueued_at":            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "completed_at":           None,
            "annotator_notes":        "",
        }
        queue.append(entry)
        existing_paths.add(page_path)
        enqueued += 1

    _save_queue(queue, path)
    logger.info("Enqueued %d pages (%d skipped); queue size = %d", enqueued, skipped, len(queue))

    return {
        "enqueued":   enqueued,
        "skipped":    skipped,
        "queue_size": len(queue),
    }


def update_status(
    page_path: str,
    new_status: str,
    escriptorium_document_id: Optional[int] = None,
    annotator_notes: str = "",
    queue_path: Optional[Path] = None,
) -> bool:
    """
    Update the status of a queue entry by page_path.
    Returns True if the entry was found and updated.
    """
    path = Path(queue_path) if queue_path else QUEUE_PATH
    queue = _load_queue(path)

    for entry in queue:
        if entry["page_path"] == page_path:
            entry["status"] = new_status
            if escriptorium_document_id is not None:
                entry["escriptorium_document_id"] = escriptorium_document_id
            if annotator_notes:
                entry["annotator_notes"] = annotator_notes
            if new_status == STATUS_COMPLETE:
                entry["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            _save_queue(queue, path)
            return True

    logger.warning("Queue entry not found for page_path: %s", page_path)
    return False


def list_pending(queue_path: Optional[Path] = None) -> list[dict]:
    """Return all entries with status='pending'."""
    path = Path(queue_path) if queue_path else QUEUE_PATH
    queue = _load_queue(path)
    return [e for e in queue if e["status"] == STATUS_PENDING]


def list_all(queue_path: Optional[Path] = None) -> list[dict]:
    """Return the full queue."""
    path = Path(queue_path) if queue_path else QUEUE_PATH
    return _load_queue(path)


def summary(queue_path: Optional[Path] = None) -> dict:
    """Return a count summary by status."""
    path = Path(queue_path) if queue_path else QUEUE_PATH
    queue = _load_queue(path)
    from collections import Counter
    counts = Counter(e["status"] for e in queue)
    return {
        "total":     len(queue),
        "pending":   counts.get(STATUS_PENDING, 0),
        "in_review": counts.get(STATUS_IN_REVIEW, 0),
        "complete":  counts.get(STATUS_COMPLETE, 0),
        "degraded":  counts.get(STATUS_DEGRADED, 0),
    }
