"""
src/al_nassikh/ingest/kraken_draft.py
=======================================
Why this file exists: §M2 item 9 — Kraken v5 BLLA segmentation runs inside
eScriptorium's Celery worker, not as a local binary. This wrapper calls the
eScriptorium REST API endpoint `POST /api/documents/{id}/segment/` to trigger
segmentation, then polls until the job completes or times out. The result is
a PAGE-XML document that the operator can then review and correct in
eScriptorium's native editor.

Why eScriptorium-hosted Kraken (not local binary): the architecture doc §5
says the system must run on any laptop without GPU. eScriptorium uses the CPU
Kraken with the BLLA baseline model. Running it locally would require installing
the full Kraken CLI and its CUDA-optional torch stack — a portability risk.
eScriptorium's Celery worker handles Kraken in its own container.

This module is a thin coordinator:
  - It calls escriptorium_client.run_kraken_segmentation() (which is the real
    API call; logic lives there).
  - It adds a structured result with job metadata.

Architecture ref: §2.3 Stage 6 (Corrected Segmentation), §M2 item 9.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

SEGMENTATION_TIMEOUT_S = 300   # §M2b says block up to 300 s
POLL_INTERVAL_S        = 5.0


def run_segmentation(
    document_id: int,
    model: str = "blla",
    escr_client=None,
    timeout_s: int = SEGMENTATION_TIMEOUT_S,
) -> dict:
    """
    Trigger Kraken BLLA segmentation on a document already uploaded to eScriptorium.

    Args:
        document_id: eScriptorium document ID (integer from create_document).
        model:       Segmentation model name — "blla" for Kraken v5 BLLA.
        escr_client: An escriptorium_client module/object (injected for testability;
                     defaults to importing al_nassikh.escriptorium_client).
        timeout_s:   Maximum seconds to wait for the job.

    Returns:
        {
          "document_id": int,
          "model":       str,
          "status":      "completed" | "timeout" | "error" | "unavailable",
          "elapsed_s":   float,
          "message":     str,
        }
    """
    if escr_client is None:
        try:
            from al_nassikh import escriptorium_client as escr_client
        except ImportError:
            logger.warning("escriptorium_client not available — returning stub result")
            return {
                "document_id": document_id,
                "model":       model,
                "status":      "unavailable",
                "elapsed_s":   0.0,
                "message":     (
                    "eScriptorium client not available. "
                    "Start eScriptorium with: cd infra/escriptorium && docker compose up -d"
                ),
            }

    start = time.time()

    try:
        result = escr_client.run_kraken_segmentation(document_id, model=model)
        elapsed = time.time() - start
        logger.info(
            "Kraken segmentation completed for document %d in %.1f s",
            document_id, elapsed
        )
        return {
            "document_id": document_id,
            "model":       model,
            "status":      "completed",
            "elapsed_s":   round(elapsed, 2),
            "message":     f"Segmentation completed in {elapsed:.1f}s.",
        }
    except TimeoutError:
        elapsed = time.time() - start
        logger.warning("Kraken segmentation timed out after %.1f s", elapsed)
        return {
            "document_id": document_id,
            "model":       model,
            "status":      "timeout",
            "elapsed_s":   round(elapsed, 2),
            "message":     f"Segmentation timed out after {elapsed:.1f}s ({timeout_s}s limit).",
        }
    except Exception as exc:
        elapsed = time.time() - start
        logger.error("Kraken segmentation error: %s", exc)
        return {
            "document_id": document_id,
            "model":       model,
            "status":      "error",
            "elapsed_s":   round(elapsed, 2),
            "message":     f"Error: {exc}",
        }
