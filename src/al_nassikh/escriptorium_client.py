"""
src/al_nassikh/escriptorium_client.py
=======================================
Why this file exists: §M2b — eScriptorium is our self-hosted transcription
platform. The Operator Console (Tab A, M9) needs to drive seven operations
without the archivist having to open a separate browser tab:
  1. ensure_project      — create or find a project for a manuscript
  2. create_document     — one document per manuscript upload
  3. upload_pages        — bulk image upload
  4. run_kraken_segmentation — trigger BLLA baseline detection
  5. set_ontology        — pre-seed 5 Khaleeji zone labels
  6. export_pagexml      — pull back the annotated PAGE-XML
  7. embed_editor_iframe_url — URL for the Streamlit iframe

All calls go through the `escriptorium-connector` PyPI package
(https://pypi.org/project/escriptorium-connector/), which wraps the
eScriptorium REST API at `{base_url}/api/`. This file is a thin veneer that:
  - Reads config from env vars (ESCR_BASE_URL, ESCR_API_TOKEN)
  - Falls back gracefully if eScriptorium is unreachable
  - Exposes the exact verbs the Streamlit tab needs, nothing else

Architecture ref: §1.6.3 (Platform), §M2b, §3.1 Tab A.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

ESCR_BASE_URL  = os.getenv("ESCR_BASE_URL",  "http://localhost:8080")
ESCR_API_TOKEN = os.getenv("ESCR_API_TOKEN", "")

SEGMENTATION_TIMEOUT_S = 300   # seconds to wait for Kraken job
POLL_INTERVAL_S        = 5.0

# The five Khaleeji zone labels (§2.3 Stage 5)
KHALEEJI_ZONE_TYPES = [
    "TWO_COLUMN_POETRY",
    "PROSE_ATTRIBUTION",
    "ORNAMENTAL_DIVIDER",
    "MARGIN_PERPENDICULAR",
    "INTERLINEAR_ANNOTATION",
]


# ── Connectivity check ────────────────────────────────────────────────────────

class EScriptoriumUnavailableError(Exception):
    """Raised when eScriptorium cannot be reached."""
    pass


def _get_connector():
    """
    Lazy-load and return an EScriptoriumConnector instance.
    Raises EScriptoriumUnavailableError if the server is unreachable or the
    package is not installed.
    """
    if not ESCR_API_TOKEN:
        raise EScriptoriumUnavailableError(
            "ESCR_API_TOKEN is not set. "
            "Get a token with: docker compose exec app python manage.py drf_create_token admin"
        )
    try:
        from escriptorium_connector import EScriptoriumConnector
    except ImportError:
        raise EScriptoriumUnavailableError(
            "escriptorium-connector not installed. Run: pip install escriptorium-connector"
        )
    try:
        conn = EScriptoriumConnector(
            url=ESCR_BASE_URL,
            api_key=ESCR_API_TOKEN,
        )
        return conn
    except Exception as exc:
        raise EScriptoriumUnavailableError(
            f"Cannot connect to eScriptorium at {ESCR_BASE_URL}: {exc}. "
            "Start eScriptorium: cd infra/escriptorium && docker compose up -d"
        ) from exc


def is_available() -> bool:
    """Return True if eScriptorium is reachable (used by Streamlit to show/hide UI)."""
    try:
        _get_connector()
        return True
    except EScriptoriumUnavailableError:
        return False


# ── Project management ────────────────────────────────────────────────────────

def ensure_project(name: str) -> dict:
    """
    Idempotently create or retrieve an eScriptorium project by name.

    Returns:
        {"project_id": int, "name": str, "created": bool}
    """
    conn = _get_connector()
    projects = conn.get_projects().results
    for p in projects:
        if p.name == name:
            logger.info("Found existing project: %s (id=%s)", name, p.pk)
            return {"project_id": p.pk, "name": name, "created": False}

    # Create new
    project = conn.create_project(name=name)
    logger.info("Created project: %s (id=%s)", name, project.pk)
    return {"project_id": project.pk, "name": name, "created": True}


# ── Document management ───────────────────────────────────────────────────────

def create_document(
    project_id: int,
    name: str,
    metadata: Optional[dict] = None,
) -> dict:
    """
    Create a new eScriptorium document (one document = one manuscript upload).

    Returns:
        {"document_id": int, "name": str, "project_id": int}
    """
    conn = _get_connector()
    doc = conn.create_document(
        name=name,
        project=project_id,
        metadata=metadata or {},
        main_script="Arabic",
        read_direction="rtl",
    )
    logger.info("Created document: %s (id=%s) in project %s", name, doc.pk, project_id)
    return {"document_id": doc.pk, "name": name, "project_id": project_id}


# ── Page upload ───────────────────────────────────────────────────────────────

def upload_pages(
    document_id: int,
    image_paths: list[Path],
) -> dict:
    """
    Bulk upload page images to an eScriptorium document.

    Args:
        document_id:  eScriptorium document ID.
        image_paths:  List of local image file paths (PNG or JPEG).

    Returns:
        {"document_id": int, "uploaded": int, "errors": list[str]}
    """
    conn = _get_connector()
    uploaded = 0
    errors: list[str] = []

    for img_path in image_paths:
        img_path = Path(img_path)
        if not img_path.exists():
            errors.append(f"File not found: {img_path}")
            continue
        try:
            conn.upload_part_image(document_id, str(img_path))
            uploaded += 1
            logger.debug("Uploaded: %s", img_path.name)
        except Exception as exc:
            errors.append(f"{img_path.name}: {exc}")
            logger.warning("Upload failed for %s: %s", img_path.name, exc)

    logger.info("Uploaded %d/%d pages to document %d", uploaded, len(image_paths), document_id)
    return {"document_id": document_id, "uploaded": uploaded, "errors": errors}


# ── Kraken segmentation ───────────────────────────────────────────────────────

def run_kraken_segmentation(
    document_id: int,
    model: str = "blla",
    timeout_s: int = SEGMENTATION_TIMEOUT_S,
) -> dict:
    """
    Trigger Kraken BLLA segmentation on a document and block until complete.

    Args:
        document_id: eScriptorium document ID.
        model:       Segmentation model identifier ("blla").
        timeout_s:   Maximum seconds to wait.

    Returns:
        {"status": "completed"|"timeout"|"error", "elapsed_s": float}

    Raises:
        TimeoutError if the job does not complete within timeout_s.
        EScriptoriumUnavailableError if the server is unreachable.
    """
    conn = _get_connector()
    start = time.time()

    logger.info("Triggering Kraken segmentation on document %d (model=%s)…", document_id, model)
    conn.segment_document(document_id, model=model)

    # Poll for completion
    while True:
        elapsed = time.time() - start
        if elapsed > timeout_s:
            raise TimeoutError(
                f"Kraken segmentation timed out after {elapsed:.0f}s for document {document_id}"
            )
        try:
            doc = conn.get_document(document_id)
            # eScriptorium sets parts_count when segmentation is done
            if hasattr(doc, "parts_count") and doc.parts_count > 0:
                logger.info(
                    "Segmentation completed for document %d in %.1fs", document_id, elapsed
                )
                return {"status": "completed", "elapsed_s": round(elapsed, 2)}
        except Exception:
            pass   # transient — keep polling
        time.sleep(POLL_INTERVAL_S)


# ── Zone ontology ─────────────────────────────────────────────────────────────

def set_ontology(
    document_id: int,
    zone_types: Optional[list[str]] = None,
) -> dict:
    """
    Pre-seed the five Khaleeji zone labels in the eScriptorium document ontology.
    This makes them available as named options in the annotator's zone-label UI.

    Returns:
        {"document_id": int, "zone_types": list[str], "status": "ok"|"error"}
    """
    conn = _get_connector()
    types = zone_types or KHALEEJI_ZONE_TYPES

    try:
        for zone_type in types:
            conn.create_annotation_taxonomy(
                document_id=document_id,
                name=zone_type,
                annotation_type="region",
            )
        logger.info("Set %d zone types on document %d", len(types), document_id)
        return {"document_id": document_id, "zone_types": types, "status": "ok"}
    except Exception as exc:
        logger.warning("set_ontology failed: %s", exc)
        return {"document_id": document_id, "zone_types": types, "status": f"error: {exc}"}


# ── PAGE-XML export ───────────────────────────────────────────────────────────

def export_pagexml(
    document_id: int,
    output_dir: Optional[Path] = None,
    fmt: str = "pagexml_alto",
) -> dict:
    """
    Export the annotated PAGE-XML for a document and write it to output_dir.

    Args:
        document_id: eScriptorium document ID.
        output_dir:  Where to write the XML file. Defaults to
                     manuscripts/Ground_Truth_Exports/.
        fmt:         Export format ("pagexml_alto" or "alto").

    Returns:
        {"document_id": int, "output_path": str, "status": "ok"|"error"}
    """
    conn = _get_connector()

    if output_dir is None:
        # Walk up to repo root
        here = Path(__file__).resolve().parent
        for candidate in [here, here.parent, here.parent.parent]:
            if (candidate / "requirements.txt").exists():
                output_dir = candidate / "manuscripts" / "Ground_Truth_Exports"
                break
        else:
            output_dir = Path("manuscripts") / "Ground_Truth_Exports"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        xml_content = conn.download_document_export(document_id, export_format=fmt)
        out_path = output_dir / f"document_{document_id}.xml"
        out_path.write_bytes(xml_content if isinstance(xml_content, bytes)
                             else xml_content.encode("utf-8"))
        logger.info("Exported PAGE-XML for document %d → %s", document_id, out_path)
        return {"document_id": document_id, "output_path": str(out_path), "status": "ok"}
    except Exception as exc:
        logger.error("export_pagexml failed for document %d: %s", document_id, exc)
        return {"document_id": document_id, "output_path": None, "status": f"error: {exc}"}


# ── Editor iframe URL ─────────────────────────────────────────────────────────

def embed_editor_iframe_url(document_id: int, part_pk: Optional[int] = None) -> str:
    """
    Return the eScriptorium editor URL for embedding in a Streamlit iframe.
    This URL opens the document directly in eScriptorium's native annotation editor
    — where the annotator does the actual zone + baseline work.

    Args:
        document_id: eScriptorium document ID.
        part_pk:     Optional specific page (part) primary key to open directly.

    Returns:
        URL string suitable for st.components.v1.iframe(url, height=800).
    """
    if part_pk:
        return f"{ESCR_BASE_URL}/document/{document_id}/part/{part_pk}/edit/"
    return f"{ESCR_BASE_URL}/document/{document_id}/edit/"
