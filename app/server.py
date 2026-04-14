"""
NABAT-AI Digitize Server
========================
FastAPI backend that:
  • Serves index.html at /
  • Exposes POST /api/digitize — accepts an image, calls Claude Vision,
    returns structured multi-variant transcription JSON.

Run:
    cd app
    python server.py
    # or: uvicorn server:app --reload --port 8000

Then open  http://localhost:8000
"""

import base64
import json
import os
from pathlib import Path

import anthropic
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

# ── App ──────────────────────────────────────────────────────────────
app = FastAPI(title="NABAT-AI Digitize API", version="1.0")

HTML_FILE = Path(__file__).parent / "index.html"


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML_FILE.read_text(encoding="utf-8")


# ── Claude Vision HTR prompt ─────────────────────────────────────────
HTR_PROMPT = """\
You are Al-Nassikh (الناسخ) — the HTR engine for NABAT-AI, a system that \
digitises handwritten Khaleeji Nabati Arabic poetry manuscripts.

Examine the manuscript image carefully and perform these tasks:

1. **Manuscript Reading** — Transcribe ALL visible Arabic text EXACTLY as \
written, line by line, preserving original spelling (dialectal forms, archaic \
ligatures, dropped letters, Bedouin orthography).

2. **Standard MSA Reading** — Rewrite the same text using standard Modern \
Standard Arabic spelling: restore القاف where گ or ج appears in its place, \
normalise ة endings, restore الهمزة.

3. **Khaleeji Dialectal Reading** — Rewrite using Najdi/Khaleeji dialect \
conventions: قاف→گاف, ة→ه at line end, استعمال الگاف instead of القاف.

4. **Verse Structure** — Detect individual verses (بيت). Nabati poetry has \
two hemistiches: Sadr (صدر) on the right column and Ajuz (عجز) on the left. \
Split each verse accordingly.

Respond ONLY in this exact JSON (no markdown fences, no explanation):
{
  "has_arabic": true,
  "verse_count": <integer>,
  "manuscript_reading": "<exact transcription — lines separated by \\n>",
  "standard_msa": "<MSA normalised version — lines separated by \\n>",
  "khaleeji_dialectal": "<dialectal version — lines separated by \\n>",
  "verses": [
    {"num": 1, "sadr": "<first hemistich>", "ajuz": "<second hemistich or empty>"},
    {"num": 2, "sadr": "...", "ajuz": "..."}
  ],
  "confidence": "<HIGH|MEDIUM|LOW>",
  "confidence_note": "<one sentence explaining confidence level>",
  "image_quality": "<CLEAR|PARTIAL|DEGRADED>"
}

If the image has no readable Arabic text, return:
{
  "has_arabic": false,
  "verse_count": 0,
  "manuscript_reading": "",
  "standard_msa": "",
  "khaleeji_dialectal": "",
  "verses": [],
  "confidence": "LOW",
  "confidence_note": "No Arabic handwriting detected",
  "image_quality": "DEGRADED"
}
"""


# ── /api/digitize ────────────────────────────────────────────────────
@app.post("/api/digitize")
async def digitize(
    image: UploadFile = File(...),
    api_key: str = Form(default=""),
    ms_name: str = Form(default=""),
    collector: str = Form(default=""),
    century: str = Form(default=""),
    page_num: str = Form(default=""),
):
    # Resolve API key: form field → env var
    key = api_key.strip() or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise HTTPException(
            status_code=400,
            detail="No API key. Provide it in the form or set ANTHROPIC_API_KEY env var.",
        )

    # Read and encode image
    img_bytes = await image.read()
    if len(img_bytes) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large (max 20 MB)")

    media_type = image.content_type or "image/jpeg"
    # Normalise MIME type — Claude accepts image/jpeg, image/png, image/gif, image/webp
    if "jpeg" in media_type or "jpg" in media_type:
        media_type = "image/jpeg"
    elif "png" in media_type:
        media_type = "image/png"
    elif "webp" in media_type:
        media_type = "image/webp"
    elif "gif" in media_type:
        media_type = "image/gif"
    else:
        media_type = "image/jpeg"  # safe default

    img_b64 = base64.standard_b64encode(img_bytes).decode()

    # Call Claude Vision
    client = anthropic.Anthropic(api_key=key)
    try:
        msg = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": img_b64,
                            },
                        },
                        {"type": "text", "text": HTR_PROMPT},
                    ],
                }
            ],
        )
    except anthropic.AuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid API key.")
    except anthropic.APIStatusError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e.message))

    raw = msg.content[0].text.strip()

    # Strip any accidental markdown fences
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:])
        raw = raw.rsplit("```", 1)[0].strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # Claude returned prose — wrap it
        result = {
            "has_arabic": bool(raw),
            "verse_count": 0,
            "manuscript_reading": raw,
            "standard_msa": raw,
            "khaleeji_dialectal": raw,
            "verses": [],
            "confidence": "LOW",
            "confidence_note": "Unstructured output — manual review needed",
            "image_quality": "PARTIAL",
        }

    # Attach metadata
    result["source"]    = ms_name or "Uploaded Manuscript"
    result["collector"] = collector or "Unknown"
    result["century"]   = century or "—"
    result["page"]      = page_num or "—"

    return JSONResponse(result)


# ── Health check ─────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {"status": "ok", "model": "claude-opus-4-6"}


# ── Entry point ──────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"\n  NABAT-AI Digitize Server")
    print(f"  Open → http://localhost:{port}")
    print(f"  API key from env: {'✓ set' if os.environ.get('ANTHROPIC_API_KEY') else '✗ not set (enter in UI)'}\n")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
