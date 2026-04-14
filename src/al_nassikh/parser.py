"""
Al-Nassikh: PAGE-XML Parser
============================
Worker 1 — Step 6: Clean Output Layer

Parses eScriptorium PAGE-XML exports and produces structured JSON documents
following the architecture specification exactly (TASK_A_B_Plan.md §A.3 Step 6).

Output schema per poem:
{
  "poem_id": str,
  "poet": { "name": str, "region": str, "dialect_profile": {...} },
  "matla": { "standard": str, "dialectal": str, "manuscript": str },
  "source_volume": str,
  "source_page": int,
  "verse_count": int,
  "occasion": str | null,
  "meter": str | null,
  "rhyme_letter": str | null,
  "confidence_score": float,
  "evaluation_path": str,
  "stanzas": [ { "stanza_num", "sadr", "ajuz", ... } ],
  "source_image_path": str,
  "digitization_timestamp": str,
  "model_versions": {...}
}

For Phase 1 (body pages): lines are treated as individual Sadr/Ajuz pairs.
For Phase 4 (TOC pages): rows are parsed as anchor registry entries.
"""

import json
import os
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET


# ─────────────────────────────────────────────
# Arabic normalization helpers
# (CAMeL Tools v1.2.0 equivalent, per spec §B.2.3)
# ─────────────────────────────────────────────

ALEF_VARIANTS = "أإآٱ"
ALEF_NORMAL = "ا"
TAA_MARBUTA = "ة"
TAA_MARBUTA_NORMAL = "ه"

def normalize_arabic(text: str, dediacritize: bool = False) -> str:
    """
    Normalize Arabic text per CAMeL Tools pipeline:
    1. Alef normalization (variants → bare alef)
    2. Taa marbuta normalization
    3. Optional dediacritization
    """
    if not text:
        return text
    # 1. Alef normalization
    for v in ALEF_VARIANTS:
        text = text.replace(v, ALEF_NORMAL)
    # 2. Taa marbuta normalization (for embedding consistency)
    text = text.replace(TAA_MARBUTA, TAA_MARBUTA_NORMAL)
    # 3. Dediacritize: remove harakat (U+064B–U+065F range)
    if dediacritize:
        text = re.sub(r'[\u064B-\u065F]', '', text)
    # 4. Strip tatweel (kashida)
    text = text.replace('\u0640', '')
    # 5. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ─────────────────────────────────────────────
# Geographic Dialect Atlas
# (Architecture spec §A.3 Layer A, Table)
# ─────────────────────────────────────────────

DIALECT_ATLAS = {
    "najd": {
        "qaaf": "gaaf",    # ق → ج/گ
        "jiim": "standard",
        "kaaf": "standard",
        "example_poets": ["حمود العبيد", "عبيد بن رشيد", "مرخان راعي الجوف"]
    },
    "uae_coast": {
        "qaaf": "gaaf",
        "jiim": "yaa",    # ج → ي in some words
        "kaaf": "ch",     # ك → تش
        "example_poets": ["سالم الشليحي"]
    },
    "oman": {
        "qaaf": "gaaf_or_q",
        "jiim": "standard",
        "kaaf": "standard",
        "example_poets": []
    },
    "al_jouf": {
        "qaaf": "gaaf",
        "jiim": "standard",
        "kaaf": "standard",
        "example_poets": ["مرخان راعي الجوف"]
    },
    "unknown": {
        "qaaf": "unknown",
        "jiim": "unknown",
        "kaaf": "unknown",
        "example_poets": []
    }
}

def infer_dialect_region(poet_name: str) -> str:
    """Infer dialect region from poet name using the atlas."""
    for region, info in DIALECT_ATLAS.items():
        if any(p in poet_name for p in info.get("example_poets", [])):
            return region
    return "unknown"

def get_dialect_profile(region: str) -> dict:
    atlas = DIALECT_ATLAS.get(region, DIALECT_ATLAS["unknown"])
    return {
        "qaaf": atlas["qaaf"],
        "jiim": atlas["jiim"],
        "kaaf": atlas["kaaf"]
    }


# ─────────────────────────────────────────────
# Multi-Variant Transcription Gateway
# (Architecture spec §A.3 Layer A)
# ─────────────────────────────────────────────

QAAF_GAAF_MAP = {
    "gaaf":     [("ج", "ق")],   # ج in manuscript → ق in standard
    "standard": [],
    "unknown":  []
}

def produce_variants(manuscript_text: str, dialect_region: str) -> dict:
    """
    Produce three transcription variants per architecture spec:
      - manuscript_reading: OCR text as-is from eScriptorium
      - standard_reading:   Normalized MSA spelling
      - dialectal_reading:  Poet's regional pronunciation preserved
    Returns dict with ambiguous_words list.
    """
    standard = manuscript_text
    dialectal = manuscript_text
    ambiguous = []

    profile = DIALECT_ATLAS.get(dialect_region, DIALECT_ATLAS["unknown"])
    qaaf_treatment = profile["qaaf"]

    if qaaf_treatment == "gaaf":
        # In Najd/Gulf manuscripts, ج often represents the dialectal /g/ sound
        # which corresponds to ق in MSA → standard reading restores ق
        # We look for standalone ج words that are common dialectal qaaf substitutes
        words = manuscript_text.split()
        std_words = list(words)
        dial_words = list(words)

        for i, word in enumerate(words):
            # Heuristic: if word starts with ج and the root makes more sense with ق
            # This is a simplified atlas lookup — production would use full morphological analysis
            qaaf_candidate = word.replace('ج', 'ق')
            if qaaf_candidate != word and len(word) > 1:
                ambiguous.append({
                    "position": i,
                    "manuscript_form": word,
                    "variants": [word, qaaf_candidate],
                    "selected_standard": qaaf_candidate,
                    "selected_dialectal": word,
                    "rationale": f"Poet is from {dialect_region}; qaaf→gaaf substitution per dialect atlas"
                })
                std_words[i] = qaaf_candidate
                dial_words[i] = word  # preserve dialectal form

        standard = " ".join(std_words)
        dialectal = " ".join(dial_words)

    return {
        "manuscript_reading": manuscript_text,
        "standard_reading": standard,
        "dialectal_reading": dialectal,
        "ambiguous_words": ambiguous
    }


# ─────────────────────────────────────────────
# PAGE-XML Parser (eScriptorium PAGE-2019 format)
# ─────────────────────────────────────────────

PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"

def parse_pagexml(xml_path: str) -> dict:
    """
    Parse a single PAGE-XML file from eScriptorium.
    Returns dict with:
      - image_filename, image_width, image_height
      - lines: list of {line_id, text, baseline, coords, custom_type}
      - created, last_change
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    ns = {"p": PAGE_NS}

    metadata = root.find("p:Metadata", ns)
    created = metadata.findtext("p:Created", default="", namespaces=ns) if metadata else ""
    last_change = metadata.findtext("p:LastChange", default="", namespaces=ns) if metadata else ""

    page_el = root.find("p:Page", ns)
    if page_el is None:
        return {}

    image_filename = page_el.get("imageFilename", "")
    image_width = int(page_el.get("imageWidth", 0))
    image_height = int(page_el.get("imageHeight", 0))

    lines = []
    for region in page_el.findall(".//p:TextRegion", ns):
        for line in region.findall("p:TextLine", ns):
            line_id = line.get("id", "")
            custom = line.get("custom", "")

            coords_el = line.find("p:Coords", ns)
            coords_pts = coords_el.get("points", "") if coords_el is not None else ""
            coords_list = _parse_points(coords_pts)
            bbox = _points_to_bbox(coords_list)

            baseline_el = line.find("p:Baseline", ns)
            baseline_pts = baseline_el.get("points", "") if baseline_el is not None else ""

            text_equiv = line.find("p:TextEquiv/p:Unicode", ns)
            text = text_equiv.text.strip() if (text_equiv is not None and text_equiv.text) else ""

            # Extract structure type from custom attribute
            struct_match = re.search(r'structure\s*\{type:(\w+);?\}', custom)
            struct_type = struct_match.group(1) if struct_match else "default"

            lines.append({
                "line_id": line_id,
                "text": text,
                "baseline": baseline_pts,
                "coords": coords_pts,
                "bbox": bbox,
                "struct_type": struct_type,
                "custom": custom
            })

    return {
        "image_filename": image_filename,
        "image_width": image_width,
        "image_height": image_height,
        "lines": lines,
        "created": created,
        "last_change": last_change
    }


def _parse_points(pts_str: str) -> list:
    """Parse 'x1,y1 x2,y2 ...' into list of (x, y) tuples."""
    pts = []
    for pair in pts_str.strip().split():
        try:
            x, y = pair.split(",")
            pts.append((int(x), int(y)))
        except Exception:
            pass
    return pts

def _points_to_bbox(pts: list) -> list:
    """Convert polygon points to [x_min, y_min, x_max, y_max]."""
    if not pts:
        return [0, 0, 0, 0]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)]


# ─────────────────────────────────────────────
# TOC Row Parser (Phase 4 pages)
# ─────────────────────────────────────────────

def parse_toc_lines(lines: list) -> list:
    """
    Parse TOC page lines into anchor registry entries.
    TOC format (per architecture spec §A.3 Step 1):
      Poet Name | Page/Verse | First Line (Matla)
    or: Poet | Page | First Line | Occasion

    Returns list of anchor dicts.
    """
    anchors = []
    header_keywords = {"اسم الشاعر", "عدد الصحيفة", "اول شطر", "أول شطر",
                       "مطلع", "المناسبة", "ص", "ع", "رقم", "الشاعر"}

    for line in lines:
        text = line["text"].strip()
        if not text:
            continue

        # Skip pure header lines
        if any(kw in text for kw in header_keywords) and "|" in text:
            parts = [p.strip() for p in text.split("|")]
            if all(any(kw in p for kw in header_keywords) for p in parts if p):
                continue

        # Parse pipe-separated TOC rows
        if "|" in text:
            parts = [p.strip() for p in text.split("|")]
            anchor = _build_anchor_from_parts(parts, line["bbox"])
            if anchor:
                anchors.append(anchor)
        # Also handle lines that are just a number (page ref continuation)
        elif re.match(r'^\d+$', text):
            pass  # standalone page numbers — skip for now
        else:
            # Could be a poet name on its own line (some TOC formats)
            pass

    return anchors

def _build_anchor_from_parts(parts: list, bbox: list) -> dict:
    """
    Try to extract poet, page_num, matla, occasion from pipe-separated parts.
    Handles variable column counts across different manuscripts.
    """
    if len(parts) < 2:
        return None

    # Heuristic: find the part that looks like a page/verse number
    page_num = None
    verse_count = None
    poet = None
    matla = None
    occasion = None

    num_pattern = re.compile(r'^\d+$')

    for p in parts:
        if num_pattern.match(p):
            if page_num is None:
                page_num = int(p)
            elif verse_count is None:
                verse_count = int(p)
        elif matla is None and len(p) > 10:
            # Longest non-numeric part is likely the Matla (first line of poem)
            matla = p
        elif poet is None and len(p) > 1:
            poet = p

    # For 3-part rows: [poet, page, matla] or [poet, matla, page]
    if len(parts) == 3:
        # Pattern: name | page | matla  OR  name | matla | page
        if num_pattern.match(parts[1]):
            poet = parts[0]
            page_num = int(parts[1])
            matla = parts[2]
        elif num_pattern.match(parts[2]):
            poet = parts[0]
            matla = parts[1]
            page_num = int(parts[2])
        else:
            poet = parts[0]
            matla = parts[1]

    # For 4-part rows: poet | page | matla | occasion
    elif len(parts) == 4:
        poet = parts[0]
        if num_pattern.match(parts[1]):
            page_num = int(parts[1])
            matla = parts[2]
            occasion = parts[3] if parts[3] else None
        elif num_pattern.match(parts[2]):
            poet = parts[0]
            matla = parts[1]
            page_num = int(parts[2])
            occasion = parts[3] if parts[3] else None

    # Fallback for 2-part rows
    elif len(parts) == 2:
        if num_pattern.match(parts[0]):
            page_num = int(parts[0])
            matla = parts[1]
        elif num_pattern.match(parts[1]):
            matla = parts[0]
            page_num = int(parts[1])
        else:
            poet = parts[0]
            matla = parts[1]

    if not matla:
        return None

    return {
        "poet_name": poet or "unknown",
        "page_number": page_num,
        "verse_count": verse_count,
        "matla_text": matla,
        "occasion": occasion,
        "bbox": bbox
    }


# ─────────────────────────────────────────────
# Main Document Builder
# (Produces architecture spec §A.3 Step 6 JSON)
# ─────────────────────────────────────────────

def build_poem_json(
    lines: list,
    source_volume: str,
    source_page: int,
    image_filename: str,
    poet_name: str = "unknown",
    dialect_region: str = "unknown",
    occasion: str = None,
    confidence_score: float = 0.95,
    evaluation_path: str = "escriptorium_hitl_corrected",
    created_ts: str = None
) -> dict:
    """
    Build a structured poem JSON document from a list of PAGE-XML lines.
    Following architecture spec §A.3 Step 6 output schema exactly.
    """
    region = dialect_region if dialect_region != "unknown" else infer_dialect_region(poet_name)
    dialect_profile = get_dialect_profile(region)

    # Build stanzas: each line is treated as a full verse (بيت)
    # In single-column manuscripts, each line is one hemistich
    # We pair consecutive lines as Sadr + Ajuz when the manuscript is single-column
    stanzas = _build_stanzas(lines, region)

    # Matla = first verse's text (standard reading of Sadr)
    matla_text = ""
    matla_std = ""
    matla_dial = ""
    if stanzas:
        first = stanzas[0]
        matla_text = first.get("full_verse_manuscript", first.get("sadr", {}).get("manuscript_reading", ""))
        matla_std = first.get("sadr", {}).get("standard_reading", matla_text)
        matla_dial = first.get("sadr", {}).get("dialectal_reading", matla_text)

    poem_id = f"{source_volume}_p{source_page:04d}"

    return {
        "poem_id": poem_id,
        "poet": {
            "name": poet_name,
            "region": region,
            "dialect_profile": dialect_profile
        },
        "matla": {
            "standard": normalize_arabic(matla_std),
            "dialectal": normalize_arabic(matla_dial),
            "manuscript": matla_text
        },
        "source_volume": source_volume,
        "source_page": source_page,
        "verse_count": len(stanzas),
        "occasion": occasion,
        "meter": None,  # Populated by meter detection module (future: BASRAH/AraPoemBERT)
        "rhyme_letter": None,  # Populated by rhyme extraction module
        "confidence_score": confidence_score,
        "evaluation_path": evaluation_path,
        "stanzas": stanzas,
        "source_image_path": image_filename,
        "digitization_timestamp": created_ts or datetime.now(timezone.utc).isoformat(),
        "model_versions": {
            "segmentation": "kraken-v5-blla (via eScriptorium)",
            "primary_hwr": "escriptorium-human-corrected",
            "structural_vlm": "n/a (Phase 1 MVP — HITL path)",
            "document_model": "n/a (Phase 1 MVP)",
            "verification_htr": "n/a (Phase 1 MVP)"
        }
    }


def _build_stanzas(lines: list, dialect_region: str) -> list:
    """
    Convert PAGE-XML lines into stanza objects.
    Architecture spec: Level 1 chunk = Sadr + Ajuz (one بيت).
    For eScriptorium exports where each line is already a verse,
    we treat each line as a complete verse with Sadr=line and Ajuz inferred
    from the // separator if present, or as a standalone hemistich.
    """
    stanzas = []
    stanza_num = 1

    # Filter out empty lines
    content_lines = [l for l in lines if l.get("text", "").strip()]

    for line in content_lines:
        text = line["text"].strip()
        bbox = line.get("bbox", [0, 0, 0, 0])

        # Check for Sadr // Ajuz separator
        if "//" in text:
            parts = text.split("//", 1)
            sadr_text = parts[0].strip()
            ajuz_text = parts[1].strip()
        else:
            # Single hemistich — treat as Sadr, Ajuz unknown
            sadr_text = text
            ajuz_text = ""

        sadr_variants = produce_variants(sadr_text, dialect_region)
        ajuz_variants = produce_variants(ajuz_text, dialect_region) if ajuz_text else {
            "manuscript_reading": "",
            "standard_reading": "",
            "dialectal_reading": "",
            "ambiguous_words": []
        }

        stanzas.append({
            "stanza_num": stanza_num,
            "full_verse_manuscript": text,
            "sadr": sadr_variants,
            "ajuz": ajuz_variants,
            "meter_valid": None,   # Populated by meter verification layer
            "rhyme_valid": None,   # Populated by rhyme verification layer
            "model_consensus": "HITL_corrected",  # This path = eScriptorium human correction
            "source_crop_bbox": bbox
        })
        stanza_num += 1

    return stanzas


# ─────────────────────────────────────────────
# Export Directory Processor
# ─────────────────────────────────────────────

def parse_filename_metadata(xml_filename: str) -> tuple:
    """
    Extract volume and page number from filename.
    e.g. 'manuscript22_p5.xml' → ('manuscript22', 5)
         '601-782_p178.xml'   → ('601-782', 178)
    """
    name = Path(xml_filename).stem
    match = re.match(r'^(.+?)_p(\d+)$', name)
    if match:
        volume = match.group(1)
        page = int(match.group(2))
        return volume, page
    return name, 0


def process_export_directory(export_dir: str, phase: str = "phase_1") -> dict:
    """
    Process all PAGE-XML files in an eScriptorium export directory.
    Returns:
      - phase 1 (body pages): dict of poem_id → poem_json
      - phase 4 (TOC pages): dict of volume → list of anchor entries
    """
    export_path = Path(export_dir)
    xml_files = sorted(export_path.glob("*.xml"))

    # Skip METS.xml (manifest file)
    xml_files = [f for f in xml_files if f.name.upper() != "METS.XML"]

    results = {}

    for xml_file in xml_files:
        print(f"  Parsing: {xml_file.name}")
        parsed = parse_pagexml(str(xml_file))
        if not parsed or not parsed.get("lines"):
            print(f"    [SKIP] No lines found.")
            continue

        volume, page = parse_filename_metadata(xml_file.name)

        if "phase_4" in phase or "toc" in phase.lower():
            # TOC parsing → anchor registry
            anchors = parse_toc_lines(parsed["lines"])
            key = f"{volume}_p{page}"
            results[key] = {
                "type": "toc_page",
                "volume": volume,
                "page": page,
                "image_filename": parsed["image_filename"],
                "anchors": anchors,
                "raw_line_count": len(parsed["lines"])
            }
            print(f"    → TOC: {len(anchors)} anchor entries extracted")
        else:
            # Body page → poem JSON
            poem = build_poem_json(
                lines=parsed["lines"],
                source_volume=volume,
                source_page=page,
                image_filename=parsed["image_filename"],
                confidence_score=0.95,  # HITL-corrected = high confidence
                evaluation_path="escriptorium_hitl_corrected",
                created_ts=parsed.get("created", "")
            )
            results[poem["poem_id"]] = poem
            print(f"    → Poem: {len(poem['stanzas'])} verses, poet='{poem['poet']['name']}'")

    return results


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).parent.parent.parent

    BASE = PROJECT_ROOT / "manuscripts" / "Ground_Truth_Exports"
    PHASE1_DIR = BASE / "export_doc5_phase_1_pagexml_20260414140737"
    PHASE4_DIR = BASE / "export_doc9_phase_4_pagexml_20260414140639"

    OUTPUT_DIR = PROJECT_ROOT / "data" / "ground_truth"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Al-Nassikh Parser — Phase 1 (Body Pages)")
    print("=" * 60)
    phase1_results = process_export_directory(str(PHASE1_DIR), phase="phase_1")
    phase1_out = OUTPUT_DIR / "phase1_poems.json"
    with open(phase1_out, "w", encoding="utf-8") as f:
        json.dump(phase1_results, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Phase 1: {len(phase1_results)} poem documents → {phase1_out}")

    print()
    print("=" * 60)
    print("Al-Nassikh Parser — Phase 4 (TOC / Anchor Registry)")
    print("=" * 60)
    phase4_results = process_export_directory(str(PHASE4_DIR), phase="phase_4_toc")
    phase4_out = OUTPUT_DIR / "phase4_toc_anchors.json"
    with open(phase4_out, "w", encoding="utf-8") as f:
        json.dump(phase4_results, f, ensure_ascii=False, indent=2)

    # Flatten anchors for easy lookup
    all_anchors = []
    for page_key, page_data in phase4_results.items():
        for anchor in page_data.get("anchors", []):
            anchor["source_volume"] = page_data["volume"]
            anchor["source_toc_page"] = page_data["page"]
            all_anchors.append(anchor)

    anchor_registry_out = OUTPUT_DIR / "anchor_registry.json"
    with open(anchor_registry_out, "w", encoding="utf-8") as f:
        json.dump(all_anchors, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Phase 4: {len(phase4_results)} TOC pages → {len(all_anchors)} anchors → {anchor_registry_out}")

    print("\nDone.")
