"""
scripts/clean_registry.py
=========================
Why this exists: the Phase-4 anchor registry has several OCR/parsing artifacts
where poet names and matla verses got confused, page numbers leaked into name
fields, and pipe-separated metadata wasn't split. This script applies a
deterministic cleaning methodology and writes a cleaned registry file.

CLEANING METHODOLOGY
--------------------
Pattern 1 — Swapped fields (most impactful, ~19 records)
  Symptom:  matla_text matches "name | page_number" (Arabic/Latin digits after pipe)
            poet_name is a verse (long Arabic text with verb structure)
  Fix:      Swap the two fields, extract page from the | suffix

Pattern 2 — Page number in poet_name via pipe (1 record confirmed)
  Symptom:  poet_name = "Real Name | ٦٤٩"
  Fix:      Split on |, keep left as poet_name, use right as page_number if null

Pattern 3 — Pure Arabic/Latin digit in poet_name (66 records)
  Symptom:  poet_name = "٤٩٠" or "٦٦٩" (only numerals — leaked page number)
  Strategy: The digit IS the page number. Carry it to page_number if page_number is null.
            Set poet_name = "unknown" since we can't recover the name from this record alone.
            Flag with tag: "poet_page_leaked"

Pattern 4 — Manuscript header in poet_name (6 records)
  Symptom:  poet_name = "مخطوطة هوبير / ١" etc. (section divider rows)
  Fix:      Set poet_name = "unknown", flag as "header_row", matla stays (often empty)

Pattern 5 — Digit-only matla (23 records from manuscript18/22 — likely OCR noise)
  Symptom:  matla_text = "٥٥١" or "1" etc.
  Fix:      Set matla_text = "" (empty), flag as "matla_ocr_noise"

Pattern 6 — Pipe in matla (non-swap cases, ~38 records)
  Symptom:  matla_text = "verse text | attribution note"  (e.g. "ياراكب | صالح السكيني")
            poet_name is already a valid name
  Fix:      Split matla on |; keep left as matla_text, move right to
            new field "matla_attribution_note"
"""

import json
import re
import sys
from pathlib import Path
from copy import deepcopy

REPO = Path(__file__).resolve().parent.parent
IN_PATH  = REPO / "data" / "ground_truth" / "anchor_registry_phase4_enriched.json"
OUT_PATH = REPO / "data" / "ground_truth" / "anchor_registry_phase4_cleaned.json"
AUDIT_PATH = REPO / "data" / "ground_truth" / "cleaning_audit.json"

# ── Helpers ────────────────────────────────────────────────────────────────────

AR_DIGIT_RE = re.compile(r'[\u0660-\u0669\u06F0-\u06F9]')
PURE_NUM_RE = re.compile(r'^[\u0660-\u0669\u06F0-\u06F9\d\s\.]+$')
PIPE_PAGE_RE = re.compile(r'^(.+?)\s*\|\s*([\u0660-\u0669\u06F0-\u06F9\d]+)\s*$')
SWAP_DETECT_RE = re.compile(r'^(.{4,50}?)\s*\|\s*([\u0660-\u0669\u06F0-\u06F9\d]{1,6})\s*$')
MS_HEADER_RE = re.compile(r'مخطوطة|هوبير.*\/|مخطوط')

# Verse openers: يقول / يا + space / specific verb forms
VERSE_OPENER_RE = re.compile(r'^(يقول|يا |قال |قالت |أقول |نقول )')

# Name connectors: بن / ابن / ابو / أبو / آل — strong signal this is a person name
NAME_CONNECTOR_RE = re.compile(r'\bبن\b|\bابن\b|\bابو\b|\bأبو\b|\bآل\b|\bبنت\b|\bأم\b')

# Known name-only words that are never verse openers
KNOWN_NAME_STARTS = re.compile(r'^(عبد|محمد|حمد|سالم|ناصر|خالد|سعد|راشد|مبارك|صالح|فهد|جابر|طلال|زايد)')


def ar_to_latin(s: str) -> str:
    """Convert Arabic-Indic digits to Latin digits."""
    table = str.maketrans('٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹', '01234567890123456789')
    return s.translate(table)


def looks_like_name(text: str) -> bool:
    """
    Return True when text is likely a person's name.
    Why multiple signals: single-character checks fail on 'عبدالله' which starts
    with 'ع' — a common Arabic letter — so we rely on:
      1. Short enough to be a name (≤ 8 words)
      2. Contains a name connector (بن / ابن / أبو)  OR
      3. Starts with a known name prefix (عبد / محمد / حمد …)
    and is NOT a clear verse opener (يقول / يا …).
    """
    t = text.strip()
    words = t.split()
    if len(words) < 1 or len(words) > 10:
        return False
    if VERSE_OPENER_RE.match(t):
        return False
    # Strong positive signals
    if NAME_CONNECTOR_RE.search(t):
        return True
    if KNOWN_NAME_STARTS.match(t):
        return True
    # Short text with no verb opener is likely a name
    if len(words) <= 4 and not VERSE_OPENER_RE.match(t):
        return True
    return False


def looks_like_verse(text: str) -> bool:
    """
    Return True when text is likely a verse opening.
    A verse is longer than a name and either has a clear opener or is long.
    """
    t = text.strip()
    if len(t) < 10:
        return False
    # Clear verse openers
    if VERSE_OPENER_RE.match(t):
        return True
    # Long text (> 4 words) without name connectors is likely a verse
    words = t.split()
    if len(words) >= 4 and not NAME_CONNECTOR_RE.search(t):
        return True
    return False


def clean_record(r: dict) -> tuple[dict, list[str]]:
    """Return (cleaned_record, list_of_applied_tags)."""
    r = deepcopy(r)
    tags = []

    poet  = (r.get('poet_name')  or '').strip()
    matla = (r.get('matla_text') or '').strip()

    # ── Pattern 1: Swapped fields (matla = "name | page") ─────────────────────
    m = SWAP_DETECT_RE.match(matla)
    if m and looks_like_name(m.group(1)) and looks_like_verse(poet):
        # matla field actually has: real_name | page_number
        # poet field actually has: the verse
        real_name  = m.group(1).strip()
        page_str   = ar_to_latin(m.group(2).strip())
        real_matla = poet  # the verse was in the wrong field

        r['poet_name']  = real_name
        r['matla_text'] = real_matla
        if not r.get('page_number'):
            try:
                r['page_number'] = int(page_str)
            except ValueError:
                pass
        tags.append('swapped_fields')

    # Re-read after potential swap
    poet  = (r.get('poet_name')  or '').strip()
    matla = (r.get('matla_text') or '').strip()

    # ── Pattern 2: Pipe + page in poet_name ────────────────────────────────────
    m = PIPE_PAGE_RE.match(poet)
    if m and not r.get('page_number'):
        real_name = m.group(1).strip()
        page_str  = ar_to_latin(m.group(2).strip())
        r['poet_name'] = real_name
        try:
            r['page_number'] = int(page_str)
        except ValueError:
            pass
        tags.append('poet_pipe_page')

    # Re-read
    poet = (r.get('poet_name') or '').strip()

    # ── Pattern 3: Pure digit in poet_name ────────────────────────────────────
    if PURE_NUM_RE.match(poet) and poet:
        page_str = ar_to_latin(poet.strip())
        if not r.get('page_number'):
            try:
                r['page_number'] = int(page_str.strip())
            except ValueError:
                pass
        r['poet_name'] = 'unknown'
        tags.append('poet_page_leaked')

    # Re-read
    poet  = (r.get('poet_name')  or '').strip()
    matla = (r.get('matla_text') or '').strip()

    # ── Pattern 4: Manuscript header in poet_name ─────────────────────────────
    if MS_HEADER_RE.search(poet):
        r['poet_name'] = 'unknown'
        tags.append('header_row')

    # ── Pattern 5: Digit-only matla (OCR noise) ────────────────────────────────
    if PURE_NUM_RE.match(matla) and matla:
        # It's a leaked page/entry number, not a verse
        page_str = ar_to_latin(matla.strip())
        if not r.get('page_number'):
            try:
                r['page_number'] = int(page_str.strip())
            except ValueError:
                pass
        r['matla_text'] = ''
        tags.append('matla_ocr_noise')

    # Re-read
    matla = (r.get('matla_text') or '').strip()

    # ── Pattern 6: Pipe in matla (attribution note) ───────────────────────────
    if '|' in matla and not 'swapped_fields' in tags:
        parts = matla.split('|', 1)
        left  = parts[0].strip()
        right = parts[1].strip()
        r['matla_text'] = left
        # Only save attribution note if non-empty and not a bare digit
        if right and not PURE_NUM_RE.match(right):
            r['matla_attribution_note'] = right
        tags.append('matla_pipe_split')

    if tags:
        r['_cleaning_tags'] = tags

    return r, tags


def main():
    print(f"Loading {IN_PATH} …")
    with open(IN_PATH, encoding='utf-8') as f:
        data = json.load(f)
    print(f"  {len(data)} records loaded")

    cleaned = []
    audit   = []
    stats   = {}

    for r in data:
        c, tags = clean_record(r)
        cleaned.append(c)
        if tags:
            for t in tags:
                stats[t] = stats.get(t, 0) + 1
            audit.append({
                'id':    r['source_row_id'],
                'tags':  tags,
                'before': {
                    'poet_name':  r.get('poet_name'),
                    'matla_text': r.get('matla_text'),
                    'page_number': r.get('page_number'),
                },
                'after': {
                    'poet_name':  c.get('poet_name'),
                    'matla_text': c.get('matla_text'),
                    'page_number': c.get('page_number'),
                    'matla_attribution_note': c.get('matla_attribution_note'),
                },
            })

    print(f"\nCleaning summary:")
    for tag, count in sorted(stats.items()):
        print(f"  {tag}: {count}")
    print(f"  Total records modified: {len(audit)}")
    print(f"  Unchanged: {len(data) - len(audit)}")

    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {OUT_PATH}")

    with open(AUDIT_PATH, 'w', encoding='utf-8') as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    print(f"Wrote {AUDIT_PATH}")


if __name__ == '__main__':
    main()
