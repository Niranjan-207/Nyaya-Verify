"""
src/utils/visualizer.py — Visual Evidence Extractor
=====================================================
Uses PyMuPDF (fitz) to crop the exact section of a PDF page where the
"winning" retrieved chunk was found, and returns PNG bytes suitable for
direct display via st.image() in the Streamlit Evidence Vault.

Rendering Strategy
------------------
  1. Take the first 6 words of the chunk text as a search key.
  2. Call page.search_for() — returns bounding rectangles of text matches.
  3. Expand the first hit rect with configurable vertical padding.
  4. Render only that cropped rectangle at `dpi` (default 150).
  5. If no hit is found, fall back to rendering the full page.

PNG bytes are returned directly (no disk I/O) so Streamlit can display
them as in-memory images without temp file clutter.

Security Note
-------------
  All file access is restricted to DATA_PDF_DIR.
  Path traversal is prevented by os.path.basename().
"""

import os
import re
from typing import Optional

import fitz  # PyMuPDF

# ---------------------------------------------------------------------------
# Base directory for clinical protocol PDFs
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
DATA_PDF_DIR = os.path.join(_PROJECT_ROOT, "data", "input_pdfs")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def extract_pdf_clip(
    filename: str,
    page_number: int,
    snippet: str,
    vertical_padding_pts: int = 45,
    dpi: int = 150,
    pdf_dir: Optional[str] = None,
) -> Optional[bytes]:
    """
    Crop and render the region of a PDF page containing `snippet`.

    Parameters
    ----------
    filename : str
        PDF filename (basename only — no path traversal).
    page_number : int
        1-indexed page number from chunk metadata.
    snippet : str
        The chunk text (first ~100 chars used as search key).
    vertical_padding_pts : int
        Points to expand the crop rect above and below the match.
    dpi : int
        Render resolution. 150 dpi is sharp enough for Streamlit display
        without excessive memory use.
    pdf_dir : str, optional
        Override for the PDF directory (useful in tests).

    Returns
    -------
    bytes | None
        PNG image bytes, or None if the file/page is unavailable.
    """
    # Security: strip any path component from the filename
    safe_name = os.path.basename(filename)
    base_dir  = pdf_dir or DATA_PDF_DIR
    pdf_path  = os.path.join(base_dir, safe_name)

    if not os.path.exists(pdf_path):
        return None

    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return None

    try:
        # Validate page number
        if page_number < 1 or page_number > len(doc):
            return _render_full_page(doc, 0, dpi)

        page = doc[page_number - 1]  # fitz uses 0-indexed pages

        # Build a compact search key from first 6 meaningful words
        search_key = _build_search_key(snippet, max_words=6)
        clip_rect  = _locate_snippet(page, search_key, vertical_padding_pts)

        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, clip=clip_rect, colorspace=fitz.csRGB)
        return pix.tobytes("png")

    except Exception:
        return None
    finally:
        doc.close()


def render_full_page(
    filename: str,
    page_number: int,
    dpi: int = 120,
    pdf_dir: Optional[str] = None,
) -> Optional[bytes]:
    """
    Render a full PDF page as PNG bytes.
    Used as an explicit fallback when snippet localisation fails.
    """
    safe_name = os.path.basename(filename)
    base_dir  = pdf_dir or DATA_PDF_DIR
    pdf_path  = os.path.join(base_dir, safe_name)

    if not os.path.exists(pdf_path):
        return None

    try:
        doc = fitz.open(pdf_path)
        return _render_full_page(doc, max(0, page_number - 1), dpi)
    except Exception:
        return None
    finally:
        try:
            doc.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------
def _build_search_key(text: str, max_words: int = 6) -> str:
    """
    Extract a short, clean search key from the start of the chunk text.
    Strips section-header prefixes (ALL CAPS lines) and trims whitespace.
    """
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    body  = ""
    for line in lines:
        alpha = [c for c in line if c.isalpha()]
        # Skip ALL-CAPS header lines
        if alpha and all(c.isupper() for c in alpha) and len(alpha) >= 3:
            continue
        body = line
        break

    if not body:
        body = text.strip()

    # Take first max_words words, strip punctuation from ends
    words = body.split()[:max_words]
    return " ".join(words).strip(".,;:()")


def _locate_snippet(
    page: fitz.Page,
    search_key: str,
    v_pad: int,
) -> fitz.Rect:
    """
    Search for search_key on page. Returns an expanded crop rect if found,
    otherwise returns the full page rect as fallback.
    """
    if not search_key:
        return page.rect

    hits = page.search_for(search_key, quads=False)

    if not hits:
        # Try with fewer words as fallback
        words = search_key.split()
        if len(words) > 3:
            hits = page.search_for(" ".join(words[:3]))

    if hits:
        r = hits[0]
        return fitz.Rect(
            max(0,                  r.x0 - 10),
            max(0,                  r.y0 - v_pad),
            min(page.rect.width,    r.x1 + 10),
            min(page.rect.height,   r.y1 + v_pad * 4),
        )

    return page.rect


def _render_full_page(doc: fitz.Document, page_idx: int, dpi: int) -> Optional[bytes]:
    """Render the entire page at given DPI and return PNG bytes."""
    try:
        page = doc[page_idx]
        mat  = fitz.Matrix(dpi / 72, dpi / 72)
        pix  = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        return pix.tobytes("png")
    except Exception:
        return None
