import fitz  # PyMuPDF
import re
import os
from typing import List, Dict, Any, Optional

def extract_text_with_metadata(
    pdf_path: str,
    doc_year: Optional[int] = None,
    # backwards-compat alias kept so existing ingest_all.py calls still work
    statute_year: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Extracts text blocks from a PDF using PyMuPDF and attaches metadata.
    `doc_year` is the canonical parameter; `statute_year` is accepted as an
    alias for backward compatibility with pre-Medify ingestion scripts.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF missing at {pdf_path}")

    filename = os.path.basename(pdf_path)

    # Resolve year: prefer doc_year, fall back to statute_year alias, then filename
    year = doc_year or statute_year
    if year is None:
        match = re.search(r'\d{4}', filename)
        if match:
            year = int(match.group())

    doc = fitz.open(pdf_path)
    extracted_blocks = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("blocks")

        for b in blocks:
            # PyMuPDF format: b[4] contains the actual text string
            text = b[4].strip()
            if not text:
                continue

            # Replace inline newlines with spaces to avoid breaking paragraphs
            clean_text = " ".join(text.splitlines())

            extracted_blocks.append({
                "text": clean_text,
                "metadata": {
                    "filename": filename,
                    "page": page_num + 1,
                    "doc_year": year,
                }
            })

    doc.close()
    return extracted_blocks
