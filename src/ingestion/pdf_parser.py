import fitz  # PyMuPDF
import re
import os
from typing import List, Dict, Any, Optional

def extract_text_with_metadata(pdf_path: str, statute_year: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Extracts text blocks from a PDF using PyMuPDF and attaches metadata.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF missing at {pdf_path}")
        
    filename = os.path.basename(pdf_path)
    
    # Identify year from filename fallback
    if statute_year is None:
        match = re.search(r'\d{4}', filename)
        if match:
            statute_year = int(match.group())
            
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
                
            # Replace inline newlines with spaces to avoid arbitrarily breaking paragraphs
            clean_text = " ".join(text.splitlines())
            
            extracted_blocks.append({
                "text": clean_text,
                "metadata": {
                    "filename": filename,
                    "page": page_num + 1,
                    "statute_year": statute_year
                }
            })
            
    doc.close()
    return extracted_blocks
