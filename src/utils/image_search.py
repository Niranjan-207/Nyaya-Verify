"""
src/utils/image_search.py — Clinical Web Image Search
======================================================
Fetches the top-1 clinical image URL from DuckDuckGo Images for a given
medical term.

SILENT FAIL CONTRACT
--------------------
  This function NEVER raises an exception and NEVER returns a placeholder.
  Any error (network, no results, bad URL, timeout) → returns None.
  The caller (app.py) simply skips the image element when None is returned.

Search Strategy
---------------
  query = "{medical_term} clinical ophthalmology finding fundus slit-lamp"
  → top 1 image result URL is returned after a reachability check.

Privacy Note
------------
  DuckDuckGo image search is performed over HTTPS and does not require an
  API key. No patient data is ever included in the search term — only the
  extracted condition name (e.g. "Glaucoma", "Cataract").
"""

import urllib.request
from typing import Optional

# Reachability check timeout in seconds — tight enough not to stall the UI
_URL_TIMEOUT = 4

# Search suffix appended to every medical term query
_SEARCH_SUFFIX = "clinical ophthalmology finding fundus slit-lamp"


def get_clinical_image(medical_term: str) -> Optional[str]:
    """
    Return the top-1 DuckDuckGo Images URL for a clinical ophthalmology term.

    Parameters
    ----------
    medical_term : str
        The primary condition name, e.g. "Glaucoma", "Cataract", "Acne".

    Returns
    -------
    str | None
        A publicly reachable image URL, or None on any failure.

    Examples
    --------
    >>> url = get_clinical_image("open-angle glaucoma")
    >>> if url:
    ...     st.image(url)   # show it
    # If None, skip entirely — no placeholder, no error message
    """
    if not medical_term or not medical_term.strip():
        return None

    try:
        from duckduckgo_search import DDGS
    except ImportError:
        # duckduckgo-search not installed — silent fail
        return None

    query = f"{medical_term.strip()} {_SEARCH_SUFFIX}"

    try:
        with DDGS() as ddgs:
            hits = list(ddgs.images(query, max_results=3))
    except Exception:
        return None

    if not hits:
        return None

    # Try each result until we find one that is actually reachable
    for hit in hits:
        url = hit.get("image") or hit.get("url") or ""
        if not url.startswith("http"):
            continue
        if _is_reachable(url):
            return url

    return None


# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------
def _is_reachable(url: str) -> bool:
    """
    Issue a HEAD request (fallback GET) to check if the URL is accessible.
    Returns False on any exception — never raises.
    """
    try:
        req = urllib.request.Request(url, method="HEAD")
        req.add_header("User-Agent", "Mozilla/5.0")
        with urllib.request.urlopen(req, timeout=_URL_TIMEOUT):
            return True
    except Exception:
        pass

    # Some servers reject HEAD — try a small GET range instead
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "Mozilla/5.0")
        req.add_header("Range", "bytes=0-1023")
        with urllib.request.urlopen(req, timeout=_URL_TIMEOUT):
            return True
    except Exception:
        return False
