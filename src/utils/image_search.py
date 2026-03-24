"""
src/utils/image_search.py — Clinical Web Image Search
======================================================
Returns a relevant medical image URL for a given condition name.

Two-Stage Strategy
------------------
  Stage 1 — Wikipedia REST API  (primary, most reliable)
    Calls https://en.wikipedia.org/api/rest_v1/page/summary/{term}
    Wikipedia has a high-quality thumbnail for every major medical condition
    (Glaucoma, Cataract, Diabetes, etc.) and the upload.wikimedia.org CDN
    is always reachable without authentication.
    If the direct slug lookup misses, falls back to the Wikipedia search API
    to resolve the correct article title first.

  Stage 2 — DuckDuckGo Images  (fallback)
    Used when Wikipedia returns no thumbnail (rare conditions, abbreviations).
    Returns the first valid http(s) URL from the results without a slow
    reachability check — the previous HEAD/GET approach was the main cause
    of images never appearing.

SILENT FAIL CONTRACT
--------------------
  get_clinical_visualization() never raises. Any failure → returns None.
  The caller (app.py) skips the image element entirely when None.
"""

import json
import urllib.request
import urllib.parse
from typing import Optional, List

_TIMEOUT = 6  # seconds — enough for Wikipedia, not long enough to stall UI


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_clinical_visualization(medical_term: str) -> Optional[str]:
    """
    Return an image URL relevant to the medical condition.

    Tries Wikipedia thumbnail first (reliable CDN, no auth).
    Falls back to DuckDuckGo Images if Wikipedia has no thumbnail.

    Parameters
    ----------
    medical_term : str
        Condition name, e.g. "Glaucoma", "Diabetic Retinopathy", "Cataract".

    Returns
    -------
    str | None
        Image URL, or None on any failure.
    """
    if not medical_term or not medical_term.strip():
        return None

    return _wikipedia_image(medical_term) or _ddgs_image(medical_term)


def get_clinical_images(medical_term: str, count: int = 3) -> List[str]:
    """
    Return up to `count` distinct image URLs for the medical term.

    Order: Wikipedia thumbnail first (reliable CDN), then DuckDuckGo fills
    remaining slots. Duplicates are removed. Silently returns fewer images
    (including an empty list) on any failure.

    Parameters
    ----------
    medical_term : str
        Condition name, e.g. "Fungal Keratitis", "Glaucoma".
    count : int
        Maximum number of URLs to return (default 3).

    Returns
    -------
    list[str]
        0–`count` unique image URLs, never raises.
    """
    if not medical_term or not medical_term.strip():
        return []

    urls: List[str] = []

    # Stage 1: Wikipedia thumbnail
    wiki = _wikipedia_image(medical_term)
    if wiki:
        urls.append(wiki)

    # Stage 2: DDGS fills remaining slots
    if len(urls) < count:
        for url in _ddgs_images(medical_term, need=count - len(urls) + 2):
            if url not in urls:
                urls.append(url)
            if len(urls) >= count:
                break

    return urls[:count]


# Backwards-compatible single-URL alias
get_clinical_image = get_clinical_visualization


# ---------------------------------------------------------------------------
# Stage 1 — Wikipedia REST API
# ---------------------------------------------------------------------------
def _wikipedia_image(term: str) -> Optional[str]:
    """
    Fetch the Wikipedia article thumbnail for the given medical term.
    Two-step: direct slug lookup → search API fallback.
    """
    # Step A: direct title lookup
    url = _wiki_summary_image(term.strip().replace(" ", "_"))
    if url:
        return url

    # Step B: search API to get the canonical article title
    try:
        search_url = (
            "https://en.wikipedia.org/w/api.php?"
            + urllib.parse.urlencode({
                "action": "query",
                "list": "search",
                "srsearch": term,
                "format": "json",
                "srlimit": "3",
            })
        )
        req = urllib.request.Request(search_url)
        req.add_header("User-Agent", "MedVerify/1.0 (medical-education-tool)")
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read().decode())

        hits = data.get("query", {}).get("search", [])
        for hit in hits:
            title = hit.get("title", "")
            img = _wiki_summary_image(title.replace(" ", "_"))
            if img:
                return img
    except Exception:
        pass

    return None


def _wiki_summary_image(slug: str) -> Optional[str]:
    """
    Call the Wikipedia summary endpoint for one article slug.
    Returns the thumbnail URL or None.
    """
    try:
        api_url = (
            "https://en.wikipedia.org/api/rest_v1/page/summary/"
            + urllib.parse.quote(slug, safe="")
        )
        req = urllib.request.Request(api_url)
        req.add_header("User-Agent", "MedVerify/1.0 (medical-education-tool)")
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read().decode())
        return data.get("thumbnail", {}).get("source") or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Stage 2 — DuckDuckGo Images fallback
# ---------------------------------------------------------------------------
def _ddgs_images(term: str, need: int = 3) -> List[str]:
    """
    DuckDuckGo image search — returns up to `need` valid https URLs.
    No blocking reachability check; first valid URLs are returned directly.
    """
    urls: List[str] = []
    try:
        from duckduckgo_search import DDGS
        query = f"{term} medical clinical"
        with DDGS() as ddgs:
            hits = list(ddgs.images(query, max_results=need + 4))
        for hit in hits:
            url = hit.get("image", "")
            if url.startswith("https://"):
                urls.append(url)
            if len(urls) >= need:
                break
    except Exception:
        pass
    return urls


def _ddgs_image(term: str) -> Optional[str]:
    """Single-URL convenience wrapper around _ddgs_images."""
    results = _ddgs_images(term, need=1)
    return results[0] if results else None
