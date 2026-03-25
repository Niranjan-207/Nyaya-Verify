"""
src/utils/image_search.py — Clinical Web Image Search
======================================================
Multi-strategy medical image fetcher for any condition.

Strategy Chain
--------------
  1. Wikipedia media-list API
       GET /api/rest_v1/page/media-list/{title}
       Returns every image embedded in the Wikipedia article body.
       Filtered to keep only clinically relevant images (anatomy diagrams,
       clinical photos, histology slides) and skip icons/flags/maps.

  2. Wikipedia summary thumbnail
       GET /api/rest_v1/page/summary/{slug}
       The article's primary infobox image — a reliable single image for
       almost every named medical condition.

  3. Wikipedia search → canonical title → repeat 1 + 2
       Resolves abbreviations and alternate spellings before the above.

  4. DuckDuckGo — anatomy / diagram query
       "{term} anatomy diagram labeled medical"
       Targets medical illustration and textbook sites.

  5. DuckDuckGo — pathophysiology query
       "{term} pathophysiology mechanism illustration"
       Targets clinical science diagrams.

  6. DuckDuckGo — clinical presentation query
       "{term} clinical signs symptoms medical"
       Last-resort for rare conditions not well covered on Wikipedia.

Filtering
---------
  • No SVG files (browser rendering of remote SVGs is unreliable).
  • Skip filenames that indicate icons, flags, maps, portraits or logos.
  • Prefer upload.wikimedia.org CDN (always reachable, no auth).
  • Deduplicate: same URL never returned twice.

SILENT FAIL CONTRACT
--------------------
  get_clinical_images() / get_clinical_visualization() never raise.
  Returns fewer images (including an empty list) on any failure.
"""

from __future__ import annotations

import json
import urllib.request
import urllib.parse
from typing import Optional, List

_TIMEOUT = 7  # seconds per HTTP call

# Filename fragments that indicate a non-clinical Wikipedia image.
# Keep this list conservative — false-positives hide useful images.
_SKIP_FRAGMENTS = (
    "flag_of_", "_flag.", "coat_of_arms", "location_map",
    "location_dot", "_map.", "_map_", "logo_of", "logo.",
    "commons-logo", "wikidata", "_icon.", "icon_",
    "portrait_of", "signature_of", "autograph",
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_clinical_images(medical_term: str, count: int = 3) -> List[str]:
    """
    Return up to `count` distinct, clinically relevant image URLs.

    Searches Wikipedia (article media-list + summary thumbnail) then
    DuckDuckGo with three different medically-focused query variants.
    Works for any medical condition — ophthalmology, cardiology,
    neurology, endocrinology, etc.

    Parameters
    ----------
    medical_term : str
        Primary condition name extracted from the query,
        e.g. "Glaucoma", "Myocardial Infarction", "Tuberculosis".
    count : int
        Maximum number of URLs to return (default 3).

    Returns
    -------
    list[str]
        0–`count` unique image URLs. Never raises.
    """
    if not medical_term or not medical_term.strip():
        return []

    term = medical_term.strip()
    seen: set[str] = set()
    urls: List[str] = []

    def _add(url: str) -> None:
        if url and url.startswith("https://") and url not in seen:
            seen.add(url)
            urls.append(url)

    # ── Stage 1 + 2: Wikipedia (canonical title first) ────────────────────
    canonical = _wiki_canonical_title(term)

    if canonical:
        # 1a. Article media-list — multiple images from the article body
        for u in _wiki_article_images(canonical, want=count):
            if len(urls) >= count:
                break
            _add(u)

        # 1b. Summary thumbnail — infobox image (reliable single image)
        if len(urls) < count:
            _add(_wiki_summary_image(canonical.replace(" ", "_")) or "")

    # ── Stage 3: Wikipedia fallback via direct slug ───────────────────────
    if len(urls) < count:
        direct_thumb = _wiki_summary_image(term.replace(" ", "_"))
        if direct_thumb:
            _add(direct_thumb)

    # ── Stages 4–6: DuckDuckGo with three targeted query variants ─────────
    if len(urls) < count:
        ddgs_queries = [
            f"{term} anatomy diagram labeled medical illustration",
            f"{term} pathophysiology mechanism clinical illustration",
            f"{term} clinical signs symptoms medical",
        ]
        for q in ddgs_queries:
            if len(urls) >= count:
                break
            for u in _ddgs_images_query(q, need=count - len(urls) + 1):
                if len(urls) >= count:
                    break
                _add(u)

    return urls[:count]


def get_clinical_visualization(medical_term: str) -> Optional[str]:
    """
    Return a single image URL for the medical condition (backwards-compat).
    Prefers Wikipedia summary thumbnail; falls back to DuckDuckGo.
    """
    if not medical_term or not medical_term.strip():
        return None
    results = get_clinical_images(medical_term, count=1)
    return results[0] if results else None


# backwards-compat alias
get_clinical_image = get_clinical_visualization


# ---------------------------------------------------------------------------
# Wikipedia helpers
# ---------------------------------------------------------------------------

def _wiki_canonical_title(term: str) -> Optional[str]:
    """
    Resolve the canonical Wikipedia article title for a medical condition.

    Step A: direct summary lookup (fast — works when the term is already
            the article title, e.g. "Glaucoma", "Myocardial infarction").
    Step B: Wikipedia search API (handles abbreviations, alternate names,
            e.g. "MI" → "Myocardial infarction", "DM" → "Diabetes mellitus").
    """
    # Step A — direct summary
    try:
        slug = term.strip().replace(" ", "_")
        data = _wiki_summary_json(slug)
        if data and data.get("type") in ("standard", "disambiguation"):
            title = data.get("title")
            if title:
                return title
    except Exception:
        pass

    # Step B — search API (add "medical" to bias results toward medicine)
    try:
        search_url = (
            "https://en.wikipedia.org/w/api.php?"
            + urllib.parse.urlencode({
                "action":   "query",
                "list":     "search",
                "srsearch": f"{term} medical condition",
                "format":   "json",
                "srlimit":  "3",
            })
        )
        req = urllib.request.Request(search_url)
        req.add_header("User-Agent", "MedVerify/1.0 (medical-education-tool)")
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read().decode())
        hits = data.get("query", {}).get("search", [])
        if hits:
            return hits[0].get("title")
    except Exception:
        pass

    return None


def _wiki_article_images(canonical_title: str, want: int = 3) -> List[str]:
    """
    Fetch images from the Wikipedia article's media-list endpoint.

    Returns up to `want` direct HTTPS CDN URLs (upload.wikimedia.org).
    Skips SVG files and filenames matching _SKIP_FRAGMENTS.
    Prefers higher-resolution srcset entries.
    """
    try:
        api_url = (
            "https://en.wikipedia.org/api/rest_v1/page/media-list/"
            + urllib.parse.quote(
                canonical_title.replace(" ", "_"), safe=""
            )
        )
        req = urllib.request.Request(api_url)
        req.add_header("User-Agent", "MedVerify/1.0 (medical-education-tool)")
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read().decode())
    except Exception:
        return []

    results: List[str] = []

    for item in data.get("items", []):
        if len(results) >= want:
            break

        # Only raster images shown in the article body
        if item.get("type") != "image":
            continue
        if not item.get("showInGallery", False):
            continue

        file_title = item.get("title", "").lower()

        # Skip non-clinical images (flags, logos, maps, portraits…)
        if any(frag in file_title for frag in _SKIP_FRAGMENTS):
            continue

        # Skip SVG — use raster only for reliable browser rendering
        if file_title.endswith(".svg"):
            continue

        # Pick the highest-resolution entry from srcset
        srcset = item.get("srcset", [])
        if not srcset:
            continue

        def _scale(entry: dict) -> float:
            try:
                return float(entry.get("scale", "1x").replace("x", ""))
            except ValueError:
                return 1.0

        best_src = sorted(srcset, key=_scale, reverse=True)[0].get("src", "")

        # srcset entries often start with "//" — normalise to https
        if best_src.startswith("//"):
            best_src = "https:" + best_src

        if best_src.startswith("https://"):
            results.append(best_src)

    return results


def _wiki_summary_image(slug: str) -> Optional[str]:
    """Return the thumbnail URL from a Wikipedia article summary."""
    data = _wiki_summary_json(slug)
    if data:
        return data.get("thumbnail", {}).get("source") or None
    return None


def _wiki_summary_json(slug: str) -> Optional[dict]:
    """Raw Wikipedia summary JSON fetch. Returns None on any failure."""
    try:
        api_url = (
            "https://en.wikipedia.org/api/rest_v1/page/summary/"
            + urllib.parse.quote(slug, safe="")
        )
        req = urllib.request.Request(api_url)
        req.add_header("User-Agent", "MedVerify/1.0 (medical-education-tool)")
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# DuckDuckGo helpers
# ---------------------------------------------------------------------------

def _ddgs_images_query(query: str, need: int = 3) -> List[str]:
    """
    DuckDuckGo image search for one query string.
    Returns up to `need` valid HTTPS URLs. Never raises.
    """
    urls: List[str] = []
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            hits = list(ddgs.images(query, max_results=need + 6))
        for hit in hits:
            url = hit.get("image", "")
            if url.startswith("https://"):
                urls.append(url)
            if len(urls) >= need:
                break
    except Exception:
        pass
    return urls


def _ddgs_images(term: str, need: int = 3) -> List[str]:
    """Legacy helper — generic DuckDuckGo query (backwards-compat)."""
    return _ddgs_images_query(f"{term} medical clinical", need=need)


def _ddgs_image(term: str) -> Optional[str]:
    """Single-URL convenience wrapper (backwards-compat)."""
    results = _ddgs_images(term, need=1)
    return results[0] if results else None
