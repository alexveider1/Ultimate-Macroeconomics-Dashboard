"""Sync HTTP client for the World Bank Documents & Reports (WDS) API (httpx).

Hits the documented public search endpoint
``https://search.worldbank.org/api/v3/wds`` (no API key) plus each document's
``txturl`` (a WB-provided plain-text rendering of the PDF — used here instead of
docling so the downloader image stays light).

- :func:`search` → the top-N documents matching a ``qterm``, each a plain dict
  with ``id``, ``display_title``, ``docty``, ``docdt``, ``pdfurl``, ``txturl``,
  ``url``, ``count`` (country) and ``lang``.
- :func:`fetch_text` → the raw plain text at a document's ``txturl``.

The WDS ``documents`` field is a dict keyed by document id (plus a ``facets``
key when facets are requested); :func:`search` normalises it into a list and
injects the id under ``"id"``. Callers wrap each call in
:func:`src.utils.downloads._call_with_retries`.

**Cloudflare Bot Management.** The document host ``documents.worldbank.org``
(where the ``txturl`` bodies live) sits behind Cloudflare, which ``403``\\ s
requests that look like bots — the default ``python-httpx`` User-Agent and
cookie-less bursts of concurrent requests both trip it (the search host does
not). Two mitigations: the client sends a realistic **browser User-Agent** +
header set (:data:`BROWSER_HEADERS`), and callers :func:`warm_up` the shared
client once so Cloudflare's ``__cf_bm`` cookie is primed before the parallel
:func:`fetch_text` burst (``httpx.Client`` persists cookies across requests).
Genuine transient ``403``\\ s are retried by the caller's
:func:`src.utils.downloads._call_with_retries` wrapper.
"""

import logging
import random
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://search.worldbank.org/api/v3/wds"
DEFAULT_TIMEOUT = 60.0
WDS_FIELDS = "id,docdt,display_title,docty,pdfurl,txturl,url,count,lang,abstracts"

# A small pool of realistic desktop-browser User-Agents. Each built client picks
# one at random and keeps it for the client's lifetime (so it stays consistent
# with the client's ``__cf_bm`` cookie), spreading the fleet across several
# fingerprints instead of every request sharing one bot-flaggable UA.
USER_AGENTS = [
    (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
        "(KHTML, like Gecko) Version/17.4 Safari/605.1.15"
    ),
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
]

# A browser-like header set so Cloudflare's bot manager on documents.worldbank.org
# doesn't 403 the txturl fetches (the default python-httpx UA reads as a bot).
# Accept-Encoding is limited to what httpx can decode without the optional brotli
# dep; the Sec-Fetch-* / DNT headers are the universal signals every real browser
# navigation sends and are cheap extra plausibility.
BROWSER_HEADERS = {
    "User-Agent": USER_AGENTS[0],
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "DNT": "1",
}
_JSON_ACCEPT = "application/json"
_TEXT_ACCEPT = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
# A same-site Referer makes the txturl fetch look like an in-site navigation from
# the documents portal rather than a bare cross-origin bot hit.
_DOCUMENTS_REFERER = "https://documents.worldbank.org/"


def build_client(timeout: float = DEFAULT_TIMEOUT, user_agent: str | None = None) -> httpx.Client:
    """Return an ``httpx.Client`` configured for the WDS API + document hosts.

    Sends :data:`BROWSER_HEADERS` (so the Cloudflare-fronted document host treats
    it as a browser) with a randomly chosen :data:`USER_AGENTS` entry (or the
    supplied ``user_agent``) fixed for the client's lifetime, and persists cookies
    across requests (the default ``httpx.Client`` behaviour) so a warmed
    ``__cf_bm`` cookie is reused.
    """
    headers = dict(BROWSER_HEADERS)
    headers["User-Agent"] = user_agent or random.choice(USER_AGENTS)
    return httpx.Client(
        timeout=timeout,
        follow_redirects=True,
        headers=headers,
    )


def warm_up(client: httpx.Client, url: str) -> bool:
    """Prime Cloudflare's ``__cf_bm`` bot cookie before a concurrent fetch burst.

    A single serial GET seeds the shared client's cookie jar; the response status
    is irrelevant (even a ``403`` sets the cookie), so failures are swallowed. The
    subsequent parallel :func:`fetch_text` calls reuse the cookie and are far less
    likely to be blocked. Returns ``True`` when the request completed (cookie
    likely set), ``False`` on a transport error.
    """
    try:
        client.get(url, headers={"Accept": _TEXT_ACCEPT, "Referer": _DOCUMENTS_REFERER})
        return True
    except Exception as exc:  # noqa: BLE001 - warm-up is best-effort
        logger.debug("WDS cookie warm-up failed (ignored): %s (%s)", url, exc)
        return False


def _normalise_documents(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Turn the WDS ``documents`` dict into a list, injecting the id per record."""
    documents = payload.get("documents")
    if not isinstance(documents, dict):
        return []
    records: list[dict[str, Any]] = []
    for key, value in documents.items():
        if key == "facets" or not isinstance(value, dict):
            continue
        value.setdefault("id", key)
        records.append(value)
    return records


def search(
    client: httpx.Client,
    qterm: str,
    rows: int,
    base_url: str = DEFAULT_BASE_URL,
    doc_types: Optional[list[str]] = None,
    lang: Optional[str] = None,
    from_year: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Return up to ``rows`` documents matching ``qterm`` (default WDS ranking).

    Args:
        client: Shared HTTP client.
        qterm: Free-text query (searched across title/abstract/country/etc.).
        rows: Max records to return (WDS ``rows`` param).
        base_url: WDS search endpoint.
        doc_types: Optional ``docty_exact`` filter; multiple values are OR-ed
            with the WDS ``^`` separator. Empty/None = no document-type filter.
        lang: Optional ``lang_exact`` filter (e.g. ``"English"``).
        from_year: Optional lower date bound → ``strdate=<year>-01-01``.
    """
    params: dict[str, Any] = {
        "format": "json",
        "qterm": qterm,
        "rows": rows,
        "fl": WDS_FIELDS,
    }
    if lang:
        params["lang_exact"] = lang
    if doc_types:
        params["docty_exact"] = "^".join(doc_types)
    if from_year:
        params["strdate"] = f"{int(from_year)}-01-01"

    resp = client.get(base_url, params=params, headers={"Accept": _JSON_ACCEPT})
    resp.raise_for_status()
    payload = resp.json()
    return _normalise_documents(payload) if isinstance(payload, dict) else []


def fetch_text(client: httpx.Client, txturl: str) -> str:
    """Return the plain text at a document's ``txturl`` (empty on non-text/HTML).

    Raises ``httpx.HTTPStatusError`` on a non-2xx status (e.g. a Cloudflare
    ``403``) so the caller's retry wrapper can back off and re-try.
    """
    resp = client.get(txturl, headers={"Accept": _TEXT_ACCEPT, "Referer": _DOCUMENTS_REFERER})
    resp.raise_for_status()
    content_type = resp.headers.get("content-type", "")
    # The txturl occasionally 200s with an HTML error/landing page; skip those so
    # we don't embed markup. Genuine text bodies are served as text/plain.
    if "html" in content_type.lower():
        logger.warning("txturl returned HTML rather than plain text: %s", txturl)
        return ""
    return resp.text
