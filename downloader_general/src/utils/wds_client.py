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
"""

import logging
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://search.worldbank.org/api/v3/wds"
DEFAULT_TIMEOUT = 60.0
WDS_FIELDS = "id,docdt,display_title,docty,pdfurl,txturl,url,count,lang,abstracts"


def build_client(timeout: float = DEFAULT_TIMEOUT) -> httpx.Client:
    """Return an ``httpx.Client`` configured for the WDS API + document hosts."""
    return httpx.Client(
        timeout=timeout,
        follow_redirects=True,
        headers={"Accept": "application/json"},
    )


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

    resp = client.get(base_url, params=params)
    resp.raise_for_status()
    payload = resp.json()
    return _normalise_documents(payload) if isinstance(payload, dict) else []


def fetch_text(client: httpx.Client, txturl: str) -> str:
    """Return the plain text at a document's ``txturl`` (empty on non-text/HTML)."""
    resp = client.get(txturl)
    resp.raise_for_status()
    content_type = resp.headers.get("content-type", "")
    # The txturl occasionally 200s with an HTML error/landing page; skip those so
    # we don't embed markup. Genuine text bodies are served as text/plain.
    if "html" in content_type.lower():
        logger.warning("txturl returned HTML rather than plain text: %s", txturl)
        return ""
    return resp.text
