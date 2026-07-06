"""Tests for the WDS HTTP client: browser headers, per-request Accept, warm-up.

The document host ``documents.worldbank.org`` is behind Cloudflare Bot
Management, which 403s the default ``python-httpx`` User-Agent and cookie-less
bursts. These tests pin the mitigations (browser UA, text/JSON Accept split,
best-effort cookie warm-up, 403-raises-for-retry) using an ``httpx.MockTransport``
so no network is touched.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from src.utils.wds_client import BROWSER_HEADERS, build_client, fetch_text, search, warm_up

Handler = Callable[[httpx.Request], httpx.Response]


def _client(handler: Handler) -> httpx.Client:
    """A client with the real browser defaults but a mocked transport."""
    return httpx.Client(
        transport=httpx.MockTransport(handler),
        headers=BROWSER_HEADERS,
        follow_redirects=True,
    )


def test_build_client_sends_a_browser_user_agent() -> None:
    client = build_client()
    try:
        ua = client.headers["user-agent"]
    finally:
        client.close()
    assert "Mozilla/5.0" in ua
    assert "python-httpx" not in ua.lower()


def test_fetch_text_uses_browser_ua_and_text_accept() -> None:
    seen: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["ua"] = request.headers.get("user-agent")
        seen["accept"] = request.headers.get("accept")
        return httpx.Response(200, headers={"content-type": "text/plain"}, text="hello world")

    with _client(handler) as client:
        text = fetch_text(client, "https://documents.worldbank.org/curated/x.txt")

    assert text == "hello world"
    assert seen["ua"] is not None and "Mozilla/5.0" in seen["ua"]
    assert seen["accept"] is not None and "text/html" in seen["accept"]


def test_fetch_text_skips_html_error_pages() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/html"}, text="<html>oops</html>")

    with _client(handler) as client:
        assert fetch_text(client, "https://documents.worldbank.org/y.txt") == ""


def test_fetch_text_raises_on_403_so_caller_can_retry() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, headers={"content-type": "text/html"}, text="<html>403</html>")

    with _client(handler) as client:
        with pytest.raises(httpx.HTTPStatusError):
            fetch_text(client, "https://documents.worldbank.org/z.txt")


def test_search_requests_json_and_normalises_documents() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("accept") == "application/json"
        assert request.url.params.get("qterm") == "inflation"
        payload = {"documents": {"D1": {"txturl": "u1"}, "facets": {"k": "v"}}}
        return httpx.Response(200, json=payload)

    with _client(handler) as client:
        docs = search(client, qterm="inflation", rows=1)

    assert [d["id"] for d in docs] == ["D1"]


def test_warm_up_is_best_effort_and_never_raises() -> None:
    # A 403 response must not raise (warm-up only seeds the cookie jar).
    with _client(lambda _r: httpx.Response(403, text="blocked")) as client:
        warm_up(client, "https://documents.worldbank.org/a.txt")

    # A transport-level failure must also be swallowed.
    def boom(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("nope")

    with _client(boom) as client:
        warm_up(client, "https://documents.worldbank.org/b.txt")
