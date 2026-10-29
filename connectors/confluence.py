"""Confluence ingestion connector.

Pulls pages from Confluence Cloud, normalizes them to clean text + metadata,
and hands them to the EXISTING ingestion pipeline (chunk -> embed -> store).

Design:
  - ConfluenceClient.fetch_pages()  -> the network part (needs a real API token).
  - normalize_page()                -> pure logic (HTML -> text + metadata), testable offline.
  - A "mock" client returns canned pages so the whole pipeline can be verified
    with NO real Confluence access. Swap in the real client + credentials later
    and nothing else changes.

Credentials (real mode) come from .env:
    CONFLUENCE_BASE_URL=https://your-org.atlassian.net/wiki
    CONFLUENCE_EMAIL=you@company.com
    CONFLUENCE_API_TOKEN=xxxx
"""

from __future__ import annotations

import base64
import json
import os
import re
import urllib.request
from dataclasses import dataclass
from typing import List, Protocol


@dataclass
class ConfluencePage:
    """A raw page as returned by Confluence (before normalization)."""
    id: str
    title: str
    body_html: str      # Confluence "storage format" (XHTML-like)
    space: str
    url: str
    updated_at: str


@dataclass
class NormalizedDoc:
    """A page turned into clean text + metadata, ready for the pipeline."""
    doc_id: str
    text: str
    metadata: dict


# --- HTML -> text (good enough for chunking; not a full HTML parser) ---
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t]+")


def html_to_text(html: str) -> str:
    # Replace block tags with newlines so structure survives, then strip tags.
    text = re.sub(r"</(p|div|h[1-6]|li|tr)>", "\n", html, flags=re.IGNORECASE)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = _TAG_RE.sub("", text)
    # Unescape the few common entities and tidy whitespace.
    text = (text.replace("&amp;", "&").replace("&lt;", "<")
                .replace("&gt;", ">").replace("&quot;", '"').replace("&#39;", "'"))
    lines = [_WS_RE.sub(" ", ln).strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def normalize_page(page: ConfluencePage) -> NormalizedDoc:
    """Pure function: raw Confluence page -> clean text + metadata.

    Stable doc_id derived from the page id so re-ingesting UPDATES rather than
    duplicates (the basis for incremental sync).
    """
    title = page.title.strip()
    body = html_to_text(page.body_html)
    # Prepend the title so it's part of the embedded/searchable text.
    text = f"# {title}\n\n{body}" if title else body
    return NormalizedDoc(
        doc_id=f"CONF:{page.id}",
        text=text,
        metadata={
            "source": "confluence",
            "page_id": page.id,
            "title": title,
            "space": page.space,
            "url": page.url,
            "updated_at": page.updated_at,
        },
    )


class PageSource(Protocol):
    """Anything that can supply Confluence pages (real API or a mock)."""
    def fetch_pages(self, limit: int = 50) -> List[ConfluencePage]: ...


class ConfluenceClient:
    """Real client: calls the Confluence Cloud REST API over HTTPS."""

    def __init__(self) -> None:
        self.base_url = os.environ.get("CONFLUENCE_BASE_URL", "").rstrip("/")
        self.email = os.environ.get("CONFLUENCE_EMAIL", "")
        self.token = os.environ.get("CONFLUENCE_API_TOKEN", "")
        if not all([self.base_url, self.email, self.token]):
            raise RuntimeError(
                "Confluence credentials missing. Set CONFLUENCE_BASE_URL, "
                "CONFLUENCE_EMAIL, CONFLUENCE_API_TOKEN in .env (or use MockConfluenceClient)."
            )

    def _auth_header(self) -> str:
        raw = f"{self.email}:{self.token}".encode("utf-8")
        return "Basic " + base64.b64encode(raw).decode("ascii")

    def fetch_pages(self, limit: int = 50) -> List[ConfluencePage]:
        url = f"{self.base_url}/rest/api/content?limit={limit}&expand=body.storage,space,version"
        req = urllib.request.Request(url, headers={"Authorization": self._auth_header()})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        pages = []
        for r in data.get("results", []):
            pages.append(ConfluencePage(
                id=str(r["id"]),
                title=r.get("title", ""),
                body_html=r.get("body", {}).get("storage", {}).get("value", ""),
                space=r.get("space", {}).get("key", ""),
                url=self.base_url + r.get("_links", {}).get("webui", ""),
                updated_at=r.get("version", {}).get("when", ""),
            ))
        return pages


class MockConfluenceClient:
    """Mock: returns canned pages so the pipeline runs with NO real Confluence.
    Lets us verify fetch -> normalize -> chunk -> embed -> retrieve end to end."""

    def fetch_pages(self, limit: int = 50) -> List[ConfluencePage]:
        return [
            ConfluencePage(
                id="1001", title="Incident Response Runbook",
                body_html="<h1>Incident Response</h1><p>When an alert fires, the on-call "
                          "engineer acknowledges it within 5 minutes.</p><p>Severity 1 "
                          "incidents require paging the team lead and opening a war room.</p>",
                space="OPS", url="https://example.atlassian.net/wiki/OPS/1001",
                updated_at="2026-01-10T09:00:00Z",
            ),
            ConfluencePage(
                id="1002", title="Deployment Checklist",
                body_html="<h1>Deployment Checklist</h1><ul><li>Run the test suite</li>"
                          "<li>Tag the release</li><li>Deploy to staging first</li>"
                          "<li>Verify health checks before production</li></ul>",
                space="ENG", url="https://example.atlassian.net/wiki/ENG/1002",
                updated_at="2026-01-12T14:30:00Z",
            ),
        ]
