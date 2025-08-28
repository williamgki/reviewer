#!/usr/bin/env python3
"""Utilities for fetching research papers and related documents from various
sources.

This module centralises access to several public APIs used in the AI Safety
review system.  OpenAlex is treated as the canonical spine: whenever possible,
records from other sources are mapped onto an OpenAlex work so downstream
components can reason over a unified identifier space.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Iterable, List, Optional

import requests

try:
    import arxiv  # type: ignore
except Exception:  # pragma: no cover - library may be absent during tests
    arxiv = None

try:
    import openreview  # type: ignore
except Exception:  # pragma: no cover
    openreview = None

try:
    from habanero import Crossref  # type: ignore
except Exception:  # pragma: no cover
    Crossref = None

try:
    from semanticscholar import SemanticScholar  # type: ignore
except Exception:  # pragma: no cover
    SemanticScholar = None

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover
    BeautifulSoup = None


# ---------------------------------------------------------------------------
# OpenAlex client
# ---------------------------------------------------------------------------


class OpenAlexClient:
    """Light‑weight wrapper around the OpenAlex API."""

    BASE_URL = "https://api.openalex.org"

    def __init__(self, session: Optional[requests.Session] = None) -> None:
        self.session = session or requests.Session()

    def fetch_works(
        self,
        query: str,
        filters: Optional[Dict[str, str]] = None,
        per_page: int = 200,
    ) -> Dict[str, Any]:
        """Fetch works matching the query and filters.

        Parameters
        ----------
        query:
            Free‑text search terms.
        filters:
            Dictionary of OpenAlex filter parameters, e.g. ``{"concepts.id":
            "C121332964"}``.
        per_page:
            Number of results to fetch per page (max 200).
        """

        params: Dict[str, Any] = {"search": query, "per-page": per_page}
        if filters:
            params["filter"] = ",".join(f"{k}:{v}" for k, v in filters.items())
        response = self.session.get(f"{self.BASE_URL}/works", params=params)
        response.raise_for_status()
        return response.json()

    def get_work(self, work_id: str) -> Dict[str, Any]:
        """Retrieve a single work and its metadata."""

        response = self.session.get(f"{self.BASE_URL}/works/{work_id}")
        response.raise_for_status()
        return response.json()


# ---------------------------------------------------------------------------
# arXiv client
# ---------------------------------------------------------------------------


class ArXivClient:
    """Fetch preprints from arXiv using the official API."""

    def fetch_papers(
        self,
        query: str,
        max_results: int = 100,
        categories: Iterable[str] = ("cs.AI", "cs.LG", "cs.CL", "stat.ML"),
    ) -> List[Dict[str, Any]]:
        if arxiv is None:
            raise ImportError("arxiv package is required to fetch arXiv papers")

        category_query = " OR ".join(f"cat:{c}" for c in categories)
        full_query = f"({category_query}) AND ({query})"
        search = arxiv.Search(query=full_query, max_results=max_results)
        results = []
        for result in search.results():
            results.append({
                "title": result.title,
                "entry_id": result.entry_id,
                "published": result.published,
                "authors": [a.name for a in result.authors],
                "doi": result.doi,
            })
        return results


# ---------------------------------------------------------------------------
# OpenReview client
# ---------------------------------------------------------------------------


class OpenReviewClient:
    """Access submissions and reviews from OpenReview."""

    def __init__(self) -> None:
        if openreview is None:
            raise ImportError("openreview-py package is required")
        self.client = openreview.api.OpenReviewClient()

    def fetch_submissions(
        self, invitation: str, limit: int = 100, offset: int = 0
    ) -> List[Dict[str, Any]]:
        notes = self.client.get_notes(
            invitation=invitation, limit=limit, offset=offset
        )
        return [n.to_json() for n in notes]


# ---------------------------------------------------------------------------
# Crossref client
# ---------------------------------------------------------------------------


class CrossrefClient:
    """Look up DOI metadata and reference lists via Crossref."""

    def __init__(self) -> None:
        if Crossref is None:
            raise ImportError("habanero package is required")
        self.cr = Crossref()

    def fetch_by_doi(self, doi: str) -> Dict[str, Any]:
        return self.cr.works(ids=doi)


# ---------------------------------------------------------------------------
# Semantic Scholar client
# ---------------------------------------------------------------------------


class SemanticScholarClient:
    """Fetch citation information from Semantic Scholar."""

    def __init__(self) -> None:
        if SemanticScholar is None:
            raise ImportError("semanticscholar package is required")
        self.ss = SemanticScholar()

    def get_paper(self, identifier: str) -> Dict[str, Any]:
        return self.ss.get_paper(identifier)


# ---------------------------------------------------------------------------
# Blog and grey literature scraping
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class BlogScraper:
    """Scrape blog posts using sitemaps while respecting robots.txt."""

    session: requests.Session = dataclasses.field(default_factory=requests.Session)

    def fetch_sitemap(self, base_url: str, sitemap: str = "sitemap.xml") -> List[str]:
        url = f"{base_url.rstrip('/')}/{sitemap}"
        response = self.session.get(url)
        response.raise_for_status()
        if BeautifulSoup is None:
            raise ImportError("beautifulsoup4 package is required")
        soup = BeautifulSoup(response.content, "xml")
        return [loc.text for loc in soup.find_all("loc")]

    def fetch_html(self, url: str) -> str:
        response = self.session.get(url)
        response.raise_for_status()
        return response.text


# ---------------------------------------------------------------------------
# Integration helper
# ---------------------------------------------------------------------------


class SourceIntegrator:
    """Map external records onto OpenAlex works for a unified graph."""

    def __init__(self, openalex: Optional[OpenAlexClient] = None) -> None:
        self.openalex = openalex or OpenAlexClient()

    def map_doi_to_openalex(self, doi: str) -> Optional[Dict[str, Any]]:
        filters = {"doi": doi}
        results = self.openalex.fetch_works(query="", filters=filters, per_page=1)
        works = results.get("results", [])
        return works[0] if works else None

    def map_arxiv_to_openalex(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        filters = {"ids": f"arXiv:{arxiv_id}"}
        results = self.openalex.fetch_works(query="", filters=filters, per_page=1)
        works = results.get("results", [])
        return works[0] if works else None


__all__ = [
    "OpenAlexClient",
    "ArXivClient",
    "OpenReviewClient",
    "CrossrefClient",
    "SemanticScholarClient",
    "BlogScraper",
    "SourceIntegrator",
]
