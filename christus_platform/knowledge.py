"""Internet knowledge interface with provenance and deduplication.

Default source: Wikipedia public API (CC BY-SA). No scraping behind logins, no
unbounded crawl, no API keys required. Network failures are reported, not hidden.
"""
from __future__ import annotations

import hashlib
import html
import re
from urllib.parse import quote

import requests

from .store import PlatformStore

USER_AGENT = "ChristusResearchPlatform/1.0 (local research; contact local user)"


def clean_query(text: str) -> str:
    q = text.strip()
    q = re.sub(r"^(请|帮我|麻烦)?(搜索|查一下|查找|查|搜一下|search|look up)\s*", "", q,
               flags=re.I)
    q = re.sub(r"(是什么|是谁|怎么样|吗|呢|？|\?)$", "", q).strip()
    return q[:120] or text[:120]


def _strip_html(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s or "")
    return html.unescape(re.sub(r"\s+", " ", s)).strip()


class KnowledgeService:
    def __init__(self, store: PlatformStore, timeout: float = 8.0):
        self.store = store
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def _cached(self, query: str, limit: int = 3) -> dict | None:
        cached = self.store.knowledge(100)
        if not cached:
            return None
        q = query.lower()
        ranked = sorted(cached, key=lambda r: (
            q in (str(r.get("title", "")) + " " + str(r.get("summary", ""))).lower(),
            float(r.get("retrieved_at") or 0)), reverse=True)
        items = [{"title": r.get("title"), "url": r.get("url"),
                  "summary": r.get("summary") or r.get("snippet") or "",
                  "snippet": r.get("snippet") or "",
                  "license": r.get("license") or "source terms apply",
                  "trust": (r.get("trust") or "") + " | cached local index",
                  "lang": "cached"} for r in ranked[:limit] if r.get("summary")]
        if not items:
            return None
        return {"ok": True, "query": query, "items": items, "source": "local-index",
                "retrieved_at": __import__("time").time()}

    def search(self, query: str, limit: int = 3, cached_first: bool = False) -> dict:
        query = clean_query(query)
        if cached_first:
            hit = self._cached(query, limit)
            if hit:
                return hit
        errors = []
        for lang in ("zh", "en"):
            try:
                result = self._wikipedia(query, lang, limit)
                if result.get("ok"):
                    return result
            except Exception as exc:
                errors.append(f"{lang}: {type(exc).__name__}: {exc}")
        try:
            result = self._duckduckgo(query)
            if result.get("ok"):
                return result
        except Exception as exc:
            errors.append(f"ddg: {type(exc).__name__}: {exc}")
        hit = self._cached(query, limit)
        if hit:
            hit["error"] = "; ".join(errors)
            return hit
        return {"ok": False, "query": query, "error": "; ".join(errors),
                "items": [], "source": "none"}

    def _wikipedia(self, query: str, lang: str, limit: int) -> dict:
        api = f"https://{lang}.wikipedia.org/w/api.php"
        params = {"action": "query", "list": "search", "srsearch": query,
                  "format": "json", "utf8": 1, "srlimit": limit}
        r = self.session.get(api, params=params, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        hits = data.get("query", {}).get("search", [])
        if not hits:
            return {"ok": False, "query": query, "items": []}
        items = []
        for hit in hits[:limit]:
            title = hit.get("title", "")
            url = f"https://{lang}.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
            summary = ""
            try:
                s = self.session.get(
                    f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{quote(title.replace(' ', '_'))}",
                    timeout=self.timeout)
                if s.ok:
                    summary = (s.json().get("extract") or "").strip()
            except Exception:
                summary = _strip_html(hit.get("snippet", ""))
            if not summary:
                summary = _strip_html(hit.get("snippet", ""))
            item = {"title": title, "url": url, "summary": summary[:1800],
                    "snippet": _strip_html(hit.get("snippet", ""))[:500],
                    "license": "CC BY-SA 4.0", "trust": "encyclopedia; cross-check needed",
                    "lang": lang}
            self.store.add_knowledge(
                query=query, title=title, url=url, license=item["license"],
                summary=item["summary"], snippet=item["snippet"], trust=item["trust"],
                content_hash=hashlib.sha256((url + summary).encode("utf-8")).hexdigest())
            items.append(item)
        return {"ok": True, "query": query, "items": items, "source": "wikipedia",
                "retrieved_at": __import__("time").time()}

    def _duckduckgo(self, query: str) -> dict:
        r = self.session.get("https://api.duckduckgo.com/",
                             params={"q": query, "format": "json", "no_html": 1},
                             timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        abstract = (data.get("AbstractText") or "").strip()
        if not abstract:
            return {"ok": False, "query": query, "items": []}
        url = data.get("AbstractURL") or "https://duckduckgo.com/"
        item = {"title": data.get("Heading") or query, "url": url,
                "summary": abstract[:1800], "snippet": abstract[:500],
                "license": "source terms apply", "trust": "search abstract; cross-check needed",
                "lang": "en"}
        self.store.add_knowledge(
            query=query, title=item["title"], url=url, license=item["license"],
            summary=item["summary"], snippet=item["snippet"], trust=item["trust"],
            content_hash=hashlib.sha256((url + abstract).encode("utf-8")).hexdigest())
        return {"ok": True, "query": query, "items": [item], "source": "duckduckgo",
                "retrieved_at": __import__("time").time()}
