"""Fetch a small licensed internet corpus for local auxiliary training.

Only public Wikipedia extracts. The corpus is stored with provenance; it is not
autobiographical experience and does not modify Subject W/P/Q.
"""
from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "christus_desktop" / "training" / "corpus.jsonl"
META = ROOT / "christus_desktop" / "training" / "corpus_manifest.json"
UA = "ChristusLocalSubject/1.0 (local research; contact local user)"

TOPICS_ZH = [
    "璇█", "绀句細", "鍚堜綔", "淇′换", "浜洪檯鍏崇郴", "鎯呯华", "蹇冪悊瀛?,
    "鍝插", "鏂囧寲", "绀句細鍖?, "浼︾悊瀛?, "瀵硅瘽", "瀛︿範", "璁板繂",
    "浜哄伐鏅鸿兘", "璁ょ煡绉戝", "缁寸壒鏍规柉鍧?, "娴峰痉鏍煎皵",
]
TOPICS_EN = [
    "Language", "Society", "Cooperation", "Trust (social science)",
    "Interpersonal relationship", "Emotion", "Psychology", "Philosophy",
    "Culture", "Socialization", "Ethics", "Conversation", "Learning",
    "Memory", "Artificial intelligence", "Cognitive science",
]


def fetch(lang: str, title: str):
    api = f"https://{lang}.wikipedia.org/w/api.php"
    r = requests.get(api, params={
        "action": "query", "prop": "extracts", "explaintext": 1,
        "titles": title, "format": "json", "redirects": 1,
    }, headers={"User-Agent": UA}, timeout=12)
    r.raise_for_status()
    pages = r.json().get("query", {}).get("pages", {})
    for _, page in pages.items():
        text = (page.get("extract") or "").strip()
        if not text:
            continue
        real_title = page.get("title", title)
        return {
            "title": real_title,
            "url": f"https://{lang}.wikipedia.org/wiki/{real_title.replace(' ', '_')}",
            "license": "CC BY-SA 4.0",
            "text": text[:12000],
            "lang": lang,
            "retrieved_at": time.time(),
        }
    return None


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    errors = []
    jobs = [("zh", t) for t in TOPICS_ZH] + [("en", t) for t in TOPICS_EN]
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(fetch, lang, title): (lang, title) for lang, title in jobs}
        for fut in as_completed(futures):
            lang, title = futures[fut]
            try:
                item = fut.result()
                if item:
                    rows.append(item)
                    print("ok", lang, item["title"], len(item["text"]), flush=True)
                else:
                    errors.append(f"{lang}:{title}: empty")
            except Exception as exc:
                errors.append(f"{lang}:{title}: {type(exc).__name__}: {exc}")
    # dedupe by URL/text hash
    seen = set()
    deduped = []
    for r in rows:
        h = hashlib.sha256((r["url"] + r["text"]).encode("utf-8")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        r["content_hash"] = h
        deduped.append(r)
    with OUT.open("w", encoding="utf-8") as f:
        for r in deduped:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {
        "source": "Wikipedia API",
        "license": "CC BY-SA 4.0",
        "retrieved_at": time.time(),
        "documents": len(deduped),
        "chars": sum(len(r["text"]) for r in deduped),
        "errors": errors,
        "provenance_file": str(OUT),
    }
    META.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

