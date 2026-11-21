"""Build a local training corpus from already-retrieved internet knowledge and
licensed local dialogue. No new uncontrolled crawl."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "christus_platform" / "state" / "platform.db"
OUT = ROOT / "christus_desktop" / "training" / "corpus.jsonl"
MANIFEST = ROOT / "christus_desktop" / "training" / "corpus_manifest.json"


def main():
    rows = []
    if DB.exists():
        con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        try:
            for r in con.execute("SELECT title,url,license,summary,snippet,query,retrieved_at FROM knowledge"):
                text = (r["summary"] or "") + "\n" + (r["snippet"] or "")
                if len(text.strip()) >= 80:
                    rows.append({
                        "kind": "internet_knowledge", "title": r["title"],
                        "url": r["url"], "license": r["license"],
                        "text": text[:12000], "retrieved_at": r["retrieved_at"],
                        "provenance": "existing Wikipedia retrieval; external knowledge",
                    })
        finally:
            con.close()
    dialogue = ROOT / "CHRISTUS-DATA-01" / "datasets" / "processed" / "dailydialog_v1" / "train.jsonl"
    if dialogue.exists():
        n = 0
        for line in dialogue.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            text = " ".join(t["text"] for t in d["turns"])
            if len(text) > 100:
                rows.append({
                    "kind": "licensed_dialogue", "title": d["dialogue_id"],
                    "url": "local:DailyDialog", "license": "CC BY-NC-SA 4.0",
                    "text": text[:4000], "retrieved_at": None,
                    "provenance": "licensed dialogue sample; not personal experience",
                })
                n += 1
                if n >= 120:
                    break
    # local integration documents: language/social vocabulary, not internet.
    for p in sorted(ROOT.glob("CHRISTUS_*REPORT.md"))[:4]:
        rows.append({"kind": "local_document", "title": p.name,
                     "url": "local:" + p.name, "license": "project-internal",
                     "text": p.read_text(encoding="utf-8", errors="ignore")[:8000],
                     "retrieved_at": None,
                     "provenance": "local research document"})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {
        "documents": len(rows),
        "internet_documents": sum(1 for r in rows if r["kind"] == "internet_knowledge"),
        "licensed_dialogue_documents": sum(1 for r in rows if r["kind"] == "licensed_dialogue"),
        "local_documents": sum(1 for r in rows if r["kind"] == "local_document"),
        "chars": sum(len(r["text"]) for r in rows),
        "licenses": sorted({r["license"] for r in rows}),
        "note": "Internet documents were previously retrieved and stored with URL/license; no new uncontrolled crawl.",
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
