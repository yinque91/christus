"""Export a subject (state + manifest + version) to a portable archive."""

from __future__ import annotations

import hashlib
import json
import sys
import tarfile
import time
from pathlib import Path

WS = Path(r"C:\Users\13394\Documents\deepseek-harness\default-workspace\WPHR_workspace")
sys.path.insert(0, r"C:\Users\13394\Documents\deepseek-harness\default-workspace\WPHR")
from wphr.persistence.store import COMPONENTS, VERSION  # noqa: E402


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def export(out_path: str) -> dict:
    src = WS / "subject"
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"version": VERSION, "exported_at": time.time(),
            "files": {}, "note": "no device identity; subject_id lives in state"}
    for name in COMPONENTS:
        p = src / f"{name}.json"
        if p.exists():
            meta["files"][name] = {"sha256": sha(p), "bytes": p.stat().st_size}
    meta_path = out.parent / "export_manifest.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    with tarfile.open(out, "w:gz") as tar:
        tar.add(src, arcname="subject")
        tar.add(meta_path, arcname="export_manifest.json")
    return {"ok": True, "path": str(out), "bytes": out.stat().st_size,
            "components": len(meta["files"]), "version": VERSION}


if __name__ == "__main__":
    print(json.dumps(export(sys.argv[1] if len(sys.argv) > 1 else str(WS / "export.tar.gz")),
                     ensure_ascii=False))
