"""Import a subject archive with integrity/version checks and rollback."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
import time
from pathlib import Path

WS = Path(r"C:\Users\13394\Documents\deepseek-harness\default-workspace\WPHR_workspace")
sys.path.insert(0, r"C:\Users\13394\Documents\deepseek-harness\default-workspace\WPHR")
from wphr.persistence.store import VERSION, SubjectStore  # noqa: E402


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def import_subject(archive: str, *, force: bool = False) -> dict:
    arc = Path(archive)
    if not arc.exists():
        return {"ok": False, "reason": "archive_missing"}
    with tarfile.open(arc, "r:gz") as tar:
        names = tar.getnames()
        if "export_manifest.json" not in names:
            return {"ok": False, "reason": "manifest_missing"}
        tmp = Path(tempfile.mkdtemp(prefix="wphr-import-"))
        tar.extractall(tmp)
        man = json.loads((tmp / "export_manifest.json").read_text(encoding="utf-8"))
        if int(man.get("version", 0)) > VERSION:
            shutil.rmtree(tmp, ignore_errors=True)
            return {"ok": False, "reason": "version_too_new", "archive": man.get("version"),
                    "local": VERSION}
        bad = [n for n, m in man.get("files", {}).items()
               if (tmp / "subject" / f"{n}.json").exists()
               and sha(tmp / "subject" / f"{n}.json") != m["sha256"]]
        if bad and not force:
            shutil.rmtree(tmp, ignore_errors=True)
            return {"ok": False, "reason": "integrity_failed", "bad": bad}
        dst = WS / "subject"
        stamp = time.strftime("%Y%m%d-%H%M%S")
        rollback = WS / "subject_rollback" / stamp
        if dst.exists():
            shutil.copytree(dst, rollback)
        store = SubjectStore(dst)
        store.backup("pre-import-" + stamp)
        dst.mkdir(parents=True, exist_ok=True)
        for name in man.get("files", {}):
            src = tmp / "subject" / f"{name}.json"
            if src.exists():
                shutil.copy2(src, dst / f"{name}.json")
        ver = store.verify()
        shutil.rmtree(tmp, ignore_errors=True)
        if not ver["ok"]:
            if rollback.exists():
                for f in rollback.iterdir():
                    shutil.copy2(f, dst / f.name)
            return {"ok": False, "reason": "post_import_verify_failed", "rollback": str(rollback)}
        return {"ok": True, "components": len(man.get("files", {})),
                "version": man.get("version"), "rollback": str(rollback) if rollback.exists() else None}


if __name__ == "__main__":
    print(json.dumps(import_subject(sys.argv[1]), ensure_ascii=False))
