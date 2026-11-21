"""CHRISTUS-LOCAL-01 desktop tests (no browser)."""
from __future__ import annotations

import json
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_required_scripts_exist():
    for name in ("START_CHRISTUS.bat", "STOP_CHRISTUS.bat", "BACKUP_CHRISTUS.bat",
                 "RESTORE_CHRISTUS.bat", "CHECK_CHRISTUS.bat"):
        assert (ROOT / name).exists(), name


def test_desktop_app_source_is_native_tk():
    src = (ROOT / "christus_desktop" / "app.py").read_text(encoding="utf-8")
    assert "tkinter" in src
    assert "http.server" not in src
    assert "localhost" not in src.lower()


def test_transformer_checkpoint_and_metrics_exist():
    m = ROOT / "christus_platform" / "models" / "local_transformer.npz"
    metrics = ROOT / "christus_platform" / "models" / "local_transformer_metrics.json"
    assert m.exists() and metrics.exists()
    d = json.loads(metrics.read_text(encoding="utf-8"))
    assert d["final_loss"] < d["initial_loss"]
    assert d["internet_documents"] >= 1
    assert d["params"] > 1000


def test_runtime_has_one_subject_and_transformer_status():
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "CHRISTUS-DATA-01"))
    from christus_platform.runtime import PlatformRuntime
    rt = PlatformRuntime()
    try:
        s = rt.public_state()
        assert s["subject_id"].startswith("subj_")
        assert s["transformer"]["available"] is True
        assert s["transformer"]["final_loss"] is not None
    finally:
        rt.close()


def test_no_web_listener():
    s = socket.socket()
    s.settimeout(1)
    try:
        s.connect(("127.0.0.1", 8765))
        connected = True
    except Exception:
        connected = False
    finally:
        s.close()
    assert not connected, "A web server is listening; desktop must not depend on it"
