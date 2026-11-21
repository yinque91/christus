"""SELF-04 => integration platform tests (requires the local server on 8765)."""
from __future__ import annotations

import requests

BASE = "http://127.0.0.1:8765"


def test_health_and_index():
    assert requests.get(BASE + "/api/health", timeout=60).json()["ok"] is True
    r = requests.get(BASE + "/", timeout=30)
    assert r.status_code == 200
    assert "Christus" in r.text


def test_chat_persists_history():
    r = requests.post(BASE + "/api/chat", json={
        "text": "你好，这是平台测试消息", "user_id": "pytest-user"}, timeout=180)
    out = r.json()
    assert out["reply"]
    assert out["session_id"]
    h = requests.get(BASE + "/api/history",
                     params={"session_id": out["session_id"]}, timeout=30).json()
    assert any(m["role"] == "user" for m in h["messages"])
    assert any(m["role"] == "assistant" for m in h["messages"])


def test_subject_state_is_real():
    s = requests.get(BASE + "/api/state", timeout=60).json()
    assert s["subject_id"].startswith("subj_")
    assert "h_norm" in s
    assert "affect" in s
    assert "memory" in s
    assert s["language_source"]


def test_correction_and_learning_record():
    r = requests.post(BASE + "/api/chat", json={
        "text": "不对，我纠正一下：这不是事实。", "user_id": "pytest-user"}, timeout=180)
    out = r.json()
    assert "纠正" in out["reply"] or "correction" in out["reply"].lower()
    events = requests.get(BASE + "/api/learning", timeout=30).json()["events"]
    assert any(e["kind"] == "interaction" for e in events)


def test_checkpoint_endpoint():
    r = requests.post(BASE + "/api/checkpoint", json={}, timeout=120).json()
    assert r["checkpoint"]["sha256"]
    assert r["checkpoint"]["path"].endswith(".npz")
