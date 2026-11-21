"""Unified Christus runtime for the local web platform.

One Subject instance (DataSubject -> existing Subject/SM-CTRN) handles every
message. The platform store holds conversations/knowledge; the Subject holds
core, memory, concept, affect and checkpoints.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT / "CHRISTUS-DATA-01"
for p in (PROJECT, DATA_DIR, DATA_DIR / "preprocessing",
          DATA_DIR / "cognition_adapter"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from cognition_adapter.adapter import DataAdapter, core_weights_hash  # noqa: E402
from preprocessing.text_features import (  # noqa: E402
    encode, intent_from_text, emotion_from_text)

from .knowledge import KnowledgeService
from .language import LocalLanguage, ExternalLanguageBridge, is_knowledge_query
from .store import PlatformStore
from .transformer_semantic import TransformerSemantic

PLATFORM = Path(__file__).resolve().parent
STATE = PLATFORM / "state"
LOGS = PLATFORM / "logs"


class PlatformRuntime:
    def __init__(self):
        STATE.mkdir(parents=True, exist_ok=True)
        LOGS.mkdir(parents=True, exist_ok=True)
        self.store = PlatformStore(STATE / "platform.db")
        self.knowledge = KnowledgeService(self.store)
        self.semantic = TransformerSemantic(
            PROJECT / "christus_platform" / "models" / "local_transformer.npz")
        self.local_language = LocalLanguage()
        self.external_language = ExternalLanguageBridge()
        self.language = self.external_language if self.external_language.enabled else self.local_language
        self.lock = threading.RLock()
        self.started_at = time.time()
        self.last_checkpoint = None
        self.paused = False
        self.turns = self.store.message_count() // 2
        self.adapter = DataAdapter(STATE / "subject", persist=True)
        try:
            ck = self.adapter.subject.restore()
            self.last_checkpoint = {"path": str(getattr(ck, "path", "")),
                                    "sha256": getattr(ck, "sha256", ""),
                                    "step": self.adapter.subject.clock.step}
        except Exception:
            # A fresh subject has no checkpoint; this is expected.
            pass
        self.state_path = STATE / "runtime_state.json"
        self.self_model = self._read_self_model()
        self._write_state()

    def _read_self_model(self):
        p = PROJECT / "self03" / "results.json"
        if not p.exists():
            return {"available": False}
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
            s = r.get("summary", {}).get("A3", {})
            return {"available": True, "mode": "A3 offline checkpoint",
                    "source_probe": s.get("source_self_mean"),
                    "temporal": s.get("temporal_mean"),
                    "note": "offline SELF-03 model; not on the language path"}
        except Exception as exc:
            return {"available": False, "error": str(exc)}

    def _state_snapshot(self, row=None):
        subj = self.adapter.subject
        row = row or {}
        memory = subj.memory.stats()
        try:
            concept = subj.registry.get("concept_language").metrics()
        except Exception:
            concept = {}
        try:
            subject_id = str(subj.subject_state.subject_id)
        except Exception:
            subject_id = "unknown"
        try:
            h_norm = float(__import__("numpy").linalg.norm(subj.core.h))
        except Exception:
            h_norm = 0.0
        return {
            "subject_id": subject_id,
            "step": int(getattr(subj.clock, "step", 0)),
            "turns": int(self.turns),
            "h_norm": round(h_norm, 6),
            "action": row.get("action", "wait"),
            "reward": float(row.get("reward", 0.0)),
            "affect": {k: round(float(v), 6) for k, v in subj.affect_values().items()},
            "memory": memory,
            "concept": concept,
            "self_model": self.self_model,
            "transformer": self.semantic.status(),
            "language_source": self.language.name,
            "paused": self.paused,
            "started_at": self.started_at,
            "last_checkpoint": self.last_checkpoint,
            "core_weights_unchanged": self.adapter.core_weights_before ==
            core_weights_hash(subj.core),
            "w_p_q_note": "W/P/Q not updated by the platform; DataAdapter only reads core state.",
        }

    def _write_state(self):
        state = self._state_snapshot()
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.state_path)
        return state

    def public_state(self):
        with self.lock:
            return self._state_snapshot()

    def _profile(self, user_id):
        return self.store.get_profile(user_id)

    def _extract_name(self, text, user_id):
        m = re.search(r"(?:我叫|我的名字是|叫我)\s*([\u4e00-\u9fffA-Za-z0-9_\-]{1,24})", text)
        if not m:
            m = re.search(r"(?:my name is|call me)\s+([A-Za-z0-9_\-]{1,24})", text, re.I)
        if m:
            name = m.group(1).strip()
            self.store.set_profile(user_id, name=name)
            return name
        return None

    def chat(self, text: str, *, session_id: str | None, user_id: str):
        text = (text or "").strip()
        if not text:
            raise ValueError("empty message")
        with self.lock:
            if self.paused:
                return {"reply": "学习/对话当前处于暂停状态。", "session_id": session_id,
                        "source": "system", "citations": [], "state": self._state_snapshot()}
            sid = self.store.ensure_session(session_id, user_id)
            history = self.store.recent_context(sid, 10)
            self._extract_name(text, user_id)
            profile = self._profile(user_id)
            kq = is_knowledge_query(text)
            knowledge = None
            if kq:
                knowledge = self.knowledge.search(text, cached_first=True)
                if knowledge and knowledge.get("ok"):
                    knowledge["items"] = self.semantic.rerank(
                        str(knowledge.get("query", text)), knowledge.get("items", []))
                    knowledge["reranked_by"] = "local-transformer" if self.semantic.available else "none"
            intent = intent_from_text(text)
            emotion = emotion_from_text(text)
            context = " ".join(m["text"] for m in history[-4:])
            row = None
            try:
                row = self.adapter.step_turn(
                    {"speaker": "user", "text": text, "intent": intent,
                     "emotion": emotion}, encode(text, context))
            except Exception as exc:
                row = {"error": f"{type(exc).__name__}: {exc}", "action": "wait",
                       "reward": 0.0}
            state = self._state_snapshot(row)
            try:
                reply, citations = self.language.respond(
                    text, history=history, state=state, knowledge=knowledge,
                    profile=profile)
            except Exception as exc:
                reply = (f"语言接口暂时失败（{type(exc).__name__}: {exc}）。"
                         "主体状态已保留；我不会伪造一个回答。")
                citations = []
            source = ("external-language-bridge" if self.language is self.external_language
                      else ("knowledge+local-language" if knowledge and knowledge.get("ok")
                            else "local-language"))
            self.store.add_message(sid, "user", text, "human", state)
            self.store.add_message(sid, "assistant", reply, source, state, citations)
            self.store.add_learning(sid, "interaction", {
                "text": text, "reply": reply, "source": source,
                "intent": intent, "emotion": emotion,
                "citations": citations, "step": state.get("step"),
            })
            if kq:
                self.store.add_learning(sid, "internet", {
                    "query": knowledge.get("query") if knowledge else text,
                    "ok": bool(knowledge and knowledge.get("ok")),
                    "items": knowledge.get("items", []) if knowledge else [],
                    "error": knowledge.get("error") if knowledge else None,
                })
            self.turns += 1
            if self.turns % 2 == 0:
                try:
                    ck = self.adapter.subject.save_checkpoint("platform exchange")
                    self.last_checkpoint = {"path": str(ck.path), "sha256": ck.sha256,
                                            "step": state.get("step")}
                except Exception as exc:
                    self.last_checkpoint = {"error": str(exc)}
            self._write_state()
            return {"reply": reply, "session_id": sid, "source": source,
                    "citations": citations, "state": self._state_snapshot(row),
                    "knowledge": {"ok": bool(knowledge and knowledge.get("ok")),
                                  "items": knowledge.get("items", []) if knowledge else [],
                                  "error": knowledge.get("error") if knowledge else None}}

    def history(self, session_id):
        return self.store.history(session_id)

    def sessions(self, user_id):
        return self.store.sessions(user_id)

    def learning(self):
        return self.store.learning()

    def knowledge_list(self):
        return self.store.knowledge()

    def pause(self, value=True):
        self.paused = bool(value)
        self._write_state()
        return self.paused

    def checkpoint(self):
        with self.lock:
            ck = self.adapter.subject.save_checkpoint("manual platform checkpoint")
            self.last_checkpoint = {"path": str(ck.path), "sha256": ck.sha256}
            self._write_state()
            return self.last_checkpoint

    def close(self):
        try:
            self.adapter.subject.save_checkpoint("platform shutdown")
        except Exception:
            pass
        try:
            self.adapter.close()
        except Exception:
            pass
        self.store.close()
