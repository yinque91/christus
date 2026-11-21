"""Lexicon: word <-> cue binding by cross-situational co-occurrence.

A word is not attached to a concept by fiat. Binding strength grows only when a
word and a cue are observed in the same situation, and the strongest binding is
reported together with its evidence count. Unknown words stay unbound rather
than being mapped to something arbitrary.
"""

from __future__ import annotations

from typing import Any


class Lexicon:
    def __init__(self) -> None:
        # word -> cue -> {"count": int, "last_step": int}
        self.bindings: dict[str, dict[str, dict[str, int]]] = {}
        self.observations = 0
        self.unknown_words: set[str] = set()

    def observe(self, word: str, cue: str, *, step: int) -> None:
        word = word.strip().lower()
        if not word or cue in ("", "none"):
            self.unknown_words.add(word)
            return
        cues = self.bindings.setdefault(word, {})
        entry = cues.setdefault(cue, {"count": 0, "last_step": step})
        entry["count"] += 1
        entry["last_step"] = step
        self.observations += 1

    def word_for(self, cue: str) -> tuple[str | None, float]:
        best_word: str | None = None
        best_weight = 0.0
        total_weight = 0.0
        for word, cues in self.bindings.items():
            total_weight += sum(entry["count"] for entry in cues.values())
            entry = cues.get(cue)
            if entry and entry["count"] > best_weight:
                best_weight = float(entry["count"])
                best_word = word
        if best_word is None:
            return None, 0.0
        confidence = best_weight / total_weight if total_weight else 0.0
        return best_word, float(confidence)

    def cue_for(self, word: str) -> tuple[str | None, float]:
        cues = self.bindings.get(word.strip().lower())
        if not cues:
            return None, 0.0
        total = sum(entry["count"] for entry in cues.values())
        cue, entry = max(cues.items(), key=lambda kv: kv[1]["count"])
        return cue, float(entry["count"] / total) if total else 0.0

    def known_words(self) -> list[str]:
        return sorted(self.bindings)

    def stats(self) -> dict[str, Any]:
        pairs = sum(len(cues) for cues in self.bindings.values())
        return {
            "words": len(self.bindings),
            "pairs": pairs,
            "observations": self.observations,
            "unknown": len(self.unknown_words),
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "bindings": {w: {c: dict(e) for c, e in cues.items()} for w, cues in self.bindings.items()},
            "observations": self.observations,
            "unknown_words": sorted(self.unknown_words),
        }

    def restore(self, payload: dict[str, Any]) -> None:
        self.bindings = {
            str(word): {str(cue): dict(entry) for cue, entry in cues.items()}
            for word, cues in payload.get("bindings", {}).items()
        }
        self.observations = int(payload.get("observations", 0))
        self.unknown_words = set(payload.get("unknown_words", []))
