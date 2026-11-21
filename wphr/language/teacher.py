"""A scripted teacher that produces language claims.

The teacher is an *external* agent, not part of the cognitive core: it is
constructed by the experiment with the rule it intends to teach, and it never
touches the subject's state. Its claims arrive as ``language_claim`` events with
an explicit reliability, and the subject may disbelieve them after contrary
evidence (MASTER SPEC 8.3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TeacherConfig:
    enabled: bool = False
    teach_every: int = 25
    reliability: float = 0.7
    vocab: dict[str, str] = field(
        default_factory=lambda: {"switch_blue": "blue", "switch_orange": "orange"}
    )
    honesty: float = 1.0


class Teacher:
    def __init__(self, config: TeacherConfig, safe_color: str) -> None:
        if safe_color not in ("blue", "orange"):
            raise ValueError(f"unknown safe colour: {safe_color!r}")
        self.config = config
        self.safe_color = safe_color
        self.claims = 0
        self._last_claim_step = -10**9

    def set_rule(self, safe_color: str) -> None:
        if safe_color not in ("blue", "orange"):
            raise ValueError(f"unknown safe colour: {safe_color!r}")
        self.safe_color = safe_color

    def claim(self, faced_cue: str, step: int) -> dict[str, Any] | None:
        """Speak at most once per ``teach_every`` steps, whenever able.

        The first implementation only spoke when ``step % teach_every == 0``
        *and* the subject happened to be facing a switch, which produced 0–2
        claims per 1000 steps and made the language condition vacuous. The
        teacher now speaks at the first opportunity after the interval.
        """
        if not self.config.enabled:
            return None
        if step - self._last_claim_step < max(1, self.config.teach_every):
            return None
        word = self.config.vocab.get(faced_cue)
        if word is None:
            return None
        self._last_claim_step = step
        is_safe = word == self.safe_color
        # an honest teacher always tells the truth about the current rule;
        # honesty < 1 models an unreliable informant
        if self.config.honesty < 1.0 and (step * 7919) % 1000 / 1000.0 > self.config.honesty:
            is_safe = not is_safe
        polarity = 1.0 if is_safe else -1.0
        self.claims += 1
        return {
            "text": f"the {word} switch is {'safe' if is_safe else 'dangerous'}",
            "words": [word],
            "polarity": polarity,
            "reliability": self.config.reliability,
            "speaker": "teacher",
            "step": step,
        }
