"""Subject state aggregate.

Holds everything that must survive a process restart except the recurrent
weight/hidden arrays, which are owned by :mod:`wphr.cognition.smctrn` and
checkpointed alongside this object.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from wphr.util.ids import new_id, utc_now


@dataclass
class SubjectState:
    subject_id: str = field(default_factory=lambda: new_id("subj_"))
    created_at: float = field(default_factory=utc_now)
    updated_at: float = field(default_factory=utc_now)
    step: int = 0
    sim_time: float = 0.0
    episode_index: int = 0
    fault_count: int = 0
    module_states: dict[str, dict[str, Any]] = field(default_factory=dict)
    shared: dict[str, Any] = field(default_factory=dict)
    rng_state: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "step": self.step,
            "sim_time": self.sim_time,
            "episode_index": self.episode_index,
            "fault_count": self.fault_count,
            "module_states": copy.deepcopy(self.module_states),
            "shared": copy.deepcopy(self.shared),
            "rng_state": self.rng_state,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SubjectState":
        return cls(
            subject_id=str(data.get("subject_id") or new_id("subj_")),
            created_at=float(data.get("created_at", utc_now())),
            updated_at=float(data.get("updated_at", utc_now())),
            step=int(data.get("step", 0)),
            sim_time=float(data.get("sim_time", 0.0)),
            episode_index=int(data.get("episode_index", 0)),
            fault_count=int(data.get("fault_count", 0)),
            module_states=copy.deepcopy(data.get("module_states", {})),
            shared=copy.deepcopy(data.get("shared", {})),
            rng_state=str(data.get("rng_state", "")),
        )
