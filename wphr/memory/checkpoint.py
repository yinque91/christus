"""Checkpoint creation, verification and restore.

A checkpoint is a single ``.npz`` file (arrays + a JSON metadata block) plus a
row in the ``checkpoints`` table holding its SHA-256. Restore refuses to proceed
when the hash, schema version or cognitive config hash do not match, so a
silently corrupted or incompatible state cannot be resumed as if valid.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from wphr.cognition.smctrn import SparseModularCTRN
from wphr.config import Config
from wphr.core.registry import ModuleRegistry
from wphr.core.subject_state import SubjectState
from wphr.memory.store import Store, StoreError
from wphr.util.ids import new_id, utc_iso, utc_now
from wphr.version import SCHEMA_VERSION, __version__


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class Checkpoint:
    checkpoint_id: str
    path: Path
    step: int
    created_at: float
    sha256: str
    meta: dict[str, Any]
    arrays: dict[str, np.ndarray]

    def restore_into(
        self,
        core: SparseModularCTRN,
        registry: ModuleRegistry,
        subject: SubjectState,
        *,
        allow_config_mismatch: bool = False,
    ) -> None:
        core.load_state_arrays(self.arrays)
        registry.restore(self.meta.get("modules", {}))
        restored = SubjectState.from_dict(self.meta.get("subject", {}))
        subject.subject_id = restored.subject_id
        subject.created_at = restored.created_at
        subject.updated_at = restored.updated_at
        subject.step = restored.step
        subject.sim_time = restored.sim_time
        subject.episode_index = restored.episode_index
        subject.fault_count = restored.fault_count
        subject.module_states = restored.module_states
        subject.shared = restored.shared
        subject.rng_state = restored.rng_state

    def summary(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "path": str(self.path),
            "step": self.step,
            "created_at": utc_iso(self.created_at),
            "sha256": self.sha256,
            "size_bytes": self.path.stat().st_size if self.path.is_file() else 0,
            "note": self.meta.get("note", ""),
            "version": self.meta.get("version", ""),
        }


class CheckpointManager:
    def __init__(self, store: Store, config: Config, directory: Path | None = None) -> None:
        self.store = store
        self.config = config
        self.directory = Path(directory) if directory else config.resolve_path("paths.checkpoints")
        self.directory.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- create
    def create(
        self,
        *,
        subject: SubjectState,
        core: SparseModularCTRN,
        registry: ModuleRegistry,
        scheduler_stats: dict[str, Any] | None = None,
        bus_stats: dict[str, Any] | None = None,
        world_state: dict[str, Any] | None = None,
        extra_meta: dict[str, Any] | None = None,
        note: str = "",
    ) -> Checkpoint:
        checkpoint_id = f"ckpt_{int(utc_now())}_{subject.step:08d}_{new_id()[:8]}"
        path = self.directory / f"{checkpoint_id}.npz"
        meta = {
            "schema_version": SCHEMA_VERSION,
            "config_hash": self.config.hash(),
            "version": __version__,
            "spec": self.config.get("project.spec", ""),
            "created_at": utc_now(),
            "step": subject.step,
            "note": note,
            "subject": subject.to_dict(),
            "modules": registry.snapshot(),
            "scheduler": scheduler_stats or {},
            "bus": bus_stats or {},
            "world_state": world_state or {},
            "extra": extra_meta or {},
            "config": self.config.to_dict(),
        }
        arrays = core.state_arrays()
        tmp = path.with_suffix(".npz.tmp")
        with tmp.open("wb") as handle:
            np.savez_compressed(handle, **arrays, __meta__=np.array(json.dumps(meta, sort_keys=True)))
        tmp.replace(path)
        digest = sha256_file(path)
        self.store.record_checkpoint(
            checkpoint_id=checkpoint_id,
            step=subject.step,
            config_hash=meta["config_hash"],
            path=str(path),
            sha256=digest,
            size_bytes=path.stat().st_size,
            note=note,
        )
        return Checkpoint(
            checkpoint_id=checkpoint_id,
            path=path,
            step=subject.step,
            created_at=meta["created_at"],
            sha256=digest,
            meta=meta,
            arrays=arrays,
        )

    # ------------------------------------------------------------------ load
    def list(self, limit: int = 20) -> list[dict[str, Any]]:
        return self.store.list_checkpoints(limit=limit)

    def _row(self, checkpoint_id: str | None) -> dict[str, Any]:
        if checkpoint_id in (None, "", "latest"):
            row = self.store.latest_checkpoint()
            if row is None:
                raise StoreError("no checkpoint has been recorded in this database")
            return row
        row = self.store.get_checkpoint(checkpoint_id)
        if row is None:
            # allow a stem or a direct path
            candidate = self.directory / f"{checkpoint_id}.npz"
            if not candidate.is_file():
                candidate = Path(checkpoint_id)
            if candidate.is_file():
                return {
                    "checkpoint_id": candidate.stem,
                    "path": str(candidate),
                    "sha256": sha256_file(candidate),
                    "step": 0,
                    "created_at": utc_now(),
                    "schema_version": SCHEMA_VERSION,
                    "config_hash": self.config.hash(),
                }
            raise StoreError(f"unknown checkpoint: {checkpoint_id}")
        return row

    def load(
        self,
        checkpoint_id: str | None = None,
        *,
        verify: bool = True,
        allow_config_mismatch: bool = False,
    ) -> Checkpoint:
        row = self._row(checkpoint_id)
        path = Path(row["path"])
        if not path.is_file():
            raise StoreError(f"checkpoint file is missing: {path}")
        if verify:
            digest = sha256_file(path)
            if digest != row["sha256"]:
                raise StoreError(
                    f"checkpoint checksum mismatch for {path.name}: "
                    f"db={row['sha256'][:16]} file={digest[:16]}"
                )
        with np.load(path, allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files if name != "__meta__"}
            meta = json.loads(str(data["__meta__"].item()))
        if meta.get("schema_version") != SCHEMA_VERSION:
            raise StoreError(
                f"checkpoint schema v{meta.get('schema_version')} != code v{SCHEMA_VERSION}"
            )
        if not allow_config_mismatch and meta.get("config_hash") != self.config.hash():
            raise StoreError(
                "checkpoint was produced with a different cognitive configuration "
                f"(checkpoint={str(meta.get('config_hash'))[:16]}, current={self.config.hash()[:16]}); "
                "use --allow-config-mismatch only when you know the arrays are compatible"
            )
        return Checkpoint(
            checkpoint_id=str(row["checkpoint_id"]),
            path=path,
            step=int(meta.get("step", row.get("step", 0))),
            created_at=float(meta.get("created_at", row.get("created_at", utc_now()))),
            sha256=str(row["sha256"]),
            meta=meta,
            arrays=arrays,
        )

    # ---------------------------------------------------------------- backup
    def backup_database(self, tag: str = "manual") -> Path:
        stamp = utc_iso().replace(":", "").replace("-", "")
        target = self.config.resolve_path("paths.backups") / f"wphr_{tag}_{stamp}.db"
        return self.store.backup(target)

    def prune(self, keep: int = 10) -> list[Path]:
        """Delete oldest checkpoint files beyond ``keep``. Irreversible: caller confirms."""
        rows = self.store.list_checkpoints(limit=10000)
        removed: list[Path] = []
        for row in rows[keep:]:
            path = Path(row["path"])
            if path.is_file():
                path.unlink()
                removed.append(path)
        return removed
