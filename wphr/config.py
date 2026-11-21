"""Configuration system.

All dimensions, rates and limits live in ``configs/*.json``; nothing numerical
is hard-coded across the code base. The config hash is used to verify that a
checkpoint is restored against the same cognitive configuration.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from wphr.version import CONFIG_VERSION

_MISSING = object()
# Paths are environment-specific and must not invalidate a checkpoint.
_UNHASHED_TOP_LEVEL = ("paths",)


class ConfigError(ValueError):
    """Raised when a configuration file is missing, malformed or inconsistent."""


def project_root() -> Path:
    """Locate the WPHR project root (directory holding ``pyproject.toml``)."""
    env = os.environ.get("WPHR_HOME")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "pyproject.toml").is_file() and (parent / "configs").is_dir():
            return parent
    return Path.cwd()


def _parse_override_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


class Config:
    """Immutable-by-convention nested configuration with dotted access."""

    def __init__(
        self,
        data: Mapping[str, Any],
        path: Path | None = None,
        overrides: list[str] | None = None,
    ) -> None:
        self._data: dict[str, Any] = copy.deepcopy(dict(data))
        self.path = Path(path) if path else None
        self.overrides: list[str] = list(overrides or [])

    # ---------------------------------------------------------------- loading
    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        overrides: list[str] | None = None,
        root: str | Path | None = None,
    ) -> "Config":
        base = Path(root) if root else project_root()
        target = Path(path) if path else base / "configs" / "default.json"
        if not target.is_absolute():
            target = base / target
        if not target.is_file():
            raise ConfigError(f"config file not found: {target}")
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigError(f"config file is not valid JSON ({target}): {exc}") from exc
        if not isinstance(data, dict):
            raise ConfigError(f"config root must be an object: {target}")
        cfg = cls(data, target, overrides)
        for spec in overrides or []:
            cfg.apply_override(spec)
        cfg.validate()
        return cfg

    def apply_override(self, spec: str) -> None:
        if "=" not in spec:
            raise ConfigError(f"override must look like dotted.key=value, got: {spec!r}")
        key, raw = spec.split("=", 1)
        key = key.strip()
        if not key:
            raise ConfigError(f"override has an empty key: {spec!r}")
        self.set(key, _parse_override_value(raw.strip()))
        self.overrides.append(spec)

    # ---------------------------------------------------------------- access
    def get(self, dotted: str, default: Any = _MISSING) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                if default is _MISSING:
                    raise ConfigError(f"missing config key: {dotted}")
                return default
        return copy.deepcopy(node) if isinstance(node, (dict, list)) else node

    require = get

    def set(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node = self._data
        for part in parts[:-1]:
            nxt = node.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                node[part] = nxt
            node = nxt
        node[parts[-1]] = value

    def section(self, dotted: str) -> dict[str, Any]:
        value = self.get(dotted)
        if not isinstance(value, dict):
            raise ConfigError(f"config key {dotted} is not a section")
        return value

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    # ---------------------------------------------------------------- derived
    def module_names(self) -> list[str]:
        return list(self.section("core.modules").keys())

    def hash(self) -> str:
        payload = {
            k: v for k, v in self._data.items() if k not in _UNHASHED_TOP_LEVEL
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def resolve_path(self, dotted: str) -> Path:
        raw = self.get(dotted)
        p = Path(str(raw))
        if not p.is_absolute():
            p = project_root() / p
        return p

    def ensure_dirs(self) -> list[Path]:
        created: list[Path] = []
        for key in (
            "paths.storage",
            "paths.checkpoints",
            "paths.backups",
            "paths.logs",
            "paths.data",
            "paths.results",
        ):
            p = self.resolve_path(key)
            p.mkdir(parents=True, exist_ok=True)
            created.append(p)
        return created

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path else (self.path or project_root() / "configs" / "default.json")
        if not target.is_absolute():
            target = project_root() / target
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return target

    # ---------------------------------------------------------------- checks
    def validate(self) -> None:
        def fail(msg: str) -> None:
            raise ConfigError(msg)

        if self.get("schema_version", None) != CONFIG_VERSION:
            fail(
                f"schema_version mismatch: file={self.get('schema_version', None)!r} "
                f"code={CONFIG_VERSION}"
            )

        runtime = self.get("runtime", None)
        if not isinstance(runtime, dict):
            fail("missing runtime section")
        if not isinstance(runtime.get("seed"), int) or runtime["seed"] < 0:
            fail("runtime.seed must be a non-negative integer")
        dt = runtime.get("dt")
        if not isinstance(dt, (int, float)) or dt <= 0:
            fail("runtime.dt must be a positive number")
        if runtime.get("finite_policy") not in ("raise", "sanitize"):
            fail("runtime.finite_policy must be 'raise' or 'sanitize'")

        core = self.get("core", None)
        if not isinstance(core, dict):
            fail("missing core section")
        sparsity = core.get("sparsity")
        if not isinstance(sparsity, (int, float)) or not (0.0 < sparsity <= 1.0):
            fail("core.sparsity must be in (0, 1]")
        concept_dim = core.get("concept_dim")
        if not isinstance(concept_dim, int) or concept_dim < 8:
            fail("core.concept_dim must be an integer >= 8")
        if core.get("activation") not in ("tanh", "clipped_linear"):
            fail("core.activation must be 'tanh' or 'clipped_linear'")
        for key in ("weight_scale", "weight_bound", "state_bound", "concept_tau"):
            if not isinstance(core.get(key), (int, float)) or core[key] <= 0:
                fail(f"core.{key} must be a positive number")

        modules = core.get("modules")
        if not isinstance(modules, dict) or len(modules) < 7:
            fail("core.modules must declare at least the seven logical modules")
        for name, spec in modules.items():
            if not isinstance(spec, dict):
                fail(f"core.modules.{name} must be an object")
            units = spec.get("units")
            if not isinstance(units, int) or units < 4:
                fail(f"core.modules.{name}.units must be an integer >= 4")
            tau = spec.get("tau")
            if not isinstance(tau, (int, float)) or tau <= 0:
                fail(f"core.modules.{name}.tau must be positive")
            if not isinstance(spec.get("input_dim"), int) or spec["input_dim"] < 1:
                fail(f"core.modules.{name}.input_dim must be a positive integer")
            if dt / tau > 0.5:
                fail(
                    f"unstable integration: dt/tau={dt / tau:.3f} > 0.5 for module {name}"
                )

        sched = self.get("scheduler", None)
        if not isinstance(sched, dict) or not sched:
            fail("missing scheduler section")
        for group, period in sched.items():
            if not isinstance(period, int) or period < 1:
                fail(f"scheduler.{group} must be a positive integer period")

        safety = self.get("safety", None)
        if not isinstance(safety, dict):
            fail("missing safety section")
        for flag in (
            "real_camera",
            "real_microphone",
            "screen_capture",
            "keyboard_capture",
            "network_access",
            "private_files",
            "robot_actuators",
            "paid_services",
        ):
            if not isinstance(safety.get(flag), bool):
                fail(f"safety.{flag} must be a boolean")
