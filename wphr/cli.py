"""WPHR command line interface.

Commands implemented in V0.1: ``doctor``, ``init``, ``run``, ``pause``, ``stop``,
``status``, ``memory``, ``checkpoint`` and ``report``. ``experiment`` and ``ui``
are declared but report NOT_IMPLEMENTED until their stage is built; the CLI never
pretends a command worked.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

from wphr.config import Config, ConfigError, project_root
from wphr.memory.store import Store
from wphr.util.ids import utc_iso, utc_now
from wphr.version import SPEC_VERSION, __version__

MIN_PYTHON = (3, 11)
MAX_PYTHON = (3, 12)
MIN_DISK_GB = 1.0


# --------------------------------------------------------------------- helpers
def load_config(args: argparse.Namespace) -> Config:
    return Config.load(getattr(args, "config", None), overrides=getattr(args, "set", None))


def _kv(label: str, value: Any, width: int = 26) -> str:
    return f"  {label.ljust(width)} {value}"


def _status_tag(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def _python_ok() -> tuple[bool, str]:
    version = sys.version_info[:2]
    ok = MIN_PYTHON <= version <= MAX_PYTHON
    return ok, f"{sys.version.split()[0]} ({platform.architecture()[0]})"


# ---------------------------------------------------------------------- doctor
def cmd_doctor(args: argparse.Namespace) -> int:
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str, required: bool = True) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": str(detail), "required": required})

    py_ok, py_detail = _python_ok()
    add("python.version", py_ok, py_detail)
    add("python.executable", True, sys.executable)
    add("platform", platform.system() == "Windows", f"{platform.system()} {platform.release()}")

    try:
        import numpy

        add("numpy", True, numpy.__version__)
    except ImportError as exc:  # pragma: no cover - only when deps are missing
        add("numpy", False, f"not importable: {exc}")

    add("sqlite3", sqlite3.sqlite_version_info >= (3, 35, 0), sqlite3.sqlite_version)

    root = project_root()
    add("project.root", root.is_dir(), str(root))
    probe = root / ".wphr_write_probe"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        add("project.writable", True, "write probe succeeded")
    except OSError as exc:
        add("project.writable", False, str(exc))

    usage = shutil.disk_usage(root)
    free_gb = usage.free / (1024**3)
    add("disk.free", free_gb >= MIN_DISK_GB, f"{free_gb:.1f} GiB free")

    config: Config | None = None
    try:
        config = Config.load(getattr(args, "config", None), overrides=getattr(args, "set", None))
        add("config.load", True, str(config.path))
        add("config.hash", True, config.hash()[:16])
        add("config.modules", len(config.module_names()) >= 7, f"{len(config.module_names())} modules")
    except (ConfigError, OSError) as exc:
        add("config.load", False, str(exc))

    gpu = "not detected"
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu = result.stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError):
        gpu = "nvidia-smi unavailable"
    add("gpu", True, gpu, required=False)

    if config is not None:
        store = Store(config.resolve_path("paths.database"), timeout=config.get("memory.db_timeout_s", 30))
        add("database", True, f"schema v{store.version()} at {store.path}", required=False)

    passed = all(item["ok"] for item in checks if item["required"])
    if args.json:
        _print_json({"ok": passed, "checks": checks})
        return 0 if passed else 1

    print(f"WPHR doctor  (version {__version__}, spec {SPEC_VERSION})")
    print(f"  root: {root}")
    print()
    print(f"  {'check'.ljust(22)} {'status'.ljust(7)} detail")
    print(f"  {'-' * 22} {'-' * 7} {'-' * 40}")
    for item in checks:
        tag = _status_tag(item["ok"]) if item["required"] else ("INFO" if item["ok"] else "WARN")
        print(f"  {item['check'].ljust(22)} {tag.ljust(7)} {item['detail']}")
    print()
    print(f"RESULT: {'OK' if passed else 'BLOCKED'}")
    return 0 if passed else 1


# ------------------------------------------------------------------------ init
def cmd_init(args: argparse.Namespace) -> int:
    config = load_config(args)
    created = config.ensure_dirs()
    store = Store(config.resolve_path("paths.database"), timeout=config.get("memory.db_timeout_s", 30))
    created_version = store.init()
    store.set_meta("initialised_at", utc_iso())
    store.set_meta("version", __version__)
    store.set_meta("spec", config.get("project.spec", SPEC_VERSION))
    store.set_meta("config_hash", config.hash())
    print(f"WPHR init: schema v{created_version}")
    print(_kv("project root", project_root()))
    print(_kv("database", store.path))
    for path in created:
        print(_kv("directory", path))
    print(_kv("tables", ", ".join(store.tables())))
    print(_kv("directories created", len(created)))
    return 0


# ------------------------------------------------------------------------- run
def _make_progress_printer(interval: int):
    counter = {"n": 0}

    def printer(result) -> None:
        counter["n"] += 1
        if interval and counter["n"] % interval == 0:
            print(
                f"  step {result.step:6d}  action={result.action:<10s} "
                f"reward={result.reward:+.3f}  err={result.prediction_error:.4f}  "
                f"goal={result.goal:<11s} h={result.h_norm:.3f}"
            )

    return printer


def cmd_run(args: argparse.Namespace) -> int:
    from wphr.runtime.subject import SafetyError, Subject

    config = load_config(args)
    if args.world:
        config.set("world.id", args.world)
        if not config.get(f"world.{args.world}", None):
            print(f"ERROR: no configuration section for world {args.world!r}", file=sys.stderr)
            return 2
    store = Store(config.resolve_path("paths.database"), timeout=config.get("memory.db_timeout_s", 30))
    try:
        subject = Subject(config, store=store, seed=args.seed, persist=not args.no_persist)
    except SafetyError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 3
    subject.initialise()
    if args.resume:
        checkpoint = subject.restore(args.resume, allow_config_mismatch=args.allow_config_mismatch)
        print(
            f"restored {checkpoint.checkpoint_id} at step {checkpoint.step} "
            f"(sha256 {checkpoint.sha256[:16]})"
        )
    print(
        f"WPHR run  world={config.get('world.id')}  seed={subject.seed}  "
        f"config={config.hash()[:16]}  persist={not args.no_persist}"
    )
    printer = _make_progress_printer(args.progress_every)
    summary = subject.run(
        args.steps,
        checkpoint_every=args.checkpoint_every,
        on_step=printer,
        stop_on_control=not args.ignore_control,
    )
    subject.write_status_snapshot()
    subject.close(status=summary.state)
    payload = summary.to_dict()
    print()
    print("run summary:")
    for key, value in payload.items():
        print(_kv(key, value))
    print()
    print("health:", subject.health()["overall"])
    return 0 if summary.state in ("completed", "stopped", "paused") else 1


def _control_request(args: argparse.Namespace, command: str) -> int:
    from wphr.runtime.control import ControlFile

    config = load_config(args)
    control = ControlFile(config.resolve_path("paths.control"))
    payload = control.request(command, note=getattr(args, "note", "") or "")
    print(f"control request written to {control.path}: {payload['command']}")
    print("this takes effect at the running process's next control poll (<=10 steps)")
    return 0


def cmd_pause(args: argparse.Namespace) -> int:
    return _control_request(args, "pause")


def cmd_stop(args: argparse.Namespace) -> int:
    return _control_request(args, "stop")


# ---------------------------------------------------------------------- status
def cmd_status(args: argparse.Namespace) -> int:
    config = load_config(args)
    status_path = config.resolve_path("paths.status")
    store = Store(config.resolve_path("paths.database"), timeout=config.get("memory.db_timeout_s", 30))
    snapshot: dict[str, Any] | None = None
    if status_path.is_file():
        try:
            snapshot = json.loads(status_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            snapshot = None
    if args.json:
        _print_json(
            {
                "version": __version__,
                "status_snapshot": snapshot,
                "database": store.stats(),
                "control": _control_state(config),
            }
        )
        return 0
    print(f"WPHR status  (version {__version__}, spec {SPEC_VERSION})")
    print(_kv("config hash", config.hash()[:16]))
    print(_kv("database", store.path))
    print(_kv("schema version", store.version()))
    print(_kv("events", store.count_events()))
    print(_kv("size", f"{store.path.stat().st_size} bytes" if store.path.is_file() else "n/a"))
    print(_kv("control", _control_state(config)))
    print()
    if snapshot is None:
        print("no live status snapshot found (run `wphr run` first); showing persisted database only")
        print()
        print("event types:", store.event_type_counts())
        checkpoints = store.list_checkpoints(limit=5)
        if checkpoints:
            print()
            print("recent checkpoints:")
            for row in checkpoints:
                print(f"  {row['checkpoint_id']}  step={row['step']}  sha={row['sha256'][:12]}")
        return 0
    print(f"live snapshot written at {snapshot.get('written_at')}")
    print(_kv("subject", snapshot.get("subject_id")))
    print(_kv("episode", snapshot.get("episode_id")))
    print(_kv("step", snapshot.get("step")))
    print(_kv("sim time", snapshot.get("sim_time")))
    print(_kv("world", snapshot.get("world")))
    print(_kv("overall status", snapshot.get("overall_status")))
    print(_kv("goal", snapshot.get("goal")))
    print(_kv("action", snapshot.get("action")))
    print(_kv("concepts", snapshot.get("concepts")))
    core = snapshot.get("core", {})
    print(_kv("core finite", core.get("finite")))
    print(_kv("core faults", core.get("faults")))
    print(_kv("spectral radius", core.get("spectral_radius")))
    print()
    print("modules:")
    for module in snapshot.get("modules", []):
        print(f"  {module['module'].ljust(18)} {module['status'].ljust(9)} {module.get('detail', '')}")
    print()
    print("affect:")
    for key, value in sorted((snapshot.get("affect") or {}).items()):
        print(f"  {key.ljust(18)} {value:.4f}")
    return 0


def _control_state(config: Config) -> dict[str, Any]:
    from wphr.runtime.control import ControlFile

    control = ControlFile(config.resolve_path("paths.control"))
    return control.state()


# ---------------------------------------------------------------------- memory
def cmd_memory(args: argparse.Namespace) -> int:
    config = load_config(args)
    store = Store(config.resolve_path("paths.database"), timeout=config.get("memory.db_timeout_s", 30))
    if not store.exists:
        print("ERROR: database not initialised (run `wphr init`)", file=sys.stderr)
        return 2
    if args.action == "inspect":
        stats = store.stats()
        print(f"WPHR memory  ({store.path})")
        print(_kv("schema version", stats["schema_version"]))
        print(_kv("events", stats["events"]))
        print(_kv("size", f"{stats['size_bytes']} bytes"))
        print(_kv("integrity", stats["integrity"]))
        print(_kv("runs", stats["runs"]))
        print(_kv("checkpoints", stats["checkpoints"]))
        print()
        print("event types:")
        for event_type, count in sorted(store.event_type_counts().items()):
            print(f"  {event_type.ljust(16)} {count}")
        print()
        print("memory layers (active records):")
        layers = store.memory_counts()
        if not layers:
            print("  (none consolidated yet)")
        for layer, count in sorted(layers.items()):
            print(f"  {layer.ljust(18)} {count}")
        print()
        print("recent semantic rules:")
        for row in store.list_memories(layer="semantic", limit=8):
            print(
                f"  {row['key'].ljust(22)} conf={row['confidence']:.3f} "
                f"support={row['support_count']} counter={row['counter_count']}"
            )
        print()
        print("recent summaries:")
        for row in store.list_summaries(limit=3):
            print(f"  [{row['start_step']}->{row['end_step']}] {row['text'][:160]}")
        print()
        print("last 10 events:")
        for event in store.recent_events(limit=10):
            print(f"  seq_ts={event.timestamp:.3f} {event.summary()}")
        return 0
    if args.action == "export":
        target = Path(args.out) if args.out else config.resolve_path("paths.results") / "memory_export.jsonl"
        count = store.export_jsonl(target, limit=args.limit)
        print(f"exported {count} events to {target}")
        return 0
    print(f"ERROR: unknown memory action {args.action!r}", file=sys.stderr)
    return 2


# ------------------------------------------------------------------ checkpoint
def cmd_checkpoint(args: argparse.Namespace) -> int:
    from wphr.memory.checkpoint import CheckpointManager

    config = load_config(args)
    store = Store(config.resolve_path("paths.database"), timeout=config.get("memory.db_timeout_s", 30))
    manager = CheckpointManager(store, config)
    if args.action == "create":
        if not args.note:
            args.note = ""
        return _control_request(args, "checkpoint")
    if args.action == "list":
        rows = manager.list(limit=args.limit)
        if not rows:
            print("no checkpoints recorded")
            return 0
        print(f"{'checkpoint_id'.ljust(46)} {'step'.rjust(8)} {'size'.rjust(10)} note")
        for row in rows:
            print(
                f"{row['checkpoint_id'].ljust(46)} {row['step']:>8d} "
                f"{row['size_bytes']:>10d} {row.get('note') or ''}"
            )
        return 0
    if args.action in ("restore", "inspect"):
        try:
            checkpoint = manager.load(args.id, allow_config_mismatch=args.allow_config_mismatch)
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        summary = checkpoint.summary()
        print("checkpoint verified")
        for key, value in summary.items():
            print(_kv(key, value))
        print(_kv("config hash", str(checkpoint.meta.get("config_hash", ""))[:16]))
        print(_kv("schema version", checkpoint.meta.get("schema_version")))
        print(_kv("modules", ", ".join(checkpoint.meta.get("modules", {}))))
        if args.action == "restore":
            print()
            print("to resume from this checkpoint: wphr run --resume " + checkpoint.checkpoint_id)
        return 0
    print(f"ERROR: unknown checkpoint action {args.action!r}", file=sys.stderr)
    return 2


# ------------------------------------------------------- experiments / reports
def cmd_experiment(args: argparse.Namespace) -> int:
    from wphr.experiments import get_experiment, list_experiments, run_experiment

    if args.action == "list":
        specs = list_experiments()
        if not specs:
            print("no experiments registered in this version")
            return 0
        for spec in specs:
            print(
                f"{spec.experiment_id}  v{spec.version}  {spec.title}\n"
                f"    metric={spec.primary_metric}  conditions={spec.conditions}  "
                f"seeds={spec.seeds}"
            )
        return 0
    if args.id is None:
        print("ERROR: `experiment run` needs an experiment id", file=sys.stderr)
        return 2
    try:
        spec = get_experiment(args.id)
    except KeyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    config = load_config(args)
    out_dir = config.resolve_path("paths.results")
    print(f"running {spec.experiment_id}: {spec.title}")
    print(f"  metric={spec.primary_metric} conditions={spec.conditions} seeds={spec.seeds}")
    result = run_experiment(spec, out_dir=out_dir)
    summary = result["summary"]
    print()
    print(f"primary metric: {summary['primary_metric']}")
    for name, bucket in summary["conditions"].items():
        print(
            f"  {name:<10} n={bucket['n']} mean={bucket['mean']} std={bucket['std']} "
            f"ci95=[{bucket['ci95_low']}, {bucket['ci95_high']}]"
        )
        for key, value in bucket["per_metric"].items():
            print(f"      {key}: {value}")
    if summary["difference"] is not None:
        print(f"  difference: {summary['difference']}")
    print()
    print(f"raw results: {result['path']}")
    print(f"raw csv:     {result['csv']}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    config = load_config(args)
    store = Store(config.resolve_path("paths.database"), timeout=config.get("memory.db_timeout_s", 30))
    out_dir = config.resolve_path("paths.docs") / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = utc_iso().replace(":", "").replace("-", "")
    target = out_dir / f"REPORT_{stamp}.md"
    lines = [
        f"# WPHR run report ({utc_iso()})",
        "",
        f"- version: `{__version__}`",
        f"- spec: `{SPEC_VERSION}`",
        f"- config hash: `{config.hash()}`",
        f"- database: `{store.path}`",
        "",
        "## Persisted state",
        "",
    ]
    stats = store.stats()
    for key in ("schema_version", "events", "size_bytes", "runs", "checkpoints", "integrity"):
        lines.append(f"- {key}: {stats.get(key)}")
    lines += ["", "## Event types", ""]
    for event_type, count in sorted(store.event_type_counts().items()):
        lines.append(f"- {event_type}: {count}")
    lines += ["", "## Runs", ""]
    for run in store.list_runs(limit=20):
        lines.append(
            f"- `{run['run_id']}` steps={run['steps']} status={run['status']} "
            f"seed={run['seed']} started={utc_iso(run['started_at'])}"
        )
    lines += ["", "## Checkpoints", ""]
    for row in store.list_checkpoints(limit=20):
        lines.append(f"- `{row['checkpoint_id']}` step={row['step']} sha256=`{row['sha256']}`")
    lines += ["", "## Recent events", ""]
    for event in store.recent_events(limit=20):
        lines.append(f"- {event.summary()}")
    status_path = config.resolve_path("paths.status")
    if status_path.is_file():
        lines += ["", "## Latest status snapshot", "", "```json",
                  status_path.read_text(encoding="utf-8")[:4000], "```"]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report written to {target}")
    return 0


def cmd_affect(args: argparse.Namespace) -> int:
    config = load_config(args)
    if args.action == "reset":
        from wphr.runtime.control import ControlFile

        control = ControlFile(config.resolve_path("paths.control"))
        control.request("reset_affect", note="manual recovery")
        print(f"affect reset requested through {control.path}")
        return 0
    status_path = config.resolve_path("paths.status")
    if not status_path.is_file():
        print("no status snapshot; run the subject first", file=sys.stderr)
        return 2
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    print(f"affect at step {payload.get('step')} ({payload.get('written_at')})")
    for key, value in sorted((payload.get("affect") or {}).items()):
        print(f"  {key.ljust(18)} {value:.4f}")
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    from wphr.ui.server import LOOPBACK, serve

    if args.host not in (LOOPBACK, "localhost"):
        print(
            f"REFUSED: the UI only binds loopback; requested host {args.host!r}",
            file=sys.stderr,
        )
        return 2
    config = load_config(args)
    if not config.resolve_path("paths.database").is_file():
        print("ERROR: database not initialised (run `wphr init` first)", file=sys.stderr)
        return 2
    try:
        return serve(port=args.port, host=LOOPBACK)
    except OSError as exc:
        print(f"ERROR: cannot bind {LOOPBACK}:{args.port}: {exc}", file=sys.stderr)
        return 1


# ------------------------------------------------------------------------ main
def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-c", "--config", default=None, help="path to a config JSON file")
    common.add_argument(
        "-s", "--set", action="append", default=[], metavar="KEY=VALUE",
        help="override a config value, e.g. -s runtime.seed=7",
    )

    parser = argparse.ArgumentParser(prog="wphr", description="WPHR digital subject")
    parser.add_argument("--version", action="version", version=f"wphr {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", parents=[common], help="environment and dependency checks")
    doctor.add_argument("--json", action="store_true")
    doctor.set_defaults(func=cmd_doctor)

    init = sub.add_parser("init", parents=[common], help="initialise directories and database")
    init.set_defaults(func=cmd_init)

    run = sub.add_parser("run", parents=[common], help="run the subject loop")
    run.add_argument("--world", default=None, help="world id (default: config world.id)")
    run.add_argument("--steps", type=int, default=1000, help="number of steps to run")
    run.add_argument("--seed", type=int, default=None)
    run.add_argument("--checkpoint-every", type=int, default=None)
    run.add_argument("--progress-every", type=int, default=100)
    run.add_argument("--no-persist", action="store_true", help="do not write to SQLite")
    run.add_argument("--ignore-control", action="store_true")
    run.add_argument("--resume", nargs="?", const="latest", default=None)
    run.add_argument("--allow-config-mismatch", action="store_true")
    run.set_defaults(func=cmd_run)

    for name, func, text in (
        ("pause", cmd_pause, "request a pause"),
        ("stop", cmd_stop, "request a safe stop"),
    ):
        sub_parser = sub.add_parser(name, parents=[common], help=text)
        sub_parser.add_argument("--note", default="")
        sub_parser.set_defaults(func=func)

    status = sub.add_parser("status", parents=[common], help="show subject and module status")
    status.add_argument("--json", action="store_true")
    status.set_defaults(func=cmd_status)

    memory = sub.add_parser("memory", parents=[common], help="inspect or export memory")
    memory.add_argument("action", choices=["inspect", "export"])
    memory.add_argument("--out", default=None)
    memory.add_argument("--limit", type=int, default=None)
    memory.set_defaults(func=cmd_memory)

    checkpoint = sub.add_parser("checkpoint", parents=[common], help="checkpoint operations")
    checkpoint.add_argument("action", choices=["create", "list", "restore", "inspect"])
    checkpoint.add_argument("id", nargs="?", default="latest")
    checkpoint.add_argument("--note", default="")
    checkpoint.add_argument("--limit", type=int, default=20)
    checkpoint.add_argument("--allow-config-mismatch", action="store_true")
    checkpoint.set_defaults(func=cmd_checkpoint)

    experiment = sub.add_parser("experiment", parents=[common], help="list or run experiments")
    experiment.add_argument("action", choices=["list", "run"])
    experiment.add_argument("id", nargs="?", default=None)
    experiment.set_defaults(func=cmd_experiment)

    affect = sub.add_parser("affect", parents=[common], help="inspect or reset functional affect")
    affect.add_argument("action", choices=["show", "reset"], nargs="?", default="show")
    affect.set_defaults(func=cmd_affect)

    report = sub.add_parser("report", parents=[common], help="build reports")
    report.add_argument("action", choices=["build"])
    report.set_defaults(func=cmd_report)

    ui = sub.add_parser("ui", parents=[common], help="start the local read-only UI")
    ui.add_argument("--port", type=int, default=8765)
    ui.add_argument("--host", default="127.0.0.1", help="loopback only; other hosts are refused")
    ui.set_defaults(func=cmd_ui)

    return parser


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to a legacy code page; Chinese output would be
    # mangled. Reconfigure only when stdout is a real stream (not captured).
    for stream in (sys.stdout, sys.stderr):
        encoding = getattr(stream, "encoding", "") or ""
        if encoding.lower().replace("-", "") != "utf8" and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
