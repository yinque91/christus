"""CLI for the permissioned Christus desktop actuator and supervisor."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .actuator import DesktopActuator
from .perception import foreground, list_windows
from .safety import SafetyBroker, SafetyError
from .supervisor import AutonomySupervisor


def _print(obj):
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def cmd_start(args):
    sup = AutonomySupervisor(mode=args.mode, interval=args.interval,
                             screenshots=args.screenshots)
    try:
        print(f"Christus autonomy supervisor started in {args.mode} mode. "
              f"Emergency hotkey: Ctrl+Alt+Shift+Q. State: {sup.state_dir}", flush=True)
        sup.run()
    except KeyboardInterrupt:
        sup.stop()


def cmd_stop(args):
    sup = AutonomySupervisor()
    sup.stop()
    print(f"stop requested: {sup.stop_path}")


def cmd_clear_stop(args):
    sup = AutonomySupervisor()
    if sup.stop_path.exists():
        sup.stop_path.unlink()
        print("emergency stop cleared")
    else:
        print("no stop file")


def cmd_status(args):
    _print(AutonomySupervisor().status())


def cmd_observe(args):
    _print({"foreground": foreground().__dict__,
            "windows": [w.__dict__ for w in list_windows()]})


def cmd_act(args):
    broker = SafetyBroker()
    actuator = DesktopActuator(broker, dry_run=args.dry_run)
    a = args.action
    if not a:
        raise SystemExit("missing action")
    name, rest = a[0], a[1:]
    try:
        if name == "move":
            actuator.move(int(rest[0]), int(rest[1]))
        elif name in ("click", "doubleclick"):
            x = int(rest[0]) if rest else None
            y = int(rest[1]) if len(rest) > 1 else None
            button = rest[2] if len(rest) > 2 else "left"
            actuator.click(x, y, button=button, double=(name == "doubleclick"))
        elif name == "scroll":
            actuator.scroll(int(rest[0]))
        elif name == "type":
            actuator.type_text(" ".join(rest))
        elif name == "key":
            actuator.key(rest[0], int(rest[1]) if len(rest) > 1 else 1)
        elif name == "hotkey":
            actuator.hotkey(rest[0])
        elif name == "activate":
            ok = actuator.activate(" ".join(rest))
            if not ok:
                raise SystemExit("window not found")
        elif name == "launch":
            actuator.launch(rest[0])
        else:
            raise SystemExit(f"unknown action: {name}")
        print("ok")
    except (SafetyError, IndexError, ValueError) as exc:
        raise SystemExit(f"blocked: {exc}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="christus_autonomy")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("start", help="start the persistent supervisor")
    p.add_argument("--mode", choices=("observe", "control"), default="observe")
    p.add_argument("--interval", type=float, default=2.0)
    p.add_argument("--screenshots", action="store_true",
                   help="save periodic screenshots (privacy-sensitive)")
    p.set_defaults(func=cmd_start)

    p = sub.add_parser("stop", help="request supervisor stop")
    p.set_defaults(func=cmd_stop)

    p = sub.add_parser("clear-stop", help="clear an emergency stop")
    p.set_defaults(func=cmd_clear_stop)

    p = sub.add_parser("status")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("observe", help="print the current foreground/window tree")
    p.set_defaults(func=cmd_observe)

    p = sub.add_parser("act", help="execute one permissioned action immediately")
    p.add_argument("action", nargs=argparse.REMAINDER,
                   help="move X Y | click X Y [left|right] | doubleclick X Y | "
                        "scroll N | type TEXT | key NAME [N] | hotkey ctrl+s | "
                        "activate TITLE | launch EXE")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_act)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
