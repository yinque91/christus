"""CORE-READOUT-01 formal driver: A0/A1/A2/A3, multiple seeds, full retention.

The experiment is pre-registered in ``CORE_READOUT_PREREGISTRATION.md``. This
script refuses to run if the document hash does not match
``results/core_readout_prereg/prereg.json``. It writes every arm/seed to its own
directory and never overwrites an existing run; failures are retained as
``failure.json`` and do not stop the remaining seeds.

Usage (from the project root, with a Python that has numpy):

    python -m christus.exp_core_readout --out results/core_readout_gate01 \
        --seeds 31100:31105 --steps 300 --gate
    python -m christus.exp_core_readout --out results/core_readout_formal01 \
        --seeds 31100:31120 --steps 1000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from .runtime import create, record, ROOT

ARMS = ("A0", "A1", "A2", "A3")
PREREG_DOC = ROOT / "CORE_READOUT_PREREGISTRATION.md"
PREREG_META = ROOT / "results" / "core_readout_prereg" / "prereg.json"


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_prereg() -> dict:
    meta = json.loads(PREREG_META.read_text(encoding="utf-8-sig"))
    actual = file_sha256(PREREG_DOC)
    if actual != meta["sha256"]:
        raise SystemExit(
            "preregistration hash mismatch: "
            f"document={actual[:16]} recorded={meta['sha256'][:16]}"
        )
    return meta


def parse_seeds(spec: str) -> list[int]:
    seeds: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            start, end = part.split(":", 1)
            seeds.extend(range(int(start), int(end)))
        else:
            seeds.append(int(part))
    # stable, duplicate-free order
    seen = set()
    out = []
    for seed in seeds:
        if seed not in seen:
            seen.add(seed)
            out.append(seed)
    return out


def _phase(rows: list[dict], lo: int, hi: int) -> dict:
    part = [r for r in rows if lo <= int(r["step"]) <= hi]
    terminal = [r for r in part if r.get("outcome")]
    caught = sum(1 for r in terminal if r["outcome"] == "caught")
    overrides = [r for r in part if r.get("readout_override")]
    aligned = sum(1 for r in overrides if r.get("readout_aligned"))
    off_center = [
        r for r in part
        if r.get("visible") and abs(float(r.get("scent_x", 0.0))) > 0.05
    ]
    correct = sum(
        1 for r in off_center
        if (float(r["scent_x"]) > 0 and r["action"] == "move_east")
        or (float(r["scent_x"]) < 0 and r["action"] == "move_west")
    )
    wrong = sum(
        1 for r in off_center
        if (float(r["scent_x"]) > 0 and r["action"] == "move_west")
        or (float(r["scent_x"]) < 0 and r["action"] == "move_east")
    )
    return {
        "steps": len(part),
        "episodes": len(terminal),
        "caught": caught,
        "catch_rate": (caught / len(terminal)) if terminal else None,
        "reward_sum": round(float(sum(float(r["reward"]) for r in part)), 6),
        "body_mae": round(float(np.mean([float(r["body_error"]) for r in part])), 6)
        if part
        else None,
        "override_rate": round(len(overrides) / len(part), 6) if part else 0.0,
        "alignment": (aligned / len(overrides)) if overrides else None,
        "steer_n": len(off_center),
        "steer_accuracy": (correct / len(off_center)) if off_center else None,
        "steer_signed": ((correct - wrong) / len(off_center)) if off_center else None,
    }


def summarize(rows: list[dict], arm: str, seed: int) -> dict:
    terminal = [r for r in rows if r.get("outcome")]
    caught = sum(1 for r in terminal if r["outcome"] == "caught")
    visible = [r for r in rows if r.get("visible")]
    off_center = [r for r in visible if abs(float(r.get("scent_x", 0.0))) > 0.05]
    steer_correct = sum(
        1
        for r in off_center
        if (float(r["scent_x"]) > 0 and r["action"] == "move_east")
        or (float(r["scent_x"]) < 0 and r["action"] == "move_west")
    )
    steer_wrong = sum(
        1
        for r in off_center
        if (float(r["scent_x"]) > 0 and r["action"] == "move_west")
        or (float(r["scent_x"]) < 0 and r["action"] == "move_east")
    )
    directional = sum(1 for r in off_center if r["action"] in ("move_east", "move_west"))
    overrides = [r for r in rows if r.get("readout_override")]
    aligned = sum(1 for r in overrides if r.get("readout_aligned"))
    pred = np.asarray([float(r.get("readout_pred", 0.0)) for r in visible], dtype=np.float64)
    target = np.asarray([float(r.get("scent_x", 0.0)) for r in visible], dtype=np.float64)
    if len(visible) >= 2 and float(np.std(pred)) > 1e-12 and float(np.std(target)) > 1e-12:
        readout_corr = float(np.corrcoef(pred, target)[0, 1])
    else:
        readout_corr = 0.0
    readout_mae = (
        float(np.mean(np.abs(pred - target))) if len(visible) else None
    )
    action_counts = {a: 0 for a in ("move_north", "move_east", "move_south",
                                    "move_west", "interact", "wait")}
    for r in rows:
        action_counts[str(r["action"])] = action_counts.get(str(r["action"]), 0) + 1
    return {
        "arm": arm,
        "seed": int(seed),
        "steps": len(rows),
        "episodes": len(terminal),
        "caught": caught,
        "catch_rate": (caught / len(terminal)) if terminal else None,
        "reward_sum": round(float(sum(float(r["reward"]) for r in rows)), 6),
        "cost_sum": round(float(sum(float(r.get("cost", 0.0)) for r in rows)), 6),
        "body_mae": round(float(np.mean([float(r["body_error"]) for r in rows])), 6)
        if rows
        else None,
        "visible_steps": len(visible),
        "steer_n": len(off_center),
        "steer_correct": steer_correct,
        "steer_wrong": steer_wrong,
        "steer_accuracy": (steer_correct / len(off_center)) if off_center else None,
        # Signed directional score removes the action-frequency confound:
        # a random left/right bias can raise unsigned accuracy simply by using
        # east/west more often, but its correct and wrong counts cancel here.
        "steer_signed": ((steer_correct - steer_wrong) / len(off_center))
        if off_center
        else None,
        "directional_share": (directional / len(off_center)) if off_center else None,
        "overrides": len(overrides),
        "override_rate": round(len(overrides) / len(rows), 6) if rows else 0.0,
        "alignment": (aligned / len(overrides)) if overrides else None,
        "readout_corr": round(readout_corr, 6),
        "readout_mae": round(readout_mae, 6) if readout_mae is not None else None,
        "action_counts": action_counts,
        "phases": {
            "normal": _phase(rows, 1, 400),
            "changed": _phase(rows, 401, 750),
            "return": _phase(rows, 751, 1000),
        },
    }


def run_one(out: Path, arm: str, seed: int, steps: int) -> dict:
    run_dir = out / arm / f"s{seed}"
    if (run_dir / "events.jsonl").exists():
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            return json.loads(summary_path.read_text(encoding="utf-8"))
        return {"arm": arm, "seed": seed, "skipped": "existing partial run"}
    run_dir.mkdir(parents=True, exist_ok=True)
    subject = None
    started = time.time()
    try:
        subject = create(run_dir, seed=seed, profile="safe", arm=arm)
        subject.config.save(run_dir / "config.json")
        rows: list[dict] = []
        with (run_dir / "events.jsonl").open("x", encoding="utf-8") as handle:
            for index in range(steps):
                row = record(subject, subject.step(), compact=True)
                rows.append(row)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                if (index + 1) % 100 == 0:
                    handle.flush()
        try:
            subject.save_checkpoint("core-readout run end")
        except Exception:
            (run_dir / "checkpoint_failure.txt").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
        summary = summarize(rows, arm, seed)
        summary["wall_seconds"] = round(time.time() - started, 3)
        summary["health"] = [h.to_dict() for h in subject.registry.health_report()]
        write_json(run_dir / "summary.json", summary)
        return summary
    except Exception:
        failure = {
            "arm": arm,
            "seed": seed,
            "steps_requested": steps,
            "traceback": traceback.format_exc(),
        }
        write_json(run_dir / "failure.json", failure)
        return {"arm": arm, "seed": seed, "error": failure["traceback"]}
    finally:
        if subject is not None:
            try:
                subject.close()
            except Exception:
                pass


def gate(out: Path, seeds: list[int]) -> dict:
    summaries: dict[str, dict[int, dict]] = {arm: {} for arm in ARMS}
    failures = []
    for arm in ARMS:
        for seed in seeds:
            path = out / arm / f"s{seed}" / "summary.json"
            if not path.exists():
                failures.append(f"{arm}/s{seed}: missing summary")
                continue
            summaries[arm][seed] = json.loads(path.read_text(encoding="utf-8"))

    def mean(arm: str, key: str):
        values = [s[key] for s in summaries[arm].values() if s.get(key) is not None]
        return float(np.mean(values)) if values else None

    a1_corr = mean("A1", "readout_corr")
    a1_align = mean("A1", "alignment")
    a1_steer = mean("A1", "steer_accuracy")
    a1_signed = mean("A1", "steer_signed")
    a3_align = mean("A3", "alignment")
    a3_steer = mean("A3", "steer_accuracy")
    a3_signed = mean("A3", "steer_signed")
    a3_corr = mean("A3", "readout_corr")
    a0_steer = mean("A0", "steer_accuracy")
    a0_signed = mean("A0", "steer_signed")
    a2_override = mean("A2", "override_rate")

    identity_mismatches = []
    for seed in seeds:
        a0 = summaries["A0"].get(seed)
        a2 = summaries["A2"].get(seed)
        if not a0 or not a2:
            identity_mismatches.append(f"s{seed}: missing A0/A2")
            continue
        keys = ("episodes", "caught", "catch_rate", "reward_sum", "cost_sum",
                "body_mae", "action_counts")
        for key in keys:
            if a0.get(key) != a2.get(key):
                identity_mismatches.append(f"s{seed}:{key} A0={a0.get(key)!r} A2={a2.get(key)!r}")

    checks = {
        "a1_readout_corr_gt_0.30": (a1_corr is not None and a1_corr > 0.30),
        "a1_alignment_gt_0.55": (a1_align is not None and a1_align > 0.55),
        "a1_steer_accuracy_gt_0.55": (a1_steer is not None and a1_steer > 0.55),
        "a1_steer_signed_gt_0.20": (a1_signed is not None and a1_signed > 0.20),
        "a2_override_rate_eq_0": (a2_override == 0.0),
        "a0_a2_identical": len(identity_mismatches) == 0,
        "a3_steer_signed_within_0.10_of_a0": (
            a3_signed is not None and a0_signed is not None
            and abs(a3_signed - a0_signed) <= 0.10
        ),
        "a3_readout_corr_lt_0.15": (a3_corr is not None and a3_corr < 0.15),
        "no_run_failures": len(failures) == 0,
    }
    result = {
        "gate": "CORE-READOUT-01 manipulation gate",
        "seeds": seeds,
        "summaries": {
            arm: {
                "readout_corr": mean(arm, "readout_corr"),
                "alignment": mean(arm, "alignment"),
                "steer_accuracy": mean(arm, "steer_accuracy"),
                "steer_signed": mean(arm, "steer_signed"),
                "catch_rate": mean(arm, "catch_rate"),
                "override_rate": mean(arm, "override_rate"),
                "readout_corr_mean": mean(arm, "readout_corr"),
            }
            for arm in ARMS
        },
        "checks": checks,
        "passed": all(checks.values()),
        "failures": failures,
        "identity_mismatches": identity_mismatches,
        "a3_alignment_descriptive": a3_align,
    }
    write_json(out / "GATE.json", result)
    lines = ["# CORE-READOUT-01 操作门控", "", f"Seeds: {seeds}", "",
             "| 检查 | 结果 |", "|---|---|"]
    for name, ok in checks.items():
        lines.append(f"| {name} | {'PASS' if ok else 'FAIL'} |")
    lines += ["", "## 臂均值", "",
              "| 臂 | readout_corr | alignment | steer_accuracy | steer_signed | catch_rate | override_rate |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        s = result["summaries"][arm]
        lines.append(
            f"| {arm} | {s['readout_corr']} | {s['alignment']} | {s['steer_accuracy']} | "
            f"{s['steer_signed']} | {s['catch_rate']} | {s['override_rate']} |"
        )
    if failures:
        lines += ["", "## 失败", ""] + [f"- {x}" for x in failures]
    if identity_mismatches:
        lines += ["", "## A0/A2 不一致", ""] + [f"- {x}" for x in identity_mismatches]
    (out / "GATE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", default="31100:31120")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--arms", default="A0,A1,A2,A3")
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() and not args.gate:
        # A fresh formal directory is required; existing runs are never mixed.
        existing = list(out.glob("A*/s*/events.jsonl"))
        if existing:
            raise SystemExit(f"refusing to mix with existing runs in {out}")
    out.mkdir(parents=True, exist_ok=True)
    prereg = verify_prereg()
    seeds = parse_seeds(args.seeds)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    manifest = {
        "experiment": "CORE-READOUT-01",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "prereg": prereg,
        "python": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "out": str(out),
        "arms": arms,
        "seeds": seeds,
        "steps": args.steps,
        "code_sha256": {
            str(p.relative_to(ROOT)): file_sha256(p)
            for p in sorted((ROOT / "christus").glob("*.py"))
        },
    }
    write_json(out / "MANIFEST.json", manifest)
    results = []
    for arm in arms:
        for seed in seeds:
            print(f"RUN {arm} seed={seed}", flush=True)
            results.append(run_one(out, arm, seed, args.steps))
            write_json(out / "RUNS.json", results)
    if args.gate:
        result = gate(out, seeds)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(0 if result["passed"] else 1)
    print(json.dumps({"runs": len(results), "out": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
