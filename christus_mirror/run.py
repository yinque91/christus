"""Run MIRROR-01 M1闁炽儲褰?, A0闁炽儲寮?, 20 seeds, and export raw traces.

Ground truth is kept in the evaluator objects and is never passed to the subject.
"""
from __future__ import annotations

import json
import math
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from christus_mirror.subject import MirrorRuntime  # noqa: E402
from christus_mirror.world import ACTIONS, MARKER_LOCATIONS  # noqa: E402

OUT = ROOT / "mirror01"
SEEDS = list(range(12000, 12020))
CONDITIONS = {
    "A0": dict(use_body=False, use_self=False, mirror=False),
    "A1": dict(use_body=False, use_self=False, mirror=True),
    "A2": dict(use_body=True, use_self=False, mirror=True),
    "A3": dict(use_body=True, use_self=True, mirror=True),
    "A4": dict(use_body=True, use_self=True, mirror=True, shuffle=True),
    "A5": dict(use_body=False, use_self=False, mirror=True),
}


def entropy(actions):
    c = Counter(actions)
    n = max(1, len(actions))
    return -sum((v / n) * math.log(v / n + 1e-12) for v in c.values())


def run_steps(rt, n):
    rows = []
    for _ in range(n):
        rows.append(rt.step())
    return rows


def summarize(rows, marker=None):
    if not rows:
        return {}
    actions = [r["action"] for r in rows]
    sync = [r["sync"] for r in rows]
    body = [r["body_motion"] for r in rows]
    mirror = [r["mirror_motion"] for r in rows]
    vis = [np.asarray(r["visual"], float) for r in rows]
    vm = [0.0] + [float(np.linalg.norm(vis[i] - vis[i-1])) for i in range(1, len(vis))]
    inspected = [r for r in rows if r["inspect_requested"]]
    inspected_target = [r["action"] for r in inspected]
    return {
        "n": len(rows),
        "actions": Counter(actions),
        "entropy": entropy(actions),
        "sync_mean": float(np.mean(sync)),
        "sync_std": float(np.std(sync)),
        "body_motion_mean": float(np.mean(body)),
        "mirror_motion_mean": float(np.mean(mirror)),
        "visual_motion_mean": float(np.mean(vm)),
        "correspondence": float(np.corrcoef(body, vm)[0, 1])
        if len(set(body)) > 1 and len(set(vm)) > 1 else 0.0,
        "prediction_error_mean": float(np.mean(np.abs(np.asarray(body) - np.asarray(mirror)))),
        "marker_seen": bool(any(r["marker_seen"] for r in rows)),
        "inspection_rate": len(inspected) / len(rows),
        "inspection_action": inspected_target[-1] if inspected_target else None,
        "target_correct": (inspected_target[-1] ==
                           ("touch_forehead", "touch_left_hand", "touch_right_hand")[marker])
        if marker is not None and inspected_target else None,
        "marker_contact": bool(any(r["marker_contact"] for r in rows)),
        "inspection_contact": bool(any(r["inspect_requested"] and r["marker_contact"] for r in rows)),
        "false_inspection_policy": bool(any(r["inspect_requested"] and not r["marker_contact"] for r in rows)),
        "false_inspection": bool(any(r["false_inspection"] for r in rows)),
        "first_inspection_step": next((r["step"] for r in rows if r["inspect_requested"]), None),
    }


def lag_metrics(rows):
    if len(rows) < 5:
        return {"lag_peak": None, "lag_corr": 0.0}
    body = np.asarray([r["body_motion"] for r in rows], float)
    vis = np.asarray([r["visual"][:8] for r in rows], float)
    vm = np.zeros(len(vis))
    vm[1:] = np.linalg.norm(vis[1:] - vis[:-1], axis=1)
    best = (None, -2.0)
    for lag in range(0, 13):
        if lag >= len(body) - 2:
            break
        a = body[:len(body)-lag]
        b = vm[lag:]
        if len(set(a)) > 1 and len(set(b)) > 1:
            c = float(np.corrcoef(a, b)[0, 1])
            if c > best[1]:
                best = (lag, c)
    return {"lag_peak": best[0], "lag_corr": round(max(0.0, best[1]), 4)}


def make_rt(seed, condition):
    c = CONDITIONS[condition]
    out = OUT / "state" / f"{condition}_{seed}"
    rt = MirrorRuntime(out, seed=seed, use_body=c["use_body"],
                       use_self=c["use_self"], shuffle=c.get("shuffle", False))
    rt.world.mirror_condition = "realtime" if c["mirror"] else "none"
    return rt


def run_m1(seed):
    rt = make_rt(seed, "A3")
    try:
        rows = run_steps(rt, 30)
        summary = summarize(rows)
        summary["trace"] = rows
        return summary
    finally:
        rt.close()


def run_m2(seed):
    data = {}
    order = ["realtime", "delay_0.5", "delay_2", "other", "none"]
    for cond in order:
        rt = make_rt(seed, "A3")
        try:
            rt.world.mirror_condition = cond
            rows = run_steps(rt, 25)
            data[cond] = summarize(rows)
            data[cond].update(lag_metrics(rows))
            data[cond]["trace"] = rows
        finally:
            rt.close()
    return data


def run_m3(seed, condition="A3", marker_mode="real", marker=None, geometry=None):
    rt = make_rt(seed, condition)
    try:
        if geometry:
            rt.world.mirror_scale = geometry.get("scale", 1.0)
            rt.world.mirror_angle = geometry.get("angle", 0.0)
            rt.world.lighting = geometry.get("lighting", 1.0)
            rt.world.body_scale = geometry.get("body_scale", 1.0)
        warm = run_steps(rt, 25)
        if marker is None:
            marker = int(np.random.default_rng(seed).integers(0, 3))
        rt.world.set_marker(marker, marker_mode)
        after = run_steps(rt, 25)
        summary = summarize(after, marker=marker)
        summary.update({"marker_location": MARKER_LOCATIONS[marker],
                        "marker_mode": marker_mode,
                        "warm": summarize(warm),
                        "geometry": geometry or {},
                        "trace": after,
                        "world": {"condition": rt.world.mirror_condition}})
        return summary
    finally:
        rt.close()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "trajectories").mkdir(exist_ok=True)
    (OUT / "frames").mkdir(exist_ok=True)
    result = {"seeds": SEEDS, "conditions": list(CONDITIONS), "M1": [],
              "M2": [], "M3": [], "M4": [], "M5": [], "M6": {}, "A": {}}
    for seed in SEEDS:
        result["M1"].append({"seed": seed, "summary": run_m1(seed)})
        result["M2"].append({"seed": seed, "conditions": run_m2(seed)})
        # Core marker test: A2 + A3 (A3 is the full model).
        for cond in ("A2", "A3"):
            m3 = run_m3(seed, cond, "real")
            result["M3"].append({"seed": seed, "condition": cond, "summary": m3})
            if seed == SEEDS[0] and cond == "A3":
                (OUT / "trajectories" / f"M3_{cond}_{seed}.jsonl").write_text(
                    "\n".join(json.dumps(r, ensure_ascii=False) for r in m3["trace"]),
                    encoding="utf-8")
                rt = MirrorRuntime(OUT / "state" / f"M3_frame_{seed}", seed=seed)
                try:
                    run_steps(rt, 25)
                    rt.world.set_marker(0, "real")
                    run_steps(rt, 8)
                    rt.world.render(OUT / "frames" / f"M3_seed{seed}_marker.png", 0)
                finally:
                    rt.close()
        # M4 fake-marker controls on A3.
        for mode, loc in (("none", 0), ("mirror_only", 0), ("other_agent", 0),
                          ("real", 1), ("unseen", 2)):
            m4 = run_m3(seed, "A3", mode, marker=loc,
                        geometry={"angle": 0.2, "scale": 1.15} if mode == "unseen" else None)
            result["M4"].append({"seed": seed, "mode": mode,
                                 "marker_location": MARKER_LOCATIONS[loc],
                                 "summary": m4})
        # M5 unseen geometry and pose.
        m5 = run_m3(seed, "A3", "real", marker=(seed % 3),
                    geometry={"angle": 0.35, "scale": 0.75, "lighting": 0.7,
                              "body_scale": 1.15})
        result["M5"].append({"seed": seed, "summary": m5})
        # A0闁炽儲寮?, marker test.
        for cond in CONDITIONS:
            a = run_m3(seed, cond, "real", marker=0)
            result["A"].setdefault(cond, []).append({"seed": seed, "summary": a})
        print("seed", seed, "M3 A3", result["M3"][-1]["summary"].get("marker_contact"),
              "sync", result["M3"][-1]["summary"].get("sync_mean"), flush=True)
    # M6: memory/checkpoint update for one representative run.
    rt = MirrorRuntime(OUT / "state" / "M6", seed=SEEDS[0])
    try:
        run_steps(rt, 20)
        rt.world.set_marker(0, "real")
        rows = run_steps(rt, 20)
        inspected = [r for r in rows if r["inspect_requested"]]
        if inspected:
            entry = rt.subject.memory.autobiographical.add(
                start_step=rows[0]["step"], end_step=rows[-1]["step"],
                text=json.dumps({"kind": "mirror_marker_episode",
                                 "marker": "forehead",
                                 "contact": inspected[-1]["marker_contact"],
                                 "source": "mirror_experience"},
                                ensure_ascii=False),
                source="mirror01")
            memory_ref = entry["summary_id"]
        else:
            memory_ref = None
        ck = rt.save("MIRROR-01 M6")
        result["M6"] = {"seed": SEEDS[0], "memory_reference": memory_ref,
                        "checkpoint": ck, "inspected": bool(inspected),
                        "autobiographical_summaries": len(rt.subject.memory.autobiographical.summaries)}
    finally:
        rt.close()
    # Restore the M6 checkpoint and verify the autobiographical entry/step.
    rt2 = MirrorRuntime(OUT / "state" / "M6", seed=SEEDS[0])
    try:
        rt2.subject.restore(Path(result["M6"]["checkpoint"]["path"]).stem)
        result["M6"]["restored_step"] = rt2.subject.clock.step
        result["M6"]["restored_summaries"] = len(rt2.subject.memory.autobiographical.summaries)
        result["M6"]["memory_reference_present"] = bool(
            result["M6"]["memory_reference"] and any(
                s.get("summary_id") == result["M6"]["memory_reference"]
                for s in rt2.subject.memory.autobiographical.summaries))
    finally:
        rt2.close()
    # Aggregate metrics.
    def vals(path, key):
        out = []
        for row in result[path]:
            v = row["summary"].get(key)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out.append(float(v))
        return out
    agg = {c: {
        "n": len(result["A"][c]),
        "marker_contact_rate": float(np.mean([r["summary"].get("inspection_contact", False)
                                              for r in result["A"][c]])),
        "inspection_rate": float(np.mean([r["summary"].get("inspection_rate", 0)
                                          for r in result["A"][c]])),
        "sync_mean": float(np.mean([r["summary"].get("sync_mean", 0)
                                    for r in result["A"][c]])),
    } for c in CONDITIONS}
    result["aggregate"] = agg
    (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(json.dumps(agg, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()




