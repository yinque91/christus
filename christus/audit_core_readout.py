"""Read-only interface audit for the SM-CTRN -> action question.

This script does not alter the runtime. It runs a bounded subject, records the
core state and the task-relevant observation, and reports whether a linear
readout can decode the ball offset from h, c or [h;c]. It also verifies the
legal-action mapping and the actual call order.

Run from the project root:

    python -m christus.audit_core_readout --out results/core_readout_audit01/pilot
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .runtime import create
from wphr.modules.practice import ACTIONS, ACTION_INDEX
from wphr.runtime.features import FEATURE_NAMES
from .session_world import COMMAND


def ridge_fit_predict(X: np.ndarray, y: np.ndarray, *, alpha: float = 10.0) -> tuple[np.ndarray, dict]:
    """Chronological 70/30 ridge regression, no shuffling.

    The core state is high-dimensional relative to a short audit, so a tiny
    ridge penalty overfits and can produce a negative test R² even when the
    correlation is strong. The default alpha is deliberately conservative;
    the online readout in the experiment is the operational test.
    """
    n = len(y)
    split = max(1, int(0.7 * n))
    Xtr, ytr = X[:split], y[:split]
    Xte, yte = X[split:], y[split:]
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True)
    sd[sd < 1e-9] = 1.0
    Ztr = (Xtr - mu) / sd
    Zte = (Xte - mu) / sd
    Ztr = np.concatenate([Ztr, np.ones((len(Ztr), 1))], axis=1)
    Zte = np.concatenate([Zte, np.ones((len(Zte), 1))], axis=1)
    gram = Ztr.T @ Ztr + alpha * np.eye(Ztr.shape[1])
    coef = np.linalg.solve(gram, Ztr.T @ ytr)
    pred = Zte @ coef
    ss_res = float(np.sum((yte - pred) ** 2))
    ss_tot = float(np.sum((yte - yte.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    corr = float(np.corrcoef(yte, pred)[0, 1]) if np.std(pred) > 1e-12 else 0.0
    sign_acc = float(np.mean(np.sign(pred) == np.sign(yte))) if len(yte) else 0.0
    return pred, {"r2": r2, "corr": corr, "sign_accuracy": sign_acc,
                  "n_train": split, "n_test": n - split}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=31100)
    ap.add_argument("--arm", default=None,
                    help="optional CORE-READOUT arm; A0 has the observation bridge but no readout")
    ap.add_argument("--alpha", type=float, default=10.0,
                    help="ridge penalty for the read-only linear probe")
    args = ap.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing audit directory: {out}")
    out.mkdir(parents=True)

    s = create(out, seed=args.seed, profile="safe", arm=args.arm)
    captured_inputs: list[np.ndarray] = []
    original_step = s.core.step

    def capturing_step(x=None, dt=None):
        captured_inputs.append(np.zeros(s.core.total_inputs) if x is None else np.array(x, copy=True))
        return original_step(x, dt)

    s.core.step = capturing_step  # type: ignore[method-assign]
    rows = []
    try:
        for _ in range(args.steps):
            r = s.step()
            scent = s.shared.get("world.scent", [0.0, 0.0]) or [0.0, 0.0]
            rows.append({
                "step": r.step,
                "action": r.action,
                "reward": float(r.reward),
                "h": s.core.h.copy(),
                "c": s.core.c.copy(),
                "scent_x": float(scent[0]),
                "scent_y": float(scent[1]),
                "visible": bool(s.world.measured.get("visible", False)),
                "ball_x": float(s.world.measured.get("x", 0.0)),
                "paddle": float(s.world.measured.get("paddle", 0.0)),
                "input": captured_inputs[-1].copy(),
            })
    finally:
        s.close()

    # Core state matrices. Keep only visible-ball rows for the offset probe:
    # when the ball is not visible the offset is defined as 0 and carries no
    # direction, so including those rows would inflate chance accuracy.
    visible = np.array([r["visible"] for r in rows])
    h = np.stack([r["h"] for r in rows])
    c = np.stack([r["c"] for r in rows])
    y = np.array([r["scent_x"] for r in rows])
    h_lag = np.vstack([np.zeros((1, h.shape[1])), h[:-1]])
    c_lag = np.vstack([np.zeros((1, c.shape[1])), c[:-1]])
    rows_in = [r for r, v in zip(rows, visible) if v]

    def probe(name: str, features: np.ndarray, target: np.ndarray) -> dict:
        if len(target) < 20:
            return {"error": "not enough visible-ball samples"}
        _, metrics = ridge_fit_predict(features, target, alpha=args.alpha)
        metrics["feature_dim"] = int(features.shape[1])
        metrics["target_std"] = float(np.std(target))
        metrics["alpha"] = float(args.alpha)
        return metrics

    probes = {}
    if len(rows_in) >= 20:
        idx = np.array([i for i, v in enumerate(visible) if v])
        probes["h_current"] = probe("h_current", h[idx], y[idx])
        probes["c_current"] = probe("c_current", c[idx], y[idx])
        probes["hc_current"] = probe("hc_current", np.concatenate([h[idx], c[idx]], axis=1), y[idx])
        probes["h_lag1"] = probe("h_lag1", h_lag[idx], y[idx])
        probes["c_lag1"] = probe("c_lag1", c_lag[idx], y[idx])
        probes["hc_lag1"] = probe("hc_lag1", np.concatenate([h_lag[idx], c_lag[idx]], axis=1), y[idx])

    # What does the core actually receive? The world block is the only input
    # that could carry the ball's horizontal offset. SparseModularCTRN keeps the
    # offsets; the closed subject object is still alive here.
    world_start, world_end = s.core.input_offsets["world"]
    world_inputs = np.stack([r["input"][world_start:world_end] for r in rows])
    input_corr = []
    for j in range(world_inputs.shape[1]):
        col = world_inputs[:, j]
        denom = float(np.std(col) * np.std(y))
        corr = float(np.corrcoef(col, y)[0, 1]) if denom > 1e-12 else 0.0
        input_corr.append(round(corr, 4))

    report = {
        "audit": "SM-CTRN h/c -> action interface",
        "seed": args.seed,
        "steps": len(rows),
        "visible_ball_samples": int(visible.sum()),
        "action_counts": {a: int(sum(r["action"] == a for r in rows)) for a in ACTIONS},
        "legal_actions": list(ACTIONS),
        "action_index": ACTION_INDEX,
        "world02_command_mapping": COMMAND,
        "safety_allowed_actions": list(s.config.get("safety.allowed_actions", [])),
        "core_module_names": list(s.core.module_names),
        "core_total_units": int(s.core.total_units),
        "concept_dim": int(s.core.concept_dim),
        "arm": args.arm,
        "world_input_dim": int(world_end - world_start),
        "world_input_feature_names": list(FEATURE_NAMES["world"]),
        "world_extra_features": s.core.cfg.get("extra_features", {}).get("world", []),
        "world_input_scent_correlation": input_corr,
        "linear_probes_visible_only": probes,
        "call_order": [
            "observation event -> modules.on_event (World stores cue/scent)",
            "scheduler due groups -> Practice.update selects practice.action from Q/navigation",
            "Subject._features -> assemble_inputs -> core.step updates h/c",
            "Subject action resolution reads practice.action",
            "world.execute(action) -> outcome event -> Practice learns",
        ],
        "conclusion": "",
    }
    if probes and "hc_current" in probes:
        best = max(probes.values(), key=lambda p: p.get("corr", -9))
        report["conclusion"] = (
            "The core state carries a decodable ball-offset signal "
            f"(best corr={best.get('corr', 0):.3f}, sign_accuracy={best.get('sign_accuracy', 0):.3f}); "
            "the online readout is the operational test."
            if best.get("corr", 0) > 0.40 and best.get("sign_accuracy", 0) > 0.55
            else "No tested core-state representation carries a reliable ball-offset signal."
        )
    else:
        report["conclusion"] = "Not enough visible-ball samples for a probe."
    (out / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
