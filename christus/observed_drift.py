"""Subject-side matched WAIT observations. All coordinates are visual pixels."""
import math

import numpy as np


def choose_probe(state, scope, ranked, push_order, step, max_age):
    if not scope:
        return ranked, "ranked"
    bucket = state.get(scope, {})
    if ("wait" not in bucket or step - bucket["wait"]["step"] > max_age
            or len(bucket["pending"]) >= 2):
        return "WAIT", "measure_drift"
    # Do not schedule extra copies that are already awaiting confirmation.
    pending = [x["action"] for x in bucket["pending"]]
    for action in bucket["review"]:
        if action not in pending:
            return action, "retest"
    if bucket["review"] and pending:
        return "WAIT", "confirm_retest"
    if ranked == "WAIT" and not pending:
        for action in push_order:
            if action not in bucket.get("sampled", []):
                return action, "sample_unmeasured"
    return ranked, "ranked"


def observe(state, record, semantic, step, gain):
    scope = record["scope"]
    if not scope:
        return {}
    bucket = state.setdefault(scope, {"pending": [], "review": {}})
    pos = (np.array(record["observation"][:2]) - .5) * 64
    actual = (np.array(record["actual"][:2]) - .5) * 64
    previous = bucket.get("wait")
    if record["action"] != "WAIT":
        if not previous or step - previous["step"] > semantic.staleness_tau_steps:
            return {"unmatched": "no fresh WAIT"}
        if len(bucket["pending"]) >= 2:
            return {"unmatched": "batch full"}
        bucket["pending"].append(dict(tid=record["tid"], action=record["action"],
                                      start=pos.tolist(), distance=float(np.linalg.norm(actual)),
                                      gain=gain))
        return {"baseline_tid": previous["tid"]}

    displacement = actual - pos
    current = dict(tid=record["tid"], delta=displacement.tolist(), step=step)
    bucket["wait"] = current
    batch = bucket["pending"]
    bucket["pending"] = []
    if not previous:
        return {"wait": current}
    # One visual pixel is the resolution used by the velocity context.
    changed = np.linalg.norm(displacement - previous["delta"]) > 1
    stale = step - previous["step"] > semantic.staleness_tau_steps
    rules = semantic.rules.get(scope, {})
    if changed or stale:
        # Round-robin retests. Enough opportunities to overturn unanimous
        # historical evidence, with a finite budget for noisy observations.
        bucket["review"] = {a: {"left": abs(r["support"] - r["counter"]) + 2,
                                "signs": []} for a, r in rules.items()}
        return {"wait": current, "baseline_tid": previous["tid"], "changed": bool(changed),
                "stale": stale, "discarded": [x["tid"] for x in batch],
                "retest_actions": list(bucket["review"])}

    offsets = (displacement + previous["delta"]) / 2
    updates = []
    for item in batch:
        action = item["action"]
        start = np.array(item["start"])
        net = float(np.linalg.norm(start + offsets) - item["distance"])
        sign = 0 if math.isclose(net, 0, abs_tol=1e-9) else int(np.sign(net))
        rule = rules.get(action)
        if sign:
            rule = semantic.record_evidence(scope, action, sign, step=step,
                                            source="coordination.observed_wait",
                                            importance_scale=item["gain"])
        review = bucket["review"].pop(action, None)
        if review:
            review["signs"].append(sign)
            review["left"] -= 1
            agrees = (rule and sign and
                      np.sign(rule["support"] - rule["counter"]) == sign)
            if review["left"] and not (agrees and review["signs"][-2:] == [sign, sign]):
                bucket["review"][action] = review
        updates.append(dict(tid=item["tid"], action=action, net_pixels=net,
                            rule_after=dict(rule) if rule else None))
        sampled = bucket.setdefault("sampled", [])
        if action not in sampled:
            sampled.append(action)
    return {"wait": current, "baseline_tid": previous["tid"], "updates": updates}
