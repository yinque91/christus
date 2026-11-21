# CHRISTUS-MIRROR-01 Environment Report

## Geometry

The mirror is a reflection about `x = 32`:

```text
mirror_x = 32 − (body_x − 32) · scale · cos(angle) + y · sin(angle) · 0.15
mirror_y = body_y · body_scale − (body_x − 32) · sin(angle) · 0.15
```

A test verifies that body point `(20, 40)` reflects to `(44, 40)`.

## Visual channels

- `visual`: mirror silhouette positions, marker pixels, mirror motion,
  lighting/scale/angle, first-person hand/head features.
- `body`: proprioceptive part positions.
- Marker is never placed in the first-person own-body channel.
- Condition labels and evaluator truth are not in the observation dictionary.

## Mirror conditions

- realtime;
- delay 0.5 s (4 steps);
- delay 2 s (8 steps);
- other-agent video (different silhouette);
- no mirror (visual channel zeroed).

## Body and actions

Body parts: head, left/right hand, torso. Actions: head left/right, hand
left/right, turn left/right, step left/right, look mirror/away, touch forehead,
touch left/right hand.

## Rendered record

`mirror01/frames/M3_seed12000_marker.png` is a real rendered frame from the M3
marker episode. Raw step traces are in `mirror01/results.json`; the M3 seed-0
trace is also in `mirror01/trajectories/M3_A3_12000.jsonl`.

## Limitation

This is a lightweight 2D feature-level environment, not a photorealistic 3D
mirror. It preserves reflection geometry, action-contingent motion, delay,
other-agent and marker visibility, but it does not model lighting optics or
full-body appearance.
