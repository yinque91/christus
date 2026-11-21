# CHRISTUS-MIRROR-01 Implementation Report

## Code

| File | Role |
|---|---|
| `christus_mirror/world.py` | true reflection geometry, 2D renderer, body sensors, markers, delayed/other/no mirror |
| `christus_mirror/subject.py` | `MirrorSubject` (existing Subject), `MirrorPolicy`, action hook |
| `christus_mirror/run.py` | M1–M6, A0–A5, 20 seeds, raw traces |
| `mirror01/analyze.py` | mean/std/95% CI analysis |
| `christus_mirror/lab_app.py` | native Tkinter Mirror Lab window |
| `christus_mirror/tests/test_mirror.py` | 6 tests |
| `START_MIRROR_LAB.bat` | native experiment viewer |

## Subject path

`MirrorSubject` is the existing `wphr.runtime.Subject` with a mirror visual
adapter. The action path is:

```text
world.observe → Subject core/SM-CTRN → MirrorPolicy.act
→ world.execute → next observation → trace
```

The environment never supplies an action. The policy uses the live core state
for ordinary actions. A visible marker plus sufficient action–visual
contingency can request a body-part touch. The marker-to-touch mapping is a
functional body schema, not an evaluator label.

## Constraints

- One Subject per run; no second agent.
- No W/P/Q modification.
- No condition/marker/answer label in the subject observation.
- No external LLM.
- No website; viewer is native Tkinter.

## Deviations/limitations

- A0 is a **no-mirror base**, not the full SELF-02 provenance stack.
- Conditions use the same seed and initialization, but not a single shared
  checkpoint object across every branch.
- M1 action selection is partly core-state/random exploration; the marker
  inspection is an explicit functional policy, not fully emergent.
- The visual channel is a feature-level 2D sensor with rendered frames, not a
  3D photorealistic camera.
