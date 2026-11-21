# CHRISTUS-MIRROR-01 M5 Report — Unseen Geometry

## Actual 20-seed test

Unseen mirror angle 0.35, scale 0.75, lighting 0.7 and body scale 1.15. A real
marker was placed on a random body part.

| Metric | Mean | s.d. | 95% CI |
|---|---:|---:|---:|
| Correct contact after inspection | 1.000 | 0.000 | [1.000, 1.000] |
| Target localization correct | 1.000 | 0.000 | [1.000, 1.000] |
| Inspection rate | 0.632 | 0.251 | [0.522, 0.742] |

Pre-registered threshold ≥ 0.70: **PASS**.

## Caveat

The geometry was changed, but the marker-to-action mapping is invariant in the
policy. This shows the control rule transfers across these geometry changes; it
does not show that a learned visual body schema generalized. M4's fake-marker
failure remains decisive.

**M5: PASS for real-marker transfer under the functional policy.**
