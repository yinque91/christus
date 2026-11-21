# CHRISTUS-MIRROR-01 M3 Report — Hidden Marker

## Actual 20-seed result (A3; A2 identical)

| Metric | Mean | s.d. | 95% CI |
|---|---:|---:|---|
| Marker seen in mirror | 1.000 | 0.000 | [1.000, 1.000] |
| Inspection rate | 0.646 | 0.266 | [0.529, 0.763] |
| Correct contact after inspection | **1.000** | 0.000 | [1.000, 1.000] |
| Target localization correct | **1.000** | 0.000 | [1.000, 1.000] |

## What happened

After 25 familiarization steps, a marker was set at a random location
(forehead, left hand or right hand). The subject's visual channel contained the
marker only in the mirror. In every seed, the policy detected the marker and,
when it inspected, touched the correct body part (perfect localization).

## Pre-registered criteria

- real marker contact ≥ 0.70: **PASS** (1.000)
- localization ≥ 0.70: **PASS** (1.000)

## Critical limitation

The inspection policy is a functional marker-contingent controller, not a fully
emergent self-recognition mechanism. Its marker-to-body mapping is explicit.
M4 shows the crucial failure: it also inspects a **mirror-only fake marker**.
Therefore M3 alone is not sufficient evidence of reliable self-mirror
recognition.

**M3: PASS for real-marker contact/localization; not sufficient alone.**
