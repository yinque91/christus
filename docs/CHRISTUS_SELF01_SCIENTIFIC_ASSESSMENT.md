# CHRISTUS-SELF-01 Scientific Assessment

## 1. What was actually established

### PASS

- A subject-side BodySelfModel predicts its own paddle displacement from its own
  action using only observable paddle positions.
- Calibration learned a real self-body relation:
  `Δpaddle = -0.2533 + 1.5059 * signed_action`.
- 77 body samples, 75 updates, confidence 0.9625.
- Body prediction error and version are recorded.
- Body and external ball motion are separated: external log 40 entries, body
  relation only uses action/own displacement.
- Body model state, pending transition, samples and relation survive checkpoint
  restore.
- SM-CTRN W/P/Q hash unchanged.
- Body candidate scores are actually computed and nonzero.

### PARTIAL

- Body candidate score enters the total candidate score (e.g. west +1.756,
  east -1.256), but in the tested target the existing Practice base strongly
  favors `move_south`; body term does not change the winner.
- Therefore body prediction is **computationally present but not causally
  decisive** for action in this target.

### FAIL

- A0 / A1 / A2 produce identical actions and outcomes:
  - east 18, south 22;
  - caught 1; mean reward 0.0178.
- A1 (frozen body update) changes body error 0.3106 → 0.3414 but not action.
- A2 (body readout off) changes body score to zero but not action.
- No body→action or body→outcome causal effect was observed.
- No ambiguous target was separately validated in this first run; the chosen
  clear-base target is one where the body model cannot override.

### NOT DONE / BLOCKED

- `CHRSTUS-03 CoordinationModule` is a BallWorld 5-action module; SessionWorld has
  6 actions. It was not used. SELF-01 uses the existing Session decision path
  (`ChristusSubject` / `IntegratedPractice` / `practice.last_scores`) plus
  `BodyCoordination` as the sole action hook. This is a **SESSION_DECISION_ADAPTER**,
  not the CHRSTUS-03 CoordinationModule.
- Interaction log is empty because SessionWorld's outcome does not expose
  `contact`; not fabricated.
- No hidden physics parameters used.
- No large multi-seed experiment.

## 2. Interpretation

The experiment demonstrates a **body self-model**, but not **body self-agency in
action selection** for this target. The body model can predict its own movement
and can produce a candidate score, but the existing base policy dominates.

This is not a reason to increase `body_weight` post hoc. The correct next step
would be a pre-defined target where base action scores are close enough that the
body model's already-computed contribution is decisive, plus a body-gain change
that makes frozen vs updating models differ in action. That test was not run in
this first version.

## 3. Self/external/interaction

| relation | status |
|---|---|
| BODY: own action → own displacement | PASS |
| EXTERNAL: ball motion logged separately | PASS |
| INTERACTION: action + contact | NOT OBSERVED (no contact field) |
| body relation polluted by external motion | NO |

## 4. Boundary

No claim of:

- self-awareness;
- body ownership;
- subjective agency;
- consciousness;
- general autonomous self-model.

The honest status is:

```text
BODY_SELF_MODEL_LEARNED       = PASS
BODY_MODEL → CANDIDATE_SCORE  = PARTIAL
CANDIDATE → ACTION            = FAIL
ACTION → OUTCOME              = FAIL
CHECKPOINT                    = PASS
BODY/EXTERNAL SEPARATION      = PASS
INTERACTION                   = NOT OBSERVED
COORDINATION_MODULE           = NOT USED
```

## 5. Required next condition (not claimed as done)

A pre-registered ambiguous target where:

```
base_score(A) - base_score(B) is smaller than the body-model contribution
```

and body gain changes between checkpoint and evaluation, so A0 can adapt while
A1 is frozen. Only then can SELF-01 test body→action causality cleanly.
