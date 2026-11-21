# CHRISTUS-MIRROR-01 Design

## Question

Can the existing Christus Subject discover the relation between its actions and
a mirror image, use the mirror to find a hidden body marker, localize it to its
own body, and retain the experience, without being told that the mirror is its
own reflection?

## Environment

A local 2D rendered mirror environment with:

- a virtual body (head, left/right hand, torso);
- a true reflection transform around a mirror axis;
- first-person and mirror visual channels;
- body sensors;
- 13 actions (head/hand/turn/step/look/touch);
- hidden markers on forehead/left hand/right hand;
- realtime, delayed, other-agent and no-mirror conditions;
- evaluator-only truth.

The subject receives feature vectors extracted from the rendered scene and the
body sensors. It never receives a condition label, marker ground truth, mirror
identity label or hidden coordinates.

## Conditions

- A0: no mirror, base Subject action policy.
- A1: realtime mirror, no body inspection module.
- A2: A1 + body schema inspection.
- A3: A2 + self-prediction flag.
- A4: A3 with shuffled action–visual temporal correspondence.
- A5: A3 with body sensor feedback removed.

## Experiments

- M1 free exploration / action–mirror correspondence.
- M2 realtime vs delayed vs other-agent vs no mirror.
- M3 hidden marker after familiarization.
- M4 fake marker, no marker, other-agent marker, other body part, unseen pose.
- M5 unseen mirror size/angle/body scale/lighting.
- M6 autobiographical memory and checkpoint retention.

## Success boundary

Behavioural evidence is required: actual touch/contact and localization, not
only a verbal report. The experiment cannot establish subjective experience.
