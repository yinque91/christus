# Research program

Christus is developed as a sequence of controlled experiments rather than a single
benchmark. The table summarizes the principal lines of evidence.

| Programme | Question | Key result |
|---|---|---|
| E-series | Can the agent learn embodied navigation and generalize across starts/maps? | Goal-conditioned success 0.799 vs 0.306 random; shared representations 0.614 vs 0.429 |
| E010 | Does integrated state/meaning/language/memory improve behaviour? | Complete architecture reward 16.478 vs 4.607 baseline; ceiling effects limit component inference |
| P1 | Does temporal information improve occlusion prediction? | Temporal ordering improved prediction; recurrent core did not beat strongest multi-frame baseline |
| CORE-READOUT-01 | Can the recurrent state guide physical action? | Catch rate 0.607 vs 0.277 disabled; directional alignment 0.646 |
| SELF-01 | Can the subject learn an action–body relation? | Δpaddle = −0.2533 + 1.5059a; confidence 0.9625; checkpoint-stable |
| SELF-02 | Can experience provenance and autobiographical continuity be maintained? | 0.8333 self/external; 96 autobiographical references; 0.7143 unseen |
| SELF-03 | Does a latent self-related representation emerge without source labels? | 0.6355 source-probe accuracy vs 0.25 chance; partial, not a stable abstract self-model |
| AFFECT-02 | Does affect modulate attention and recurrent state? | Attention Δ0.000464; recurrent Δ9.54×10⁻⁵; no final-action change |
| MIRROR-01 | Can a mirror guide inspection of the subject's own body? | 20/20 seeds correct contact and localization; replicated under altered geometry |

Detailed reports are in `docs/`.
