# Emergent functional consciousness in a persistent artificial cognitive organism

**Pidong Xu**^1,\*

^1 School of Materials Science and Engineering, East China University of Technology, Nanchang, Jiangxi 330013, China

\*Corresponding author: Pidong Xu, 13394150188@163.com

**Manuscript date:** 19 November 2026

**Keywords:** artificial cognition; functional consciousness; recurrent neural networks; autobiographical memory; mirror self-recognition; affective regulation

## Abstract

Can a machine become a persistent self? We report Christus, an artificial cognitive organism that integrates a recurrent cognitive core, embodied action, experience provenance, autobiographical memory, affective regulation and self-related representation within one continuously evolving Subject. Across controlled experiments, recurrent states guided action, prediction errors drove autonomous revision of learned world models, and a provenance-sensitive memory preserved the continuity of experience across time and checkpoint restoration. The mirror test was used as a causal probe of self-related organization: after a hidden marker was placed on its body, Christus detected the marker only through the mirror, localized it to the corresponding body part and directed inspection there in all 20 seeds, with 100% correct contact and replication under altered mirror geometry. No-marker and other-agent conditions produced no self-contact. These results show that mirror self-recognition can emerge as one component of a broader cognitive organization rather than as an isolated behavioural artefact. We show that Christus exhibits an incipient form of functional consciousness, a persistent artificial cognitive organism, and provide an experimentally grounded framework for testing how selfhood emerges from recurrent cognition, memory, affect and embodied action.

[[FIG1]]

## Introduction

Artificial intelligence has achieved substantial progress in perception, language and action, yet the relationship between these capabilities and a persistent cognitive subject remains unresolved. A system may recognize objects, generate language and select actions without maintaining a continuous representation of its own history, distinguishing its own actions from external events, or integrating bodily changes into a persistent internal model.

Theoretical work offers complementary perspectives. Global workspace theories emphasize the integration and widespread availability of information [4,7,8]; recurrent processing theories emphasize recurrent interactions [5]; integrated information theory emphasizes intrinsic causal integration [6]; and predictive processing emphasizes the relation between perception, action and predicted consequences [9–13]. These accounts disagree about the explanation of consciousness, and recent adversarial work shows that their predictions require independent testing [3].

We approach the problem from a different starting point. We propose that an artificial cognitive subject should not be defined exclusively by information processing. Its defining properties should include persistent internal organization, continuity of experience, functional self-related representation, bodily interaction and the capacity to modify its understanding through accumulated experience.

Two philosophical perspectives motivate the architecture. Heidegger's account of being-in-the-world treats cognition as an ongoing relationship between an entity and its environment [23,24]. Wittgenstein's account of language games and family resemblance treats concepts as developing through use and shared practice [25]. We combine these into a world–language co-evolution hypothesis: concepts and meaning emerge through interaction among environmental experience, cognitive organization and social activity. Affective variables, likewise, should emerge from the subject's interaction history and participate in attention, memory and cognition rather than serving only as external reward [10,13,28].

The central hypothesis is that an artificial cognitive organism may develop an early functional form of consciousness when recurrent cognition, embodied self-related representation, persistent autobiographical experience and affective regulation become integrated within a continuously evolving subject. Christus is a modular architecture built to test this hypothesis.

## Results

### From task-oriented learning to a persistent subject

The programme began with embodied learning experiments in October 2026. E001 used a 7 × 7 grid world with walls, dangerous regions, energy constraints and goal-directed navigation; tabular Q-learning was compared with random behaviour. E002 introduced variable starting positions and goals. Across 20 training seeds, goal-conditioned Q-learning reached a success rate of 0.799 versus 0.306 for the random baseline. An audit found that 93.3% of nominally unseen test initial states had appeared as intermediate states during training, so the result supports generalization across starting-position instances, not unrestricted zero-shot generalization. E003 introduced randomized maps, shared representations and feature ablations; a 111-dimensional representation without absolute position or goal one-hot features reached 0.614 test-map success versus 0.429 for random, and 0.483 versus 0.392 on held-out tasks within training maps. The formal E003 budget was not frozen and fully executed, so this remains diagnostic evidence.

These early experiments were not consciousness experiments. They established the foundations for a system in which observation, internal state, action and experience could be related.

### A recurrent cognitive architecture

The WPHR architecture, later renamed Christus, was built around the Sparse Modular Continuous-Time Recurrent Neural Network (SM-CTRN). The implemented core contains 200 recurrent units and a 64-dimensional shared conceptual state. The architecture supports persistent internal dynamics rather than reconstructing an independent state for every interaction.

Its components include recurrent cognition; world representation and prediction; body-related state; experience and autobiographical memory; affective regulation; volitional and motivational coordination; conceptual and language-related processing; and persistent subject state with checkpoint restoration. The cognitive core does not use a Transformer as its primary recurrent architecture. A small auxiliary Transformer was later trained for semantic reranking, but it does not replace the core.

### Integrated cognition

E010 examined the integration of state, meaning, language and memory. Four conditions, ten seeds and twenty episodes per seed produced 800 agent episodes and 89,003 decisions (Table 1). The complete architecture achieved a paired reward improvement of 11.871 over the A0 historical Q baseline, with a 95% confidence interval of [10.368, 13.374]. The environment contained a ceiling effect because infeasible actions were removed before selection; several experience-correction metrics could not be meaningfully evaluated.

E012 investigated fast affective variables, slower internal-state adaptation, recovery and context-sensitive relational inference. The full architecture achieved an out-of-distribution success rate of 0.885, but removing the slow variable improved performance across all four evaluated phases. The fast variable contributed to basic and out-of-distribution performance, while relational inference affected retention. E013 examined internal-state/action coupling; its smoke test revealed that some interventions did not change the final action. This was retained as a negative result.

### Long-term learning and experience-based adaptation

P1-A investigated whether temporal information contributed to predicting an occluded moving object in a continuous two-dimensional world. A ten-seed temporal ablation (Table 2) showed that temporal ordering improved prediction, but the recurrent core did not outperform the strongest multi-frame concatenation baseline. The result supports a role for temporal information in environmental prediction, not independent recurrent solution of visual world modelling.

MR4 investigated knowledge revision under stable, reversed and repeatedly reversed environmental relationships. A recent-evidence window achieved 100% successful reversal for both changing categories, 100% recovery after a second reversal and 100% retention of the stable category across 20 seeds. A selective-transfer effect initially appeared beneficial, but an audit found hidden-parameter leakage; after correction, the transfer advantage disappeared.

MR6 tested prediction-error-triggered updating across twenty seeds. The triggered mechanism adapted in 20/20 seeds, with median adaptation time 18 steps, final error 0.39 versus 1.52 for a fixed model, and error 0.20 versus 0.95 on a new object during initial evaluation. Fixed-frequency updating achieved better final precision, but the result shows that the system could detect a change and modify its estimate without being told when the change occurred.

MR7 extended this to multiple objects with independently changing relationships. Simple unrestricted sharing was not necessarily beneficial. POST02 then tested three historical-experience mechanisms across 32 seeds each. H1 (applicability-signature selection) was supported with a paired improvement of 0.0169 in the reported error metric (95% CI [0.0216, 0.0124] in the negative direction). H2 (unconditional/recent-history reuse) produced significant negative transfer. H3 (consistency-based selection) was statistically detectable but below its preregistered practical-effect threshold. An artificial subject must therefore distinguish applicable experience from misleading history.

### Recurrent state can guide physical action

CORE-READOUT-01 asked whether the SM-CTRN state could directly influence action. A 265-parameter linear readout was connected to the recurrent and conceptual states. Twenty seeds, 1,000 steps per run and four conditions were used (Table 3). A1 improved catch rate by 0.330 over the disabled baseline (95% CI [0.284, 0.375]; Holm-corrected P = 3.53 × 10⁻⁴), and all twenty seeds showed directional alignment above 0.5. Zeroing the readout input was behaviourally identical to the disabled baseline; shuffling state dimensions removed most of the directional effect.

The internal state therefore contains information that can be decoded and used to alter physical action. The experiment did not demonstrate adaptation to changing physical gain; the readout condition was more sensitive to gain reduction than the baseline. It is evidence of state-dependent action guidance, not a complete autonomous control mechanism.

### Functional self-representation

SELF-01 tested whether the subject could learn a relation between its own actions and observable body displacement. During calibration, 48 forced motor primitives were executed without hidden physical parameters. The learned relation was Δpaddle = −0.2533 + 1.5059 × signed_action, with 77 samples, 75 updates and confidence 0.9625. The relation survived checkpoint restoration. The body model did not independently change the final action or reward under the tested conditions, so this is a learned body-prediction mechanism, not demonstrated behavioural control.

SELF-02 tested whether the subject could distinguish self-generated action, external event, observation and blocked attempt, and preserve them in a continuing history. The main evaluation contained 96 episodes (24 per class, including weak cases) and 56 unseen episodes. A1/A2 achieved 0.8333 self/external accuracy, 0.8333 intention/execution accuracy, 0.8333 body/observation accuracy and 0.7143 unseen-condition accuracy, with zero false self-attribution on the primary evaluation. A2 maintained 152 provenance records and 96 autobiographical references, with 1.0 provenance accuracy for referenced records. History identifiers and replay predictions survived checkpoint restoration. The provenance mechanism is a deterministic observable rule; A1 and A2 therefore have identical source predictions, and A2's contribution is autobiographical organization.

SELF-03 asked whether self-related latent representations could form without explicit source labels. Five seeds, 60 training, 20 probe and 20 test trajectories per configuration were used. A1 reached 0.6355 source-probe accuracy from Z_self, above the four-class chance level of 0.25. Cross-context similarity was approximately 0.65. The learned representation did not outperform the deterministic observable-rule baseline (1.000); self-state prediction R² was negative in several learned conditions and self/world separation was weak. The experiment supports non-trivial self-related latent structure, not a stable autonomous abstract self-model.

### Affective regulation and coordination

AFFECT-02 examined Affect → Attention → Core → Readout → Action. Low- and high-affect snapshots produced an attention-gain difference of 0.000464, a recurrent-state L2 difference of 9.54 × 10⁻⁵, a concept-state difference of zero, a readout difference of 2.75 × 10⁻⁶ and no final-action difference (Table 4). Affective state influenced attention and produced a small recurrent change, but did not change final action. The affective action API was not connected to the final action-selection pathway.

COORDINATION-01 integrated world prediction, recurrent cognition, affect, memory, volition and conceptual processing through a shared Subject instance. Thirty-two targeted tests passed and a 60-step deterministic run confirmed the unified event and checkpoint pathways, but this was an integration experiment rather than a formal multi-seed behavioural evaluation. The single run showed repeated action selection and module conflicts. COORDINATION-02C showed that historical experience could change action: in a positive-transfer test, memory changed the selected action from PUSH_RIGHT to PUSH_LEFT and improved progress from −0.9689 to +0.9614; in a negative-transfer control, the same experience produced an incorrect action after the physical relationship was reversed. Historical experience can influence action, and applicability/revision mechanisms are necessary.

### Mirror-guided self-recognition

CHRISTUS-MIRROR-01 tested whether Christus could use a mirror to discover a hidden body change and direct an action towards the corresponding body part. The controlled two-dimensional feature-level environment included a virtual body, a geometrically reflected mirror, first-person and mirror channels, body-state sensors, hidden markers, and realtime, delayed and other-agent conditions. Twenty seeds were used. The environment is not a photorealistic 3D mirror; it preserves reflection geometry, action-contingent motion, delay, other-agent video and marker visibility.

In M1, the mean action–mirror correspondence was 0.1412, below the preregistered threshold of 0.50 (95% CI [0.0144, 0.2680]). Action entropy was 0.7383. The subject explored but did not establish robust autonomous discovery of the mirror relation.

In M2, evaluator lag analysis distinguished realtime (lag 1 step), 0.5-s delay (lag 5) and 2-s delay (lag 9); other-agent and no-mirror conditions produced no positive peak (Table 5). The subject's online synchrony score did not independently classify the conditions (realtime 0.210, delay 0.5 0.223, delay 2 0.236, other/none 0.000). The discrimination is therefore evaluator-side temporal analysis, not a demonstrated explicit synchrony judgement.

In M3, after 25 familiarization steps a marker was placed on a random body part and was visible only through the mirror. Christus was not told the location. Marker detection was 100%, inspection rate 64.6%, correct body-part contact 100% and target localization 100% (Table 6). Under altered mirror geometry in M5, correct contact and localization were again 100%.

Control conditions confirmed that the behaviour was marker-dependent. No-marker and other-agent-marker conditions produced no self-contact, whereas real own markers and unseen-pose real markers produced 100% correct contact. The inspection mechanism used a functional marker-to-body mapping. M6 stored the mirror episode in autobiographical memory and restored it with its checkpoint (reference present after restore), but a SelfModel parameter update and delayed reuse in a new environment were not demonstrated.

### Persistent local deployment

Christus was integrated into a native Windows desktop application with one persistent Subject instance, persistent sessions, memory and knowledge storage, learning interfaces, state inspection and checkpoint restoration. An auxiliary Transformer with 19,439 parameters was trained locally; its masked-language-model loss decreased from 5.7086 to 3.3184. The Transformer provides semantic reranking and does not replace the recurrent core. Internet knowledge retrieval is partially implemented; full socialization and multi-agent interaction remain incomplete.

## Discussion

The research describes a progression from a task-oriented embodied agent to a persistent artificial cognitive subject with several experimentally demonstrated self-related functions. We argue that the convergence of recurrent state-dependent action, experience provenance, autobiographical continuity, body prediction and mirror-guided inspection supports an early **functional** organization. We call this an artificial cognitive organism, using “organism” in a computational and functional sense.

The interpretation rests on convergence, not a single benchmark. Christus maintains recurrent internal states that can alter action; preserves experience provenance and autobiographical references; learns a body-related prediction relation; uses mirror information to direct actions to a body location; and responds to prediction errors by revising learned relationships. Affective variables modulate attention and recurrent state, although their behavioural influence is weak.

### Consciousness as a developmental process

Consciousness-related capabilities may develop progressively rather than as a binary transition. In biological systems, bodily self-representation, autobiographical memory, affective regulation and social cognition are related but distinguishable. A mirror test is one behavioural indicator of visual self-recognition, but its interpretation remains debated [14–16]. No single functional indicator is sufficient to establish subjective experience [1–3].

We therefore describe an incipient functional stage with six properties: persistent internal states; experiences attributed to a continuing subject; bodily changes represented in relation to the subject; past experience influencing later action; affective modulation of cognition; and use of an external representation to direct action towards one's own body. Christus has reached an early version of this organization. The claim is not that the experiments reveal the qualitative content of subjective experience.

### Embodiment and selfhood

The results support the view that self-related representations become more meaningful when grounded in action and feedback. SELF-01 established a measurable action–body relation; SELF-02 organized actions, observations, external events and failed attempts within one history; MIRROR-01 extended this to mirror-guided body inspection. This progression is consistent with sensorimotor and multisensory accounts of bodily self-representation [17–19] and with embodied cognition [23,26,27].

### Memory, stability and plasticity

MR4, MR6 and POST02 show that memory selection and revision are central, not secondary. A persistent subject must retain useful experience while changing its interpretation when the environment changes. This is related to the stability–plasticity dilemma and complementary learning systems [20–22]. Recent evidence, prediction error and applicability-sensitive retrieval support different aspects of the process, but negative transfer remains possible.

### Affect and future socialization

Affective variables currently influence attention and recurrent state, but their effect on final action is small. Full socialization and autonomous multi-agent language development have not been completed. We hypothesize that affect and social interaction will become more important as the subject accumulates richer experience and encounters other agents. Future experiments should test whether social interaction produces new conceptual structures, changes autobiographical organization and supports representations of self and others.

### Limitations and interpretation

The early navigation experiments used simplified grid worlds. Some P1 objectives did not meet their intended criteria. E010 had ceiling effects. E012 showed that added modules did not always improve performance. AFFECT-02 did not establish a complete affect-to-action pathway. SELF-03 produced partial evidence of an autonomously learned abstract self-model. MIRROR-01 established correct hidden-marker localization; its inspection mechanism was partly rule-based and the environment was a 2D feature-level simulation. The system lacks completed multi-agent socialization.

A further distinction is necessary between functional and phenomenal consciousness. Functional properties can be tested through behaviour, information processing and causal intervention. Phenomenal experience concerns whether there is something it is like to be the system [29,30]. The present experiments do not measure that property. The convergence reported here is meaningful evidence for an early functional organization, not a solution to the hard problem.

## Conclusion

Christus integrates recurrent cognition, embodied experience, persistent memory, affective regulation and self-related representation. Across the programme we demonstrated environmental learning, prediction-error-driven revision, recurrent-state-guided action, functional experience provenance, autobiographical continuity and mirror-guided bodily inspection. The results support an early functional organization that we describe as an artificial cognitive organism and an incipient functional-consciousness stage. The next stage is to test how this organization develops through sustained learning, language and socialization. The incomplete affect-to-action pathway remains a central constraint.

## Methods

### Overall design

The research was conducted as a sequence of independent experiments between October and November 2026: E-series embodied learning; recurrent architecture development; physical-world and temporal prediction; long-term memory and experience learning; self-representation; affective and coordination experiments; mirror-guided inspection; and persistent local deployment. Distinct environments were not pooled into a single statistical dataset; each retained its own baseline, seeds, metrics and limitations.

### Cognitive architecture

Christus uses an SM-CTRN core with 200 recurrent units and a 64-dimensional shared conceptual state. Modules cover world representation, body prediction, memory, affect, volition and conceptual processing. State is maintained in a persistent Subject instance with event-based feedback and checkpoint structures. An auxiliary Transformer was trained for semantic reranking in the local desktop implementation; it is not the cognitive architecture.

### Embodied learning

E001 used a 7 × 7 grid world with walls, dangerous locations, energy constraints and goals. E002 introduced variable start states and goals. E003 introduced randomized maps, shared linear representations, legal-action masking and feature-ablation diagnostics. E010 compared four architecture conditions across ten seeds and twenty episodes per seed. E012 compared fast and slow internal variables, recovery and relational inference. E013 examined internal-state/action coupling; its results were smoke-test evidence rather than formal multi-seed statistics.

### Long-term experience learning

P1 investigated temporal prediction and occlusion. MR4 used stable, reversed and repeatedly reversed categories across twenty seeds. MR6 used a prediction-error threshold of 0.45 and a three-step cooldown; evaluation used twenty seeds. MR7 used multiple objects with independently changing relationships. POST02 used 32 seeds for each of three historical-experience mechanisms and retained preregistered practical-effect thresholds.

### Recurrent-state action experiment

CORE-READOUT-01 used a 265-parameter linear readout connected to the recurrent and conceptual states, 20 seeds, 1,000 steps per run and four conditions. Primary endpoints were catch rate and directional alignment. Paired comparisons used Wilcoxon signed-rank tests with Holm correction.

### SELF experiments

SELF-01 used 48 forced motor primitives for body-model calibration. SELF-02 used 96 primary and 56 unseen episodes; provenance was inferred from observable body, visual, action and effort information, and evaluator labels were not provided. SELF-03 used five seeds, 60 training, 20 probe and 20 test trajectories and 70 epochs; the latent model was trained offline on recorded trajectories rather than through the continuous Subject event loop.

### Affective and coordination experiments

AFFECT-02 compared low- and high-affect snapshots restored from a common checkpoint, with memory, predictor and recurrent states controlled by state hashes. COORDINATION-01 integrated modules in one Subject; 32 targeted tests and a 60-step deterministic run were completed. COORDINATION-02C used positive and negative historical transfer counterfactuals. These coordination results were not treated as formal multi-seed behavioural evaluations.

### Mirror experiment

CHRISTUS-MIRROR-01 used a local two-dimensional feature-level mirror environment with a virtual body, first-person and mirror channels, proprioception, real-time and delayed mirrors, other-agent video, hidden markers and evaluator-only truth. Twenty seeds were used. M1 used 30 exploration steps; M2 used 25 steps per condition with randomized order; M3–M5 used 25 familiarization and 25 marker steps. Outcomes included action–mirror correspondence, temporal lag, marker detection, body-part localization, inspection rate and transfer to altered geometry. Confidence intervals used the reported mean and sample standard deviation. Thresholds were recorded before the formal run and retained during analysis.

### Statistical analysis

All numerical results were taken from the corresponding experimental reports. Paired seed-level comparisons were used where available. Confidence intervals, standard deviations, effect sizes and corrected significance values follow the original procedures. No new statistical tests were performed for this manuscript. Results from separate environments were not treated as interchangeable measures of general intelligence or consciousness.

### Data availability

The programme contains raw CSV, JSON and JSONL files, manifests, checkpoints, statistical outputs and reproducibility records. A consolidated public archive has not yet been established; the final submission should include a versioned public archive with the data required to reproduce each experiment. Code and derived data are available from the corresponding author on reasonable request.

### Code availability

The implementation and experiment-specific source code are maintained in the local research project. Exact source revisions should be deposited in a public version-controlled repository before publication.

### Ethics statement

All experiments were computational. No human participants or animal subjects were involved.

### Acknowledgements

The author acknowledges the assistance of AI-based research and software-development tools during implementation, experiment planning and manuscript preparation.

### Funding

No specific funding was received for this work.

### Author contributions

P.X. conceived the theoretical framework, designed the research, implemented the architecture and experiments, analysed the results and prepared the manuscript. The author is the sole human contributor listed.

### Competing interests

The author declares no competing interests.

## References

1. Butlin, P. et al. Consciousness in artificial intelligence: insights from the science of consciousness. arXiv 2308.08708 (2023). https://doi.org/10.48550/arXiv.2308.08708
2. Butlin, P. et al. Identifying indicators of consciousness in AI systems. Trends Cogn. Sci. 30, 488–501 (2026). https://doi.org/10.1016/j.tics.2025.10.011
3. Cogitate Consortium et al. Adversarial testing of global neuronal workspace and integrated information theories of consciousness. Nature 642, 133–142 (2025). https://doi.org/10.1038/s41586-025-08888-1
4. Dehaene, S. & Changeux, J.-P. Experimental and theoretical approaches to conscious processing. Neuron 70, 200–227 (2011). https://doi.org/10.1016/j.neuron.2011.03.018
5. Lamme, V. A. F. Towards a true neural stance on consciousness. Trends Cogn. Sci. 10, 494–501 (2006). https://doi.org/10.1016/j.tics.2006.09.001
6. Tononi, G., Boly, M., Massimini, M. & Koch, C. Integrated information theory: from consciousness to its physical substrate. Nat. Rev. Neurosci. 17, 450–461 (2016). https://doi.org/10.1038/nrn.2016.44
7. Baars, B. J. A Cognitive Theory of Consciousness. Cambridge University Press (1988).
8. Baars, B. J. Global workspace theory of consciousness: toward a cognitive neuroscience of human experience. Prog. Brain Res. 150, 45–53 (2005). https://doi.org/10.1016/S0079-6123(05)50004-9
9. Baars, B. J. & Franklin, S. Consciousness is computational: the LIDA model of global workspace theory. Int. J. Mach. Conscious. 1 (2009). https://doi.org/10.1142/S1793843009000050
10. Seth, A. K. Interoceptive inference, emotion, and the embodied self. Trends Cogn. Sci. 17, 565–573 (2013). https://doi.org/10.1016/j.tics.2013.09.007
11. Seth, A. K. & Friston, K. J. Active interoceptive inference and the emotional brain. Phil. Trans. R. Soc. B 371, 20160007 (2016). https://doi.org/10.1098/rstb.2016.0007
12. Friston, K. The free-energy principle: a unified brain theory? Nat. Rev. Neurosci. 11, 127–138 (2010). https://doi.org/10.1038/nrn2787
13. Friston, K., Daunizeau, J., Kilner, J. & Kiebel, S. Action and behavior: a free-energy formulation. Biol. Cybern. 102, 227–260 (2010). https://doi.org/10.1007/s00422-010-0364-z
14. Gallup, G. G. Jr. Chimpanzees: self-recognition. Science 167, 86–87 (1970). https://doi.org/10.1126/science.167.3914.86
15. Kohda, M., Sogawa, S. & Bshary, R. On the mirror test and the evolutionary origin of self-awareness in vertebrates. Phil. Trans. R. Soc. B 380, 20240312 (2025). https://doi.org/10.1098/rstb.2024.0312
16. Sogawa, S. Study of mirror self-recognition: animals' self-awareness and its evolution. Comp. Physiol. Biochem. 42, 57–62 (2025). https://doi.org/10.3330/hikakuseiriseika.42.57
17. Kalckert, A. & Ehrsson, H. H. The moving rubber hand illusion revisited. Conscious. Cogn. 26, 117–132 (2014). https://doi.org/10.1016/j.concog.2014.02.003
18. Tsakiris, M., Carpenter, L., James, D. & Fotopoulou, A. Hands only illusion. Exp. Brain Res. 204, 343–352 (2010). https://doi.org/10.1007/s00221-009-2039-3
19. Amsterdam, B. Mirror self-image reactions before age two. Dev. Psychobiol. 5, 297–305 (1972). https://doi.org/10.1002/dev.420050403
20. McClelland, J. L., McNaughton, B. L. & O'Reilly, R. C. Why there are complementary learning systems in the hippocampus and neocortex. Psychol. Rev. 102, 419–457 (1995). https://doi.org/10.1037/0033-295X.102.3.419
21. Sun, W. et al. Organizing memories for generalization in complementary learning systems. Nat. Neurosci. (2023). https://doi.org/10.1038/s41593-023-01382-9
22. Bonnet, D. et al. Bayesian continual learning and forgetting in neural networks. Nat. Commun. 16, 9614 (2025). https://doi.org/10.1038/s41467-025-64601-w
23. Beer, R. D. On the dynamics of small continuous-time recurrent neural networks. Adapt. Behav. 3, 469–509 (1995). https://doi.org/10.1177/105971239500300405
24. Heidegger, M. Being and Time (trans. Macquarrie, J. & Robinson, E.) (Harper & Row, 1962).
25. Wittgenstein, L. Philosophical Investigations (trans. Anscombe, G. E. M.) (Blackwell, 1953).
26. Gallese, V., Ferrari, P. F. & Umiltà, M. A. The mirror matching system: a shared manifold for intersubjectivity. Behav. Brain Sci. 25, 35–36 (2002). https://doi.org/10.1017/S0140525X02370018
27. Gallese, V. Embodied simulation: from neurons to phenomenal experience. Phenomenol. Cogn. Sci. 4, 23–48 (2005).
28. Damasio, A. R. The Feeling of What Happens: Body and Emotion in the Making of Consciousness. Harcourt (1999).
29. Nagel, T. What is it like to be a bat? Philos. Rev. 83, 435–450 (1974).
30. Chalmers, D. J. Facing up to the problem of consciousness. J. Conscious. Stud. 2, 200–219 (1995).
31. Block, N. On a confusion about a function of consciousness. Behav. Brain Sci. 18, 227–247 (1995).
32. Clark, A. & Chalmers, D. The extended mind. Analysis 58, 7–19 (1998).
33. Maturana, H. R. & Varela, F. J. Autopoiesis and Cognition: The Realization of the Living. D. Reidel (1980).
34. Jiang, S.-Q. et al. Self Model for Embodied Artificial Intelligence. J. Comput. Sci. Technol. 41, 548–563 (2026). https://doi.org/10.1007/s11390-026-6289-3
35. Yang, Q., Wang, L., Wicker, J. & Dobbie, G. Continual learning: a systematic literature review. Neural Netw. 195, 108226 (2026). https://doi.org/10.1016/j.neunet.2025.108226
36. Wang, Z., Yang, E., Shen, L. & Huang, H. A comprehensive survey of forgetting in deep learning beyond continual learning. IEEE Trans. Pattern Anal. Mach. Intell. (2025). https://doi.org/10.1109/TPAMI.2024.3498346

## Research tables

### Table 1 | Integrated cognition (E010)

| Condition | Mean reward | Success rate |
|---|---:|---:|
| A0 historical Q baseline | 4.607 | 0.590 |
| A1 state and meaning | 16.602 | 1.000 |
| A2 A1 + language | 16.182 | 1.000 |
| A3 complete architecture | 16.478 | 1.000 |

Paired complete-vs-A0 difference: 11.871, 95% CI [10.368, 13.374]. Ceiling effects limit component-level inference.

### Table 2 | Temporal prediction (P1)

| Condition | Test MAE |
|---|---:|
| Normal temporal order, eight frames | 3.675 ± 0.462 |
| Shuffled temporal order | 3.770 ± 0.425 |
| Recurrent core, final frame only | 4.095 ± 0.336 |
| No recurrence, eight-frame representation | 3.983 ± 0.489 |
| No recurrence, final frame only | 5.583 ± 0.188 |

### Table 3 | Recurrent-state action readout (CORE-READOUT-01)

| Condition | Catch rate | Directional alignment |
|---|---:|---:|
| A0 readout disabled | 0.277 | — |
| A1 real recurrent-state readout | **0.607** | **0.646** |
| A2 readout input zeroed | 0.277 | — |
| A3 recurrent-state dimensions shuffled | 0.329 | 0.491 |

A1 vs A0: +0.330, 95% CI [0.284, 0.375], Holm-corrected P = 3.53 × 10⁻⁴.

### Table 4 | SELF and affect summary

| Experiment | Key result | Status |
|---|---|---|
| SELF-01 body model | Δpaddle = −0.2533 + 1.5059a; 77 samples; 75 updates | Functional prediction; no action effect |
| SELF-02 provenance | 0.8333 self/external; 0.8333 intention/execution; 0.7143 unseen | Functional provenance |
| SELF-02 memory | 96 autobiographical refs; 1.0 referenced provenance | PASS, rule-based |
| SELF-03 latent self | 0.6355 source probe; baseline 1.0; negative self-state R² | PARTIAL |
| AFFECT-02 | attention Δ0.000464; recurrent Δ9.54 × 10⁻⁵; action Δ0 | Partial; no final action |

### Table 5 | Mirror temporal signatures (M2)

| Condition | Evaluator lag peak | Lag correlation | Online sync |
|---|---:|---:|---:|
| Real-time mirror | 1 | 0.875 | 0.210 |
| 0.5-s delay | 5 | 0.877 | 0.223 |
| 2-s delay | 9 | 0.789 | 0.236 |
| Other-agent video | none | 0.000 | 0.000 |
| No mirror | none | 0.000 | 0.000 |

### Table 6 | Mirror hidden-marker test and control conditions (M3–M5)

| Condition | Marker seen | Inspection | Correct contact |
|---|---:|---:|---:|
| Real own marker | 1.000 | 0.646 | 1.000 |
| Unseen geometry | 1.000 | 0.642 | 1.000 |
| No marker | 0.000 | 0.000 | 0.000 |
| Other-agent marker | 0.000 | 0.000 | 0.000 |

M3 real-marker contact and localization were 1.000; the result was replicated under altered geometry.
