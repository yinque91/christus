# CHRISTUS-SELF-02 Scientific Assessment

## 1. Answers to the eight questions

| # | question | answer |
|---:|---|---|
| 1 | 区分自身主动行为与外部作用？ | **YES, functional**: E1 accuracy 0.8333, macro F1 0.9091, false self attribution 0.0 |
| 2 | 区分行动意图与实际执行结果？ | **YES, functional**: E2 0.8333; blocked attempts retained |
| 3 | 区分身体变化与观察参考系变化？ | **YES, functional**: E3 0.8333 on clear episodes |
| 4 | 将不同来源经历归入同一主体历史？ | **YES, functional autobiographical**: A2 96 references, provenance accuracy 1.0 |
| 5 | 保留失败、被动、观察经历？ | **YES**: all four classes recorded; blocked/passive/observed retained |
| 6 | 长期运行/恢复后保持来源一致？ | **YES, persistence**: E5 history IDs and replay predictions equal |
| 7 | 迁移到未见情境？ | **PARTIAL**: E6 accuracy 0.7143, macro F1 0.8333, abstention 0.2857; all non-abstained predictions correct |
| 8 | 现有架构是否具备进一步自我表征基础？ | **YES, necessary foundation**: Subject, event bus, BodySelfModel, MemoryStore, autobiographical memory and checkpoint are sufficient for functional provenance; **not sufficient to claim subjective self** |

## 2. Functional vs subjective

### Functional provenance

PASS on clear episodes:

- self vs external;
- attempt vs execution;
- body vs observation;
- failure retention;
- provenance-sensitive autobiographical reference.

### Autobiographical memory

A2 stores full `ExperienceRecord` JSON with source, confidence, transition and
memory reference. A1 stores records but no autobiographical links. A0 has
neither.

### Self-representation

The system maintains a **same-subject history** across events and checkpoint
restore. It is a functional continuity of records, not a demonstrated
phenomenological self.

### Subjective self

**NOT TESTED / NOT CLAIMED.** This experiment cannot establish subjective
experience, body ownership, or consciousness.

## 3. Important limitations

- Source inference is a deterministic observable rule, not a learned model.
  A1/A2 therefore have identical predictions.
- A2's advantage is memory provenance, not better source classification.
- The episodes are synthetic/controlled Session-style events, not natural
  human social experience.
- A/B body magnitudes are matched by construction; E6 tests new directions,
  magnitudes and frame shifts but still within the same environment family.
- Weak episodes are explicitly `UNCERTAIN`, so abstention is intentional.
- No final-action optimization is used as success criterion.

## 4. Status statement

```text
FUNCTIONAL PROVENANCE              PASS
AUTOBIOGRAPHICAL PROVENANCE        PASS (A2)
SELF-REPORT / SUBJECTIVE SELF      NOT CLAIMED
TEMPORAL RECORD CONTINUITY         PASS
E6 GENERALIZATION                  PARTIAL
```

**Conclusion:** Christus now has a functional experience-provenance and
autobiographical-memory layer that distinguishes self action, external
happening, observation, and blocked attempt, retains them under one subject
history, and restores provenance after checkpoint. It does not establish
subjective selfhood.
