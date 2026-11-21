"""Pre-registered V0.2 experiment specifications.

These specs are frozen before any run: changing a seed, a condition or a metric
means a new experiment id (E001-v2), not an edit of the old one.
"""

from __future__ import annotations

from wphr.experiments.registry import ExperimentSpec, register

E001 = register(
    ExperimentSpec(
        experiment_id="E001",
        version="0.2.0",
        title="行动反馈是否产生可迁移的线索—动作规则",
        hypothesis=(
            "在相同地图、相同步数、相同探索率的条件下，使用真实行动结果更新线索—动作价值的组"
            "在未见地图上的平均奖励高于禁用线索学习的对照组。"
        ),
        primary_metric="eval_reward_mean",
        secondary_metrics=[
            "eval_success",
            "eval_penalty",
            "q_food_interact",
            "q_hazard_interact",
            "learned_updates",
        ],
        conditions=["learned", "frozen"],
        seeds=[0, 1, 2, 3, 4],
        episodes_per_seed=6,
        steps_per_episode=180,
        exclusion_rules=["core_faults > 0 的运行整体作废"],
        stopping_rule="固定 5 个种子 × 2 个条件，不做中途停止",
        notes="对照组禁用 practice.learn_cues，其余完全一致。",
        runner="e001",
        kwargs={"conditions": {"learned": [], "frozen": ["practice.learn_cues=false"]}},
    )
)

E002 = register(
    ExperimentSpec(
        experiment_id="E002",
        version="0.2.0",
        title="规则反转后旧知识能否被新证据修正",
        hypothesis=(
            "学会 blue 开关安全之后，把世界规则反转为 orange 安全，"
            "仍允许学习的组在反转阶段的奖励恢复量大于禁用学习的对照组。"
        ),
        primary_metric="reversed_eval_reward",
        secondary_metrics=[
            "transfer_reward",
            "transfer_success",
            "reversed_recovery",
            "reversed_eval_penalty",
            "q_switch_blue_interact",
            "q_switch_orange_interact",
        ],
        conditions=["learned", "frozen"],
        seeds=[0, 1, 2, 3, 4],
        episodes_per_seed=6,
        steps_per_episode=180,
        exclusion_rules=["core_faults > 0 的运行整体作废"],
        stopping_rule="固定 5 个种子 × 2 个条件",
        notes=(
            "反转阶段共 6 个 block × 220 步，epsilon=0.25；正式指标为反转后贪心评估的平均奖励。"
            "此前一次 pilot（4 block、epsilon=0.10、指标为 block_last-block_0）因指标被噪声主导而作废，"
            "原因记录在 V0.2 报告中。"
        ),
        runner="e002",
        kwargs={"conditions": {"learned": [], "frozen": ["practice.learn_cues=false"]}},
    )
)

E003 = register(
    ExperimentSpec(
        experiment_id="E003",
        version="0.3.0",
        title="记忆不确定性驱动探索能否缓解过度固化",
        hypothesis=(
            "在同样的规则反转任务里，把语义记忆的置信度衰减接入每线索探索率的组，"
            "在反转后的贪心评估奖励高于把该机制关掉的对照组（后者仍保留线索学习）。"
        ),
        primary_metric="reversed_eval_reward",
        secondary_metrics=[
            "transfer_reward",
            "reversed_recovery",
            "reversed_eval_penalty",
            "q_switch_orange_interact",
        ],
        conditions=["memory_uncertainty", "no_memory_uncertainty"],
        seeds=[0, 1, 2, 3, 4],
        episodes_per_seed=6,
        steps_per_episode=180,
        exclusion_rules=["core_faults > 0 的运行整体作废"],
        stopping_rule="固定 5 个种子 × 2 个条件",
        notes=(
            "两组都保留 practice.learn_cues=true，唯一差别是 memory.semantic.uncertainty_gain "
            "（0.5 对 0.0）。目的是单独检验 E002 中暴露的过度固化机制。"
        ),
        runner="e002",
        kwargs={
            "conditions": {
                "memory_uncertainty": ["memory.semantic.uncertainty_gain=0.5"],
                "no_memory_uncertainty": ["memory.semantic.uncertainty_gain=0.0"],
            }
        },
    )
)

E004 = register(
    ExperimentSpec(
        experiment_id="E004",
        version="0.4.0",
        title="功能性情调制是否带来可测量且稳定的行为差异",
        hypothesis=(
            "在相同任务与相同线索学习条件下，完整情绪调制的组在未见地图上的平均奖励与"
            "行为多样性（动作熵）高于把所有通道冻结在基线的消融组。"
        ),
        primary_metric="eval_reward_mean",
        secondary_metrics=["action_entropy", "cue_coverage", "affect_min", "affect_max", "eval_success"],
        conditions=["full", "none"],
        seeds=[0, 1, 2, 3, 4],
        episodes_per_seed=6,
        steps_per_episode=180,
        exclusion_rules=["core_faults > 0 的运行整体作废"],
        stopping_rule="固定 5 个种子 × 2 个条件",
        notes=(
            "ablation=none 时七个通道固定在 init、探索率固定在 base，五条因果通路全部中性化；"
            "同时记录通道取值区间以检查是否锁死在边界。"
        ),
        runner="e004",
        kwargs={"conditions": {"full": ["affect.ablation=full"], "none": ["affect.ablation=none"]}},
    )
)

E005 = register(
    ExperimentSpec(
        experiment_id="E005",
        version="0.5.0",
        title="哪类证据足以形成正确概念（共现 / 行动 / 完整 / 预设标签）",
        hypothesis=(
            "只有把自身行动的后果归因到线索上的条件，才能在未见地图上形成正确概念；"
            "纯共现条件停留在随机水平；完整条件还能在规则反转后被反例修正。"
        ),
        primary_metric="belief_correct",
        secondary_metrics=["belief_revised", "splits", "revisions", "interventions"],
        conditions=["cooccurrence", "action", "full", "preset"],
        seeds=[0, 1, 2, 3, 4],
        episodes_per_seed=6,
        steps_per_episode=180,
        exclusion_rules=["core_faults > 0 的运行整体作废"],
        stopping_rule="固定 5 个种子 × 4 个条件",
        notes=(
            "preset 条件把正确答案直接注入概念图，只作为上限参照；cooccurrence 只记录线索共现，"
            "不把行动结果归因给线索。belief_correct 表示 belief(blue) > belief(orange)。"
        ),
        runner="e005",
        kwargs={
            "conditions": {
                "cooccurrence": ["concepts.mode=cooccurrence"],
                "action": ["concepts.mode=action"],
                "full": ["concepts.mode=full"],
                "preset": ["concepts.mode=preset"],
            }
        },
    )
)

E006 = register(
    ExperimentSpec(
        experiment_id="E006",
        version="0.6.0",
        title="语言陈述能否被亲身证据推翻（世界—语言双向）",
        hypothesis=(
            "当教师错误地声称 blue 安全而世界实际奖励 orange 时，只有同时拥有行动证据与语言输入的"
            "共同演化条件会依据亲身观测推翻该陈述；仅语言条件会相信错误陈述；仅世界条件没有可推翻的陈述。"
        ),
        primary_metric="evidence_over_claim",
        secondary_metrics=["naming_accuracy", "claims_seen", "productions", "belief_blue", "belief_orange"],
        conditions=["world_only", "language_only", "co_evolution"],
        seeds=[0, 1, 2, 3, 4],
        episodes_per_seed=6,
        steps_per_episode=180,
        exclusion_rules=["core_faults > 0 的运行整体作废"],
        stopping_rule="固定 5 个种子 × 3 个条件",
        notes=(
            "教师可靠性 0.7，每 25 步陈述一次；世界规则为 orange 安全（教师故意说反）。"
            "language_only 把 concepts.mode 设为 cooccurrence，使其无法把行动结果归因给线索。"
        ),
        runner="e006",
        kwargs={
            "conditions": {
                "world_only": ["language.teacher_enabled=false", "concepts.mode=full"],
                "language_only": ["language.teacher_enabled=true", "concepts.mode=cooccurrence"],
                "co_evolution": ["language.teacher_enabled=true", "concepts.mode=full"],
            }
        },
    )
)

E007 = register(
    ExperimentSpec(
        experiment_id="E007",
        version="0.7.0",
        title="他者行为模型是否优于频率基线",
        hypothesis=(
            "另一个模拟主体具有方向持续性（0.8 概率延续上一动作），因此对它的一步动作预测，"
            "基于转入计数的转移模型准确率高于始终预测最频繁动作的频率基线。"
        ),
        primary_metric="prediction_accuracy",
        secondary_metrics=[
            "frequency_accuracy",
            "predictions",
            "interactions",
            "transitions_learned",
            "intent_confidence",
        ],
        conditions=["transition", "frequency"],
        seeds=[0, 1, 2, 3, 4],
        episodes_per_seed=6,
        steps_per_episode=180,
        exclusion_rules=["core_faults > 0 的运行整体作废"],
        stopping_rule="固定 5 个种子 × 2 个条件",
        notes=(
            "两组看到同一个模拟主体与同一批地图，唯一差别是 social.mode。"
            "frequency 条件自身就是频率基线，因此该条件可作为对照。"
        ),
        runner="e007",
        kwargs={
            "conditions": {
                "transition": ["social.mode=transition"],
                "frequency": ["social.mode=frequency"],
            }
        },
    )
)

E008 = register(
    ExperimentSpec(
        experiment_id="E008",
        version="0.8.0",
        title="目标导向的计划与失败恢复是否改善资源恢复",
        hypothesis=(
            "在每幕起始能量 25% 的资源压力下，允许 M7 目标驱动、由规划器组合短计划的组，"
            "在未见地图上的平均奖励高于完全交给上下文策略的组。"
        ),
        primary_metric="eval_reward_mean",
        secondary_metrics=[
            "eval_success",
            "plans_made",
            "plans_completed",
            "plan_detours",
            "plan_failures",
            "energy_end",
        ],
        conditions=["planning", "no_planning"],
        seeds=[0, 1, 2, 3, 4],
        episodes_per_seed=6,
        steps_per_episode=180,
        exclusion_rules=["core_faults > 0 的运行整体作废"],
        stopping_rule="固定 5 个种子 × 2 个条件",
        notes=(
            "规划器是符号层（不是学习得到的）：为当前目标组合动作原语，逐步监测，被阻挡时改道，"
            "连续失败则放弃并重规划。目标选择仍由 M7 负责。"
        ),
        runner="e008",
        kwargs={
            "conditions": {
                "planning": ["planning.enabled=true"],
                "no_planning": ["planning.enabled=false"],
            }
        },
    )
)

E009 = register(
    ExperimentSpec(
        experiment_id="E009",
        version="0.9.0",
        title="模块消融矩阵：逐个关掉一个机制",
        hypothesis=(
            "在同一任务上逐个关闭情绪调制、分层记忆、概念模块、语言与语言通道、社会模块，"
            "完整系统在未见地图上的平均奖励与预测误差应不劣于各消融组；若某个消融组更好，"
            "则说明该机制在本任务上是负担，必须如实报告。"
        ),
        primary_metric="eval_reward_mean",
        secondary_metrics=[
            "eval_success",
            "eval_prediction_error",
            "disabled",
        ],
        conditions=["full", "no_affect", "no_memory", "no_concepts", "no_language", "no_social"],
        seeds=[0, 1, 2, 3, 4],
        episodes_per_seed=6,
        steps_per_episode=180,
        exclusion_rules=["core_faults > 0 的运行整体作废"],
        stopping_rule="固定 5 个种子 × 6 个条件",
        notes=(
            "no_affect 用 affect.ablation=none；no_memory 用 memory.enabled=false；"
            "no_concepts/no_social 用 modules.disabled；no_language 用 language.enabled=false。"
            "这是消融矩阵，不是为每个模块辩护；负结果同样保留。"
        ),
        runner="e009",
        kwargs={
            "conditions": {
                "full": [],
                "no_affect": ["affect.ablation=none"],
                "no_memory": ["memory.enabled=false"],
                "no_concepts": ["modules.disabled=concept_language"],
                "no_language": ["language.enabled=false"],
                "no_social": ["modules.disabled=social"],
            }
        },
    )
)

__all__ = ["E001", "E002", "E003", "E004", "E005", "E006", "E007", "E008", "E009"]
